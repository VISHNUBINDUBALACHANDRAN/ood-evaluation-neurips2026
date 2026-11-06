"""
run_ahss_evaluation.py — Phase 0: AHSS Proof-of-Concept Validation.

Validates the core AHSS hypothesis:
  ID data → low sensitivity to head masking (redundant encoding)
  OOD data → high sensitivity to head masking (fragile encoding)

Uses EXISTING trained models and saved AHM masks to compute AHSS scores
without any new training. This is a fast check.

Strategy:
  1. Load saved AHM masks (5 per experiment) + generate additional random masks
  2. For each split (eval-ID, OOD): extract unmasked + per-mask embeddings
  3. Compute AHSS(x) = (1/T) Σ ||f(x;M_t) - f(x;1)||²
  4. Evaluate: AUROC(AHSS), FPR@95(AHSS)
  5. Compare with Mahalanobis baseline
  6. Test combined scoring: α*Mahalanobis + (1-α)*(-AHSS)

Pass criterion: AHSS AUROC > 0.85 on at least 2/4 experiments.

Usage:
    python run_ahss_evaluation.py \
        --dataset financedocs \
        --experiment_type intra \
        --seed 42

    # Run all 4 experiments in sequence:
    python run_ahss_evaluation.py --run_all
"""

import argparse
import os
import sys
import logging
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.compat  # noqa: F401 — datasets 3.6.0 compatibility

from configs.config import get_default_config, get_label2id, get_id2label
from models.layoutlmv3_classifier import (
    LayoutLMv3DocumentClassifier,
    extract_embeddings_with_multiple_masks,
)
from models.attention_masking import AttentionHeadMasker
from ood_methods.sensitivity_based import AHSSScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from evaluation.metrics import evaluate_ood_method, compute_auroc, compute_fpr_at_tpr
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ahss_eval")


def load_masks(ahm_dir: str, masker: AttentionHeadMasker, num_extra: int = 20,
               masking_percentages=None, seed: int = 42):
    """
    Load saved AHM masks and optionally generate additional random masks.

    Returns:
        masks: list of (num_layers, num_heads) tensors
        mask_sources: list of str ("ahm_best_N" or "random_N")
    """
    if masking_percentages is None:
        masking_percentages = [0.1, 0.2]

    masks = []
    sources = []

    # Load saved AHM masks
    for i in range(100):  # upper bound
        path = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if os.path.exists(path):
            mask = masker.load_mask(path)
            masks.append(mask)
            sources.append(f"ahm_best_{i}")
        else:
            break

    logger.info(f"Loaded {len(masks)} AHM masks from {ahm_dir}")

    # Generate additional random masks
    if num_extra > 0:
        rng = np.random.RandomState(seed + 9999)  # different from AHM seeds
        per_pct = num_extra // len(masking_percentages)
        remainder = num_extra % len(masking_percentages)
        for p_idx, p in enumerate(masking_percentages):
            count = per_pct + (1 if p_idx < remainder else 0)
            for j in range(count):
                mask = masker.create_mask(
                    masking_percentage=p,
                    seed=int(rng.randint(0, 2**31)),
                )
                masks.append(mask)
                sources.append(f"random_p{p}_{j}")

        logger.info(f"Generated {num_extra} additional random masks")

    return masks, sources


def run_ahss_with_precomputed_embeddings(
    embedding_dir: str,
    ahm_dir: str,
    config,
    seed: int = 42,
    num_extra_masks: int = 20,
    model_path: str = None,
    data_path: str = None,
    ood_data_path: str = None,
    dataset_name: str = "financedocs",
    experiment_type: str = "intra",
):
    """
    Run AHSS evaluation.

    This function supports two modes:
    A) Fast mode: Use existing AHM mask embeddings (no model loading)
       - Uses the 5 saved AHM masks' embeddings that already exist
       - Treats the unmasked embedding as the reference
       - Very fast but limited to 5 masks

    B) Full mode: Load model and extract embeddings under many masks
       - Loads fine-tuned model from checkpoint
       - Extracts embeddings under 25+ masks
       - Slower but more comprehensive

    We'll try Fast mode first, fall back to Full mode if model_path provided.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    masker = AttentionHeadMasker(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )

    # Load precomputed standard embeddings (unmasked)
    eval_unmasked = np.load(os.path.join(embedding_dir, "eval_embeddings.npy"))
    ood_unmasked = np.load(os.path.join(embedding_dir, "ood_embeddings.npy"))
    train_unmasked = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))
    eval_labels = np.load(os.path.join(embedding_dir, "eval_labels.npy"))

    logger.info(f"Loaded unmasked embeddings: train={train_unmasked.shape}, "
                f"eval={eval_unmasked.shape}, ood={ood_unmasked.shape}")

    # ====================================================================
    # Try Full mode: extract embeddings under many masks via model
    # ====================================================================
    if model_path and os.path.exists(model_path):
        logger.info("=== FULL MODE: Loading model for multi-mask extraction ===")

        dataset_config = config.financedocs if dataset_name == "financedocs" else config.tobacco
        if experiment_type == "intra":
            id_classes = dataset_config.id_classes
        else:
            id_classes = dataset_config.all_classes

        label2id = get_label2id(id_classes)
        id2label = get_id2label(id_classes)
        num_labels = len(id_classes)

        model = LayoutLMv3DocumentClassifier(
            model_name=model_path,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        model.to(device)
        model.eval()

        # Load masks (AHM + random)
        masks, mask_sources = load_masks(
            ahm_dir, masker,
            num_extra=num_extra_masks,
            masking_percentages=config.ahss.masking_percentages,
            seed=seed,
        )

        # Load dataset and create dataloaders
        from datasets import load_from_disk
        from sklearn.model_selection import train_test_split

        full_ds = load_from_disk(data_path)

        if experiment_type == "intra":
            id_ds = filter_dataset_by_classes(full_ds, id_classes)
            ood_ds = filter_dataset_by_classes(full_ds, [dataset_config.ood_class])
        else:
            id_ds = filter_dataset_by_classes(full_ds, id_classes)
            # For cross-dataset, OOD is the OTHER dataset
            if ood_data_path and os.path.exists(ood_data_path):
                ood_ds = load_from_disk(ood_data_path)
            else:
                # Try standard paths
                base = os.path.dirname(os.path.abspath(data_path))
                if dataset_name == "financedocs":
                    cross_path = os.path.join(base, "tobacco", "colated_tensors")
                else:
                    # tobacco → financedocs (parent dir has colated_tensors)
                    cross_path = os.path.join(base, "colated_tensors")
                if os.path.exists(cross_path):
                    ood_ds = load_from_disk(cross_path)
                    logger.info(f"Cross-dataset OOD loaded from: {cross_path}")
                else:
                    logger.warning(f"Cross-dataset OOD not found at {cross_path}")
                    ood_ds = None

        id_ds = encode_labels(id_ds, label2id)
        labels_arr = np.array(id_ds["labels"])
        indices = np.arange(len(id_ds))
        train_idx, eval_idx = train_test_split(
            indices, train_size=config.train_eval_split,
            random_state=seed, stratify=labels_arr,
        )
        eval_ds = id_ds.select(eval_idx)
        eval_dl = create_dataloader(eval_ds, batch_size=8, shuffle=False)

        if ood_ds is not None:
            ood_dummy_labels = {lbl: num_labels for lbl in set(ood_ds["labels"])}
            ood_ds = encode_labels(ood_ds, ood_dummy_labels)
            ood_dl = create_dataloader(ood_ds, batch_size=8, shuffle=False)

        # Extract embeddings under all masks
        logger.info(f"Extracting eval embeddings under {len(masks)} masks...")
        eval_unmasked_new, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
            model, eval_dl, masks, device=device,
            desc="Eval multi-mask",
        )

        ood_masked_list = None
        ood_unmasked_new = None
        if ood_ds is not None:
            logger.info(f"Extracting OOD embeddings under {len(masks)} masks...")
            ood_unmasked_new, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
                model, ood_dl, masks, device=device,
                desc="OOD multi-mask",
            )

        # Use freshly extracted unmasked embeddings (should match precomputed)
        if eval_unmasked_new is not None:
            eval_unmasked = eval_unmasked_new
        if ood_unmasked_new is not None:
            ood_unmasked = ood_unmasked_new

        del model
        torch.cuda.empty_cache()

    else:
        # ====================================================================
        # Fast mode: construct pseudo-masked embeddings from AHM saved data
        # ====================================================================
        logger.info("=== FAST MODE: Using saved AHM embeddings (no model needed) ===")
        logger.info("Loading AHM-masked embeddings as pseudo-mask samples...")

        # The AHM pipeline ensembles top-5 masks into a single embedding.
        # For AHSS, we need per-mask embeddings. We DON'T have those saved
        # individually. But we have the ensemble and the unmasked embeddings.
        #
        # For a quick Phase 0 test, we need the model to extract per-mask embeddings.
        # Let's check if model_path/data_path are available.

        logger.warning(
            "Fast mode not supported for AHSS — need per-mask embeddings. "
            "Please provide --model_path and --data_path for full extraction."
        )
        logger.info(
            "Hint: --model_path outputs/checkpoints/<experiment>/best_model "
            "--data_path data/<dataset_tensors>"
        )

        # As a VERY rough approximation, we can use the AHM ensemble embedding
        # as a single "masked" embedding and compute displacement from unmasked.
        # This gives AHSS with T=1, which is noisy but can indicate the signal.
        eval_ahm_path = os.path.join(ahm_dir, "eval_ahm_embeddings.npy")
        ood_ahm_path = os.path.join(ahm_dir, "ood_ahm_embeddings.npy")

        if os.path.exists(eval_ahm_path) and os.path.exists(ood_ahm_path):
            eval_ahm = np.load(eval_ahm_path)
            ood_ahm = np.load(ood_ahm_path)
            eval_masked_list = [eval_ahm]
            ood_masked_list = [ood_ahm]
            mask_sources = ["ahm_ensemble"]
            logger.info(
                f"Using AHM ensemble as single mask proxy (T=1). "
                f"eval_ahm={eval_ahm.shape}, ood_ahm={ood_ahm.shape}"
            )
        else:
            logger.error("No AHM embeddings found. Cannot proceed with AHSS evaluation.")
            return None

    # ====================================================================
    # Compute AHSS scores
    # ====================================================================
    logger.info("=" * 60)
    logger.info("COMPUTING AHSS SCORES")
    logger.info("=" * 60)

    scorer = AHSSScorer(normalize=config.ahss.normalize_embeddings)

    # AHSS for eval (ID) data
    eval_ahss_raw = scorer.score_raw(eval_unmasked, eval_masked_list)
    eval_ahss = scorer.score(eval_unmasked, eval_masked_list)

    # AHSS for OOD data
    ood_ahss_raw = scorer.score_raw(ood_unmasked, ood_masked_list)
    ood_ahss = scorer.score(ood_unmasked, ood_masked_list)

    logger.info(f"AHSS (raw) — ID mean: {eval_ahss_raw.mean():.6f}, "
                f"std: {eval_ahss_raw.std():.6f}")
    logger.info(f"AHSS (raw) — OOD mean: {ood_ahss_raw.mean():.6f}, "
                f"std: {ood_ahss_raw.std():.6f}")
    logger.info(f"AHSS separation ratio (OOD/ID mean): "
                f"{ood_ahss_raw.mean() / (eval_ahss_raw.mean() + 1e-10):.2f}x")

    # ====================================================================
    # Evaluate AHSS as OOD detector
    # ====================================================================
    results = []

    # Pure AHSS
    ahss_result = evaluate_ood_method(eval_ahss, ood_ahss, "AHSS")
    results.append(ahss_result)
    logger.info(f"AHSS → AUROC: {ahss_result['auroc']:.4f}, "
                f"FPR@95: {ahss_result['fpr']:.4f}")

    # ====================================================================
    # Mahalanobis baseline (from precomputed unmasked embeddings)
    # ====================================================================
    mah = MahalanobisScorer()
    mah.fit(train_unmasked, train_labels)
    eval_mah = mah.score(eval_unmasked)
    ood_mah = mah.score(ood_unmasked)
    mah_result = evaluate_ood_method(eval_mah, ood_mah, "Mahalanobis")
    results.append(mah_result)
    logger.info(f"Mahalanobis → AUROC: {mah_result['auroc']:.4f}, "
                f"FPR@95: {mah_result['fpr']:.4f}")

    # kNN baseline
    knn = KNNScorer(k=config.ood.knn_k, metric="euclidean")
    knn.fit(train_unmasked)
    eval_knn = knn.score(eval_unmasked)
    ood_knn = knn.score(ood_unmasked)
    knn_result = evaluate_ood_method(eval_knn, ood_knn, "kNN")
    results.append(knn_result)
    logger.info(f"kNN → AUROC: {knn_result['auroc']:.4f}, "
                f"FPR@95: {knn_result['fpr']:.4f}")

    # ====================================================================
    # Combined scoring: AHSS + Mahalanobis
    # ====================================================================
    # FIX: Normalize on concatenated eval+OOD BEFORE splitting, so that
    # the relative ranking across ID and OOD is preserved.
    def _rank_normalize(eval_s, ood_s):
        """Rank-based normalization on concatenated eval+OOD scores."""
        all_s = np.concatenate([eval_s, ood_s])
        # Rank normalization: rank / N → uniform [0, 1]
        ranks = np.argsort(np.argsort(all_s)).astype(np.float64)
        ranks /= (len(ranks) - 1 + 1e-10)
        return ranks[:len(eval_s)], ranks[len(eval_s):]

    logger.info("\n--- Combined Scoring (AHSS + Mahalanobis) ---")
    best_combined_auroc = 0.0
    best_alpha = 0.0

    eval_ahss_rn, ood_ahss_rn = _rank_normalize(eval_ahss, ood_ahss)
    eval_mah_rn, ood_mah_rn = _rank_normalize(eval_mah, ood_mah)

    for alpha in config.ahss.alpha_grid:
        eval_combined = alpha * eval_mah_rn + (1.0 - alpha) * eval_ahss_rn
        ood_combined = alpha * ood_mah_rn + (1.0 - alpha) * ood_ahss_rn
        auroc = compute_auroc(eval_combined, ood_combined)
        fpr = compute_fpr_at_tpr(eval_combined, ood_combined)

        if auroc > best_combined_auroc:
            best_combined_auroc = auroc
            best_alpha = alpha
            best_combined_result = {
                "method": f"AHSS+Mah(α={alpha:.1f})",
                "auroc": auroc,
                "fpr": fpr,
            }

        logger.info(f"  α={alpha:.1f} → AUROC: {auroc:.4f}, FPR@95: {fpr:.4f}")

    results.append(best_combined_result)
    logger.info(f"\nBest combined: α={best_alpha:.1f}, "
                f"AUROC: {best_combined_auroc:.4f}")

    # ====================================================================
    # Combined: AHSS + kNN
    # ====================================================================
    logger.info("\n--- Combined Scoring (AHSS + kNN) ---")
    best_knn_auroc = 0.0

    eval_knn_rn, ood_knn_rn = _rank_normalize(eval_knn, ood_knn)

    for alpha in config.ahss.alpha_grid:
        eval_combined = alpha * eval_knn_rn + (1.0 - alpha) * eval_ahss_rn
        ood_combined = alpha * ood_knn_rn + (1.0 - alpha) * ood_ahss_rn
        auroc = compute_auroc(eval_combined, ood_combined)
        fpr = compute_fpr_at_tpr(eval_combined, ood_combined)

        if auroc > best_knn_auroc:
            best_knn_auroc = auroc
            best_knn_result = {
                "method": f"AHSS+kNN(α={alpha:.1f})",
                "auroc": auroc,
                "fpr": fpr,
            }

    results.append(best_knn_result)

    # ====================================================================
    # Summary
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"AHSS Phase 0 Results — {dataset_name} {experiment_type} (seed {seed})")
    logger.info("=" * 70)
    logger.info(f"{'Method':<30} {'AUROC':>8} {'FPR@95':>8}")
    logger.info("-" * 48)
    for r in results:
        logger.info(f"{r['method']:<30} {r['auroc']:>8.4f} {r['fpr']:>8.4f}")
    logger.info("-" * 48)

    # Pass check
    ahss_auroc = ahss_result["auroc"]
    if ahss_auroc > 0.85:
        logger.info(f"✓ AHSS AUROC {ahss_auroc:.4f} > 0.85 — PASS")
    else:
        logger.info(f"✗ AHSS AUROC {ahss_auroc:.4f} < 0.85 — NEEDS INVESTIGATION")

    # AHSS statistics for analysis
    stats = {
        "dataset": dataset_name,
        "experiment_type": experiment_type,
        "seed": seed,
        "num_masks": len(eval_masked_list),
        "mask_sources": mask_sources if 'mask_sources' in dir() else ["unknown"],
        "id_ahss_mean": float(eval_ahss_raw.mean()),
        "id_ahss_std": float(eval_ahss_raw.std()),
        "id_ahss_median": float(np.median(eval_ahss_raw)),
        "ood_ahss_mean": float(ood_ahss_raw.mean()),
        "ood_ahss_std": float(ood_ahss_raw.std()),
        "ood_ahss_median": float(np.median(ood_ahss_raw)),
        "separation_ratio": float(ood_ahss_raw.mean() / (eval_ahss_raw.mean() + 1e-10)),
        "results": results,
    }

    return stats


def run_single_experiment(args, config):
    """Run AHSS evaluation for a single experiment configuration."""
    dataset = args.dataset
    exp_type = args.experiment_type
    seed = args.seed

    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = f"{dataset}_{exp_type}_seed{seed}"

    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(base_dir, "outputs", "ahm_masks", exp_name)

    model_path = None
    if args.model_path:
        model_path = args.model_path
    else:
        default_model = os.path.join(
            base_dir, "outputs", "checkpoints", exp_name, "best_model"
        )
        if os.path.exists(default_model):
            model_path = default_model

    data_path = args.data_path

    if not os.path.exists(embedding_dir):
        logger.error(f"Embedding dir not found: {embedding_dir}")
        return None

    if not os.path.exists(ahm_dir):
        logger.error(f"AHM dir not found: {ahm_dir}")
        return None

    logger.info(f"\n{'='*70}")
    logger.info(f"Running AHSS evaluation: {exp_name}")
    logger.info(f"  Embedding dir: {embedding_dir}")
    logger.info(f"  AHM dir: {ahm_dir}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"{'='*70}")

    start = time.time()
    stats = run_ahss_with_precomputed_embeddings(
        embedding_dir=embedding_dir,
        ahm_dir=ahm_dir,
        config=config,
        seed=seed,
        num_extra_masks=args.num_extra_masks,
        model_path=model_path,
        data_path=data_path,
        ood_data_path=getattr(args, 'ood_data_path', None),
        dataset_name=dataset,
        experiment_type=exp_type,
    )
    elapsed = time.time() - start
    logger.info(f"Completed in {elapsed:.1f}s")

    # Save results
    if stats is not None:
        output_dir = os.path.join(
            base_dir, "outputs", "ahss_results", exp_name
        )
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "ahss_phase0_results.json")
        with open(results_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return stats


def run_all_experiments(args, config):
    """Run AHSS evaluation across all 4 experiment configurations."""
    experiments = [
        ("financedocs", "intra"),
        ("financedocs", "cross"),
        ("tobacco", "intra"),
        ("tobacco", "cross"),
    ]

    all_stats = []
    pass_count = 0

    for dataset, exp_type in experiments:
        args.dataset = dataset
        args.experiment_type = exp_type
        stats = run_single_experiment(args, config)
        if stats is not None:
            all_stats.append(stats)
            if stats["results"][0]["auroc"] > 0.85:  # AHSS result
                pass_count += 1

    # ====================================================================
    # Final summary
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 0 — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<30} {'AHSS AUROC':>12} {'Mah AUROC':>12} {'Best Combo':>12} {'Status':>8}")
    logger.info("-" * 76)

    for s in all_stats:
        exp = f"{s['dataset']}_{s['experiment_type']}"
        ahss_auroc = s["results"][0]["auroc"]
        mah_auroc = s["results"][1]["auroc"]
        best_combo = max(r["auroc"] for r in s["results"])
        status = "PASS" if ahss_auroc > 0.85 else "FAIL"
        logger.info(f"{exp:<30} {ahss_auroc:>12.4f} {mah_auroc:>12.4f} {best_combo:>12.4f} {status:>8}")

    logger.info("-" * 76)
    logger.info(f"Passed: {pass_count}/{len(all_stats)} experiments (need ≥2 for GO)")

    if pass_count >= 2:
        logger.info(">>> DECISION: GO — Proceed to Phase 1 (full AHSS implementation) <<<")
    else:
        logger.info(">>> DECISION: INVESTIGATE — AHSS signal weaker than expected <<<")
        logger.info("    Consider: more masks, different layers, unnormalized AHSS")

    # Save aggregated results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agg_dir = os.path.join(base_dir, "outputs", "ahss_results")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, f"phase0_summary_seed{args.seed}.json")
    with open(agg_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Aggregated results saved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AHSS Phase 0: Proof-of-concept validation"
    )
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model checkpoint (best_model dir)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to preprocessed tensor dataset (HuggingFace Dataset)")
    parser.add_argument("--num_extra_masks", type=int, default=20,
                        help="Number of additional random masks beyond AHM masks")
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD dataset (for cross-dataset experiments)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 4 experiments (financedocs/tobacco × intra/cross)")

    args = parser.parse_args()
    config = get_default_config()

    if args.run_all:
        run_all_experiments(args, config)
    else:
        run_single_experiment(args, config)


if __name__ == "__main__":
    main()
