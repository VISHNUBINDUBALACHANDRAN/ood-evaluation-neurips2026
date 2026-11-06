"""
run_dahm_evaluation.py — Step 1: D-AHM vs AHM vs Random Mask Evaluation.

Validates whether D-AHM (Differentiable Attention Head Masking) produces
masks that improve AHSS-based OOD detection over random and AHM masks.

For each experiment (4 configs × 5 seeds = 20 runs):
  1. Load fine-tuned LayoutLMv3 model
  2. Load D-AHM masks (5), AHM masks (5), generate random masks (25)
  3. Extract embeddings under each mask set
  4. Compute AHSS scores per mask type
  5. Evaluate AUROC, FPR@95 for each
  6. Compare: D-AHM vs AHM vs Random
  7. Also test combined scoring (AHSS + Mahalanobis) per mask type

Usage:
    python run_dahm_evaluation.py --run_all
    python run_dahm_evaluation.py --dataset financedocs --experiment_type intra --seed 42
"""

import argparse
import os
import sys
import logging
import json
import time
import numpy as np
import torch
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import utils.compat  # noqa: F401

from configs.config import get_default_config, get_label2id, get_id2label
from models.layoutlmv3_classifier import (
    LayoutLMv3DocumentClassifier,
    extract_embeddings_with_multiple_masks,
)
from models.attention_masking import AttentionHeadMasker
from ood_methods.sensitivity_based import AHSSScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dahm_eval")


def load_dahm_masks(dahm_dir, masker):
    """Load D-AHM masks from directory. Skip all-ones mask (mask 0 often)."""
    masks = []
    sources = []
    for i in range(100):
        path = os.path.join(dahm_dir, f"dahm_mask_{i}.npy")
        if not os.path.exists(path):
            break
        raw = np.load(path)
        # Skip masks that are all-ones (no actual masking)
        if (raw < 0.5).sum() == 0:
            logger.info(f"  Skipping dahm_mask_{i} (all ones, no masking)")
            continue
        mask = masker.load_mask(path)
        masks.append(mask)
        sources.append(f"dahm_{i}")
    logger.info(f"Loaded {len(masks)} D-AHM masks from {dahm_dir}")
    return masks, sources


def load_ahm_masks(ahm_dir, masker):
    """Load AHM masks from directory."""
    masks = []
    sources = []
    for i in range(100):
        path = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if not os.path.exists(path):
            break
        mask = masker.load_mask(path)
        masks.append(mask)
        sources.append(f"ahm_{i}")
    logger.info(f"Loaded {len(masks)} AHM masks from {ahm_dir}")
    return masks, sources


def generate_random_masks(masker, num_masks=25, masking_pcts=(0.1, 0.2), seed=42):
    """Generate random binary masks."""
    masks = []
    sources = []
    rng = np.random.RandomState(seed + 7777)
    per_pct = num_masks // len(masking_pcts)
    for p_idx, p in enumerate(masking_pcts):
        count = per_pct + (1 if p_idx < num_masks % len(masking_pcts) else 0)
        for j in range(count):
            mask = masker.create_mask(
                masking_percentage=p,
                seed=int(rng.randint(0, 2**31)),
            )
            masks.append(mask)
            sources.append(f"random_p{p}_{j}")
    logger.info(f"Generated {len(masks)} random masks")
    return masks, sources


def extract_embeddings_for_mask_set(model, dataloader, masks, device, desc=""):
    """Extract unmasked + per-mask embeddings for a set of masks."""
    unmasked, masked_list, labels = extract_embeddings_with_multiple_masks(
        model, dataloader, masks, device=device, desc=desc,
    )
    return unmasked, masked_list, labels


def compute_ahss_auroc(eval_unmasked, eval_masked_list, ood_unmasked, ood_masked_list,
                       train_unmasked, train_labels, normalize=True):
    """Compute AHSS AUROC and combined AHSS+Maha AUROC for a mask set."""
    scorer = AHSSScorer(normalize=normalize)

    # Raw AHSS scores (higher = more OOD)
    eval_ahss_raw = scorer.compute_ahss(eval_unmasked, eval_masked_list, normalize=normalize)
    ood_ahss_raw = scorer.compute_ahss(ood_unmasked, ood_masked_list, normalize=normalize)

    # Negated for scoring convention (higher = more ID)
    eval_ahss = -eval_ahss_raw
    ood_ahss = -ood_ahss_raw

    ahss_auroc = compute_auroc(eval_ahss, ood_ahss)
    ahss_fpr = compute_fpr_at_tpr(eval_ahss, ood_ahss)

    # Mahalanobis baseline
    mah = MahalanobisScorer()
    mah.fit(train_unmasked, train_labels)
    eval_mah = mah.score(eval_unmasked)
    ood_mah = mah.score(ood_unmasked)
    mah_auroc = compute_auroc(eval_mah, ood_mah)
    mah_fpr = compute_fpr_at_tpr(eval_mah, ood_mah)

    # Combined: search best alpha
    def _rank_normalize(eval_s, ood_s):
        all_s = np.concatenate([eval_s, ood_s])
        ranks = np.argsort(np.argsort(all_s)).astype(np.float64)
        ranks /= (len(ranks) - 1 + 1e-10)
        return ranks[:len(eval_s)], ranks[len(eval_s):]

    eval_ahss_rn, ood_ahss_rn = _rank_normalize(eval_ahss, ood_ahss)
    eval_mah_rn, ood_mah_rn = _rank_normalize(eval_mah, ood_mah)

    best_alpha = 0.0
    best_combined_auroc = 0.0
    for alpha in np.arange(0.0, 1.05, 0.1):
        ec = alpha * eval_mah_rn + (1.0 - alpha) * eval_ahss_rn
        oc = alpha * ood_mah_rn + (1.0 - alpha) * ood_ahss_rn
        a = compute_auroc(ec, oc)
        if a > best_combined_auroc:
            best_combined_auroc = a
            best_alpha = alpha

    combined_fpr = compute_fpr_at_tpr(
        best_alpha * eval_mah_rn + (1.0 - best_alpha) * eval_ahss_rn,
        best_alpha * ood_mah_rn + (1.0 - best_alpha) * ood_ahss_rn,
    )

    return {
        "ahss_auroc": float(ahss_auroc),
        "ahss_fpr95": float(ahss_fpr),
        "mah_auroc": float(mah_auroc),
        "mah_fpr95": float(mah_fpr),
        "combined_auroc": float(best_combined_auroc),
        "combined_fpr95": float(combined_fpr),
        "combined_best_alpha": float(best_alpha),
        "ahss_id_mean": float(eval_ahss_raw.mean()),
        "ahss_id_std": float(eval_ahss_raw.std()),
        "ahss_ood_mean": float(ood_ahss_raw.mean()),
        "ahss_ood_std": float(ood_ahss_raw.std()),
        "separation_ratio": float(ood_ahss_raw.mean() / (eval_ahss_raw.mean() + 1e-10)),
    }


def run_experiment(dataset, experiment_type, seed, config):
    """Run D-AHM evaluation for a single experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    base_dir = SCRIPT_DIR

    # Paths
    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints", exp_name, "best_model")
    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(base_dir, "outputs", "ahm_masks", exp_name)
    dahm_dir = os.path.join(base_dir, "outputs", "dahm_masks", exp_name)
    output_dir = os.path.join(base_dir, "outputs", "dahm_eval_results", exp_name)

    # Skip if already done
    results_path = os.path.join(output_dir, "dahm_eval_results.json")
    if os.path.exists(results_path):
        logger.info(f"Results already exist: {results_path}, skipping.")
        with open(results_path) as f:
            return json.load(f)

    for d in [embedding_dir, ahm_dir, dahm_dir, checkpoint_dir]:
        if not os.path.exists(d):
            logger.error(f"Missing directory: {d}")
            return None

    logger.info(f"\n{'='*70}")
    logger.info(f"D-AHM EVALUATION: {exp_name}")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Setup masker
    masker = AttentionHeadMasker(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )

    # Load masks
    dahm_masks, dahm_sources = load_dahm_masks(dahm_dir, masker)
    ahm_masks, ahm_sources = load_ahm_masks(ahm_dir, masker)
    random_masks, random_sources = generate_random_masks(
        masker, num_masks=25, masking_pcts=config.ahss.masking_percentages, seed=seed,
    )

    if len(dahm_masks) == 0:
        logger.warning(f"No valid D-AHM masks for {exp_name}, skipping.")
        return None

    # Load precomputed embeddings for reference
    train_unmasked = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))

    # Load model for mask-wise extraction
    dataset_config = config.financedocs if dataset == "financedocs" else config.tobacco
    if experiment_type == "intra":
        id_classes = dataset_config.id_classes
    else:
        id_classes = dataset_config.all_classes

    label2id = get_label2id(id_classes)
    id2label = get_id2label(id_classes)
    num_labels = len(id_classes)

    logger.info(f"Loading model from {checkpoint_dir}")
    model = LayoutLMv3DocumentClassifier(
        model_name=checkpoint_dir,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    model.eval()

    # Load data
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split
    from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

    data_base = os.path.join(base_dir, "data")
    if dataset == "financedocs":
        data_path = os.path.join(data_base, "colated_tensors")
    else:
        data_path = os.path.join(data_base, "tobacco", "colated_tensors")

    full_ds = load_from_disk(data_path)

    if experiment_type == "intra":
        id_ds = filter_dataset_by_classes(full_ds, id_classes)
        ood_ds = filter_dataset_by_classes(full_ds, [dataset_config.ood_class])
    else:
        id_ds = filter_dataset_by_classes(full_ds, id_classes)
        if dataset == "financedocs":
            cross_path = os.path.join(data_base, "tobacco", "colated_tensors")
        else:
            cross_path = os.path.join(data_base, "colated_tensors")
        ood_ds = load_from_disk(cross_path)

    id_ds = encode_labels(id_ds, label2id)
    labels_arr = np.array(id_ds["labels"])
    indices = np.arange(len(id_ds))
    _, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )
    eval_ds = id_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=64, shuffle=False)

    if experiment_type == "cross":
        ood_dummy_labels = {lbl: num_labels for lbl in set(ood_ds["labels"])}
        ood_ds = encode_labels(ood_ds, ood_dummy_labels)
    else:
        ood_dummy_labels = {dataset_config.ood_class: num_labels}
        ood_ds = encode_labels(ood_ds, ood_dummy_labels)
    ood_dl = create_dataloader(ood_ds, batch_size=64, shuffle=False)

    # Extract embeddings under each mask type
    results = {"experiment": exp_name, "dataset": dataset,
               "experiment_type": experiment_type, "seed": seed}

    mask_sets = {
        "dahm": (dahm_masks, dahm_sources),
        "ahm": (ahm_masks, ahm_sources),
        "random": (random_masks, random_sources),
    }

    for mask_type, (masks, sources) in mask_sets.items():
        logger.info(f"\n--- Extracting embeddings for {mask_type} masks ({len(masks)} masks) ---")
        eval_unmasked, eval_masked_list, _ = extract_embeddings_for_mask_set(
            model, eval_dl, masks, device, desc=f"{exp_name} eval {mask_type}",
        )
        _, ood_masked_list, _ = extract_embeddings_for_mask_set(
            model, ood_dl, masks, device, desc=f"{exp_name} OOD {mask_type}",
        )
        ood_unmasked = np.load(os.path.join(embedding_dir, "ood_embeddings.npy"))

        logger.info(f"Computing AHSS for {mask_type} masks...")
        mask_results = compute_ahss_auroc(
            eval_unmasked, eval_masked_list,
            ood_unmasked, ood_masked_list,
            train_unmasked, train_labels,
            normalize=config.ahss.normalize_embeddings,
        )
        mask_results["num_masks"] = len(masks)
        mask_results["mask_sources"] = sources
        results[mask_type] = mask_results

        logger.info(f"  {mask_type}: AHSS AUROC={mask_results['ahss_auroc']:.4f}, "
                     f"Combined AUROC={mask_results['combined_auroc']:.4f}, "
                     f"Separation={mask_results['separation_ratio']:.2f}x")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Summary
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"\n{'='*70}")
    logger.info(f"DAHM EVAL SUMMARY: {exp_name} ({elapsed:.1f}s)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Mask Type':<12} {'#Masks':>6} {'AHSS AUROC':>12} {'Comb AUROC':>12} {'Sep Ratio':>10}")
    logger.info("-" * 54)
    for mt in ["dahm", "ahm", "random"]:
        r = results[mt]
        logger.info(f"{mt:<12} {r['num_masks']:>6} {r['ahss_auroc']:>12.4f} "
                     f"{r['combined_auroc']:>12.4f} {r['separation_ratio']:>10.2f}x")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {results_path}")

    return results


def run_all(config):
    """Run D-AHM evaluation across all experiments and seeds."""
    experiments = [
        ("financedocs", "intra"),
        ("financedocs", "cross"),
        ("tobacco", "intra"),
        ("tobacco", "cross"),
    ]
    seeds = config.seeds

    all_results = []
    for dataset, exp_type in experiments:
        for seed in seeds:
            try:
                r = run_experiment(dataset, exp_type, seed, config)
                if r is not None:
                    all_results.append(r)
            except Exception as e:
                logger.error(f"FAILED {dataset}_{exp_type}_seed{seed}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate results per experiment
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATED D-AHM EVALUATION RESULTS")
    logger.info(f"{'='*80}")

    agg = {}
    for dataset, exp_type in experiments:
        exp_key = f"{dataset}_{exp_type}"
        exp_results = [r for r in all_results if r["dataset"] == dataset
                       and r["experiment_type"] == exp_type]
        if not exp_results:
            continue

        agg[exp_key] = {}
        for mask_type in ["dahm", "ahm", "random"]:
            aurocs = [r[mask_type]["ahss_auroc"] for r in exp_results]
            combined = [r[mask_type]["combined_auroc"] for r in exp_results]
            sep = [r[mask_type]["separation_ratio"] for r in exp_results]
            agg[exp_key][mask_type] = {
                "ahss_auroc_mean": float(np.mean(aurocs)),
                "ahss_auroc_std": float(np.std(aurocs)),
                "combined_auroc_mean": float(np.mean(combined)),
                "combined_auroc_std": float(np.std(combined)),
                "separation_mean": float(np.mean(sep)),
                "n_seeds": len(aurocs),
            }

        logger.info(f"\n{exp_key}:")
        logger.info(f"  {'Mask':<10} {'AHSS AUROC':>20} {'Combined AUROC':>20} {'Sep Ratio':>12}")
        for mt in ["dahm", "ahm", "random"]:
            a = agg[exp_key][mt]
            logger.info(f"  {mt:<10} {a['ahss_auroc_mean']:.4f}±{a['ahss_auroc_std']:.4f}"
                         f"    {a['combined_auroc_mean']:.4f}±{a['combined_auroc_std']:.4f}"
                         f"    {a['separation_mean']:.2f}x")

    # Save aggregated
    agg_path = os.path.join(SCRIPT_DIR, "outputs", "dahm_eval_results", "aggregated_results.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({"per_experiment": agg, "all_results": all_results}, f, indent=2)
    logger.info(f"\nAggregated results saved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="D-AHM vs AHM vs Random Mask Evaluation")
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    config = get_default_config()

    if args.run_all:
        run_all(config)
    else:
        run_experiment(args.dataset, args.experiment_type, args.seed, config)


if __name__ == "__main__":
    main()
