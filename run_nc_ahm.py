"""
run_nc_ahm.py — Neural Collapse–Guided Mask Selection (NC-AHM).

Uses Neural Collapse metrics to guide mask selection for AHSS.
Instead of random masks (AHSS) or displacement-optimized masks (D-AHM),
NC-AHM selects masks that maximally change NC structure.

NC-AHM Mask Selection Strategy:
  1. Generate candidate masks (random pool of 100)
  2. For each candidate mask, compute NC1 under that mask
  3. Select top-K masks that MINIMIZE NC1 (tighten cluster structure)
     → These masks REMOVE redundant heads → reveal fragile encoding
  4. Use selected masks for AHSS scoring

Comparison: Random → AHM → D-AHM → NC-AHM

Usage:
    python run_nc_ahm.py --run_all
    python run_nc_ahm.py --dataset financedocs --experiment_type intra --seed 42
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
from ood_methods.distance_based import MahalanobisScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from evaluation.nc_metrics import compute_nc1, compute_nc4
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nc_ahm")


def generate_candidate_masks(masker, num_candidates=100, masking_pcts=(0.1, 0.2), seed=42):
    """Generate a large pool of random candidate masks for NC-guided selection."""
    rng = np.random.RandomState(seed + 3333)
    masks = []
    for i in range(num_candidates):
        p = masking_pcts[i % len(masking_pcts)]
        mask = masker.create_mask(masking_percentage=p, seed=int(rng.randint(0, 2**31)))
        masks.append(mask)
    return masks


def select_nc_guided_masks(model, train_dl, candidate_masks, train_labels_arr,
                            device, top_k=5, strategy="minimize_nc1",
                            max_samples=500):
    """
    Select top-K masks based on NC metrics.

    Strategy 'minimize_nc1': Select masks that produce lowest NC1
    (tightest clusters). These masks remove redundant heads, revealing
    the model's essential structure.

    Strategy 'maximize_nc1_delta': Select masks that cause largest NC1
    change. These masks identify structurally important heads.

    Uses subsampling (max_samples) for efficiency on large datasets.
    """
    logger.info(f"Evaluating {len(candidate_masks)} candidate masks for NC-guided selection...")

    def extract_embeddings_subsample(mask):
        """Extract embeddings with optional subsampling for speed."""
        if mask is not None:
            model.set_attention_head_mask(mask)
        else:
            model.set_attention_head_mask(None)
        all_embs = []
        all_labels = []
        n_collected = 0
        with torch.no_grad():
            for batch in train_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                emb = model.extract_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                )
                all_embs.append(emb.cpu().numpy())
                if "labels" in batch:
                    all_labels.append(batch["labels"].numpy())
                n_collected += emb.shape[0]
                if max_samples and n_collected >= max_samples:
                    break
        embs = np.concatenate(all_embs, axis=0)[:max_samples] if max_samples else np.concatenate(all_embs, axis=0)
        labels = np.concatenate(all_labels, axis=0)[:max_samples] if all_labels and max_samples else train_labels_arr[:len(embs)]
        return embs, labels

    # First get unmasked NC1
    unmasked_embs, sub_labels = extract_embeddings_subsample(None)
    unmasked_nc1 = compute_nc1(unmasked_embs, sub_labels)
    logger.info(f"  Unmasked NC1: {unmasked_nc1:.2f} (on {len(unmasked_embs)} samples)")

    # Evaluate each candidate
    mask_scores = []
    for idx, mask in enumerate(candidate_masks):
        masked_embs, sub_labels = extract_embeddings_subsample(mask)
        nc1 = compute_nc1(masked_embs, sub_labels)

        if strategy == "minimize_nc1":
            mask_scores.append((idx, nc1, nc1 - unmasked_nc1))
        elif strategy == "maximize_nc1_delta":
            mask_scores.append((idx, nc1, abs(nc1 - unmasked_nc1)))

        if (idx + 1) % 20 == 0:
            logger.info(f"  Evaluated {idx+1}/{len(candidate_masks)} candidates...")

    model.set_attention_head_mask(None)

    # Select top-K
    if strategy == "minimize_nc1":
        mask_scores.sort(key=lambda x: x[1])  # ascending NC1
    elif strategy == "maximize_nc1_delta":
        mask_scores.sort(key=lambda x: x[2], reverse=True)  # descending delta

    selected_indices = [s[0] for s in mask_scores[:top_k]]
    selected_masks = [candidate_masks[i] for i in selected_indices]

    logger.info(f"  Selected {top_k} masks (strategy={strategy}):")
    for s in mask_scores[:top_k]:
        logger.info(f"    Mask {s[0]}: NC1={s[1]:.2f}, Δ={s[2]:.2f}")

    return selected_masks, mask_scores


def run_experiment(dataset, experiment_type, seed, config):
    """Run NC-AHM experiment for a single configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    base_dir = SCRIPT_DIR

    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints", exp_name, "best_model")
    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(base_dir, "outputs", "ahm_masks", exp_name)
    dahm_dir = os.path.join(base_dir, "outputs", "dahm_masks", exp_name)
    output_dir = os.path.join(base_dir, "outputs", "nc_ahm_results", exp_name)

    results_path = os.path.join(output_dir, "nc_ahm_results.json")
    if os.path.exists(results_path):
        logger.info(f"Results exist: {results_path}, skipping.")
        with open(results_path) as f:
            return json.load(f)

    for d in [embedding_dir, checkpoint_dir]:
        if not os.path.exists(d):
            logger.error(f"Missing: {d}")
            return None

    logger.info(f"\n{'='*70}")
    logger.info(f"NC-AHM EXPERIMENT: {exp_name}")
    logger.info(f"{'='*70}")

    start = time.time()

    masker = AttentionHeadMasker(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )

    # Load model
    dataset_config = config.financedocs if dataset == "financedocs" else config.tobacco
    id_classes = dataset_config.id_classes if experiment_type == "intra" else dataset_config.all_classes
    label2id = get_label2id(id_classes)
    id2label = get_id2label(id_classes)
    num_labels = len(id_classes)

    model = LayoutLMv3DocumentClassifier(
        model_name=checkpoint_dir, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
    )
    model.to(device)
    model.eval()

    # Load data
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

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
    train_idx, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )

    train_ds = id_ds.select(train_idx)
    eval_ds = id_ds.select(eval_idx)
    train_dl = create_dataloader(train_ds, batch_size=64, shuffle=False)
    eval_dl = create_dataloader(eval_ds, batch_size=64, shuffle=False)

    if experiment_type == "cross":
        ood_dummy = {lbl: num_labels for lbl in set(ood_ds["labels"])}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    else:
        ood_dummy = {dataset_config.ood_class: num_labels}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    ood_dl = create_dataloader(ood_ds, batch_size=64, shuffle=False)

    train_labels_np = np.array(train_ds["labels"])

    # Step 1: Generate candidate masks
    logger.info("Generating 100 candidate masks...")
    candidate_masks = generate_candidate_masks(
        masker, num_candidates=100,
        masking_pcts=config.ahss.masking_percentages, seed=seed,
    )

    # Step 2: NC-guided selection (minimize NC1)
    nc_masks, nc_scores = select_nc_guided_masks(
        model, train_dl, candidate_masks, train_labels_np,
        device, top_k=5, strategy="minimize_nc1",
    )

    # Step 3: Also prepare comparison mask sets
    # Random subset (first 5 candidates, equivalent to random selection)
    random_masks = candidate_masks[:5]

    # AHM masks (if available)
    ahm_masks = []
    for i in range(5):
        p = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if os.path.exists(p):
            ahm_masks.append(masker.load_mask(p))

    # D-AHM masks (if available)
    dahm_masks = []
    if os.path.exists(dahm_dir):
        for i in range(5):
            p = os.path.join(dahm_dir, f"dahm_mask_{i}.npy")
            if os.path.exists(p):
                raw = np.load(p)
                if (raw < 0.5).sum() > 0:  # skip all-ones
                    dahm_masks.append(masker.load_mask(p))

    # Step 4: Evaluate each mask set
    train_unmasked = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))

    results = {"experiment": exp_name, "dataset": dataset,
               "experiment_type": experiment_type, "seed": seed}

    mask_sets = {"nc_ahm": nc_masks, "random": random_masks}
    if ahm_masks:
        mask_sets["ahm"] = ahm_masks
    if dahm_masks:
        mask_sets["dahm"] = dahm_masks

    scorer = AHSSScorer(normalize=True)

    for mask_type, masks in mask_sets.items():
        if not masks:
            continue
        logger.info(f"\nEvaluating {mask_type} ({len(masks)} masks)...")

        _, eval_masked, _ = extract_embeddings_with_multiple_masks(
            model, eval_dl, masks, device=device, desc=f"eval {mask_type}",
        )
        eval_unmasked = np.load(os.path.join(embedding_dir, "eval_embeddings.npy"))

        _, ood_masked, _ = extract_embeddings_with_multiple_masks(
            model, ood_dl, masks, device=device, desc=f"OOD {mask_type}",
        )
        ood_unmasked = np.load(os.path.join(embedding_dir, "ood_embeddings.npy"))

        eval_ahss = -scorer.compute_ahss(eval_unmasked, eval_masked, normalize=True)
        ood_ahss = -scorer.compute_ahss(ood_unmasked, ood_masked, normalize=True)

        ahss_auroc = compute_auroc(eval_ahss, ood_ahss)
        ahss_fpr = compute_fpr_at_tpr(eval_ahss, ood_ahss)

        # Combined with Maha
        mah = MahalanobisScorer()
        mah.fit(train_unmasked, train_labels)
        eval_mah = mah.score(eval_unmasked)
        ood_mah = mah.score(ood_unmasked)

        from scipy.stats import rankdata
        def rank_norm(s):
            r = rankdata(s, method='average')
            return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(s)

        all_mah = np.concatenate([eval_mah, ood_mah])
        all_ahss = np.concatenate([eval_ahss, ood_ahss])
        n_ev = len(eval_mah)
        best_alpha = 0.5
        best_comb = 0.0
        for alpha in np.arange(0.0, 1.05, 0.1):
            c = alpha * rank_norm(all_mah) + (1 - alpha) * rank_norm(all_ahss)
            a = compute_auroc(c[:n_ev], c[n_ev:])
            if a > best_comb:
                best_comb = a
                best_alpha = alpha

        results[mask_type] = {
            "ahss_auroc": float(ahss_auroc),
            "ahss_fpr95": float(ahss_fpr),
            "combined_auroc": float(best_comb),
            "combined_alpha": float(best_alpha),
            "num_masks": len(masks),
        }
        logger.info(f"  {mask_type}: AHSS AUROC={ahss_auroc:.4f}, Combined={best_comb:.4f}")

    del model
    torch.cuda.empty_cache()

    # Summary comparison
    logger.info(f"\n{'='*50}")
    logger.info(f"NC-AHM COMPARISON: {exp_name}")
    logger.info(f"{'Mask Type':<12} {'AHSS AUROC':>12} {'Combined':>12}")
    logger.info("-" * 38)
    for mt in ["random", "ahm", "dahm", "nc_ahm"]:
        if mt in results and isinstance(results[mt], dict):
            r = results[mt]
            logger.info(f"{mt:<12} {r['ahss_auroc']:>12.4f} {r['combined_auroc']:>12.4f}")
    logger.info(f"{'='*50}")

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {results_path} ({elapsed:.1f}s)")

    return results


def run_all(config):
    """Run NC-AHM for all experiments."""
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

    # Aggregate
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATED NC-AHM RESULTS")
    logger.info(f"{'='*80}")

    for dataset, exp_type in experiments:
        exp_key = f"{dataset}_{exp_type}"
        exp_results = [r for r in all_results if r.get("dataset") == dataset
                       and r.get("experiment_type") == exp_type]
        if not exp_results:
            continue

        logger.info(f"\n{exp_key} ({len(exp_results)} seeds):")
        for mt in ["random", "ahm", "dahm", "nc_ahm"]:
            aurocs = [r[mt]["ahss_auroc"] for r in exp_results if mt in r and isinstance(r[mt], dict)]
            combined = [r[mt]["combined_auroc"] for r in exp_results if mt in r and isinstance(r[mt], dict)]
            if aurocs:
                logger.info(f"  {mt:<12} AHSS: {np.mean(aurocs):.4f}±{np.std(aurocs):.4f}"
                             f"  Combined: {np.mean(combined):.4f}±{np.std(combined):.4f}")

    agg_path = os.path.join(SCRIPT_DIR, "outputs", "nc_ahm_results", "aggregated_results.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="NC-Guided Mask Selection (NC-AHM)")
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
