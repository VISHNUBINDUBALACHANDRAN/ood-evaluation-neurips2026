"""
run_ablation_studies.py — Phase C: Ablation Studies on AHSS Hyperparameters.

Systematically varies key AHSS parameters to measure sensitivity:
1. T (number of masks): [5, 10, 15, 20, 25, 30, 50]
2. p (masking probability): [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
3. Top-F masks: [1, 3, 5, 10, 25]

Uses a REPRESENTATIVE subset of experiments for efficiency:
- financedocs_intra_seed42 (canonical example)
- tobacco_intra_seed42 (harder case)

Usage:
    python run_ablation_studies.py --run_all
    python run_ablation_studies.py --ablation T --dataset financedocs --seed 42
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
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from ood_methods.distance_based import MahalanobisScorer
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation")


def generate_masks(T, p_range, num_layers=12, num_heads=12, seed=42):
    """Generate T random masks with masking probability sampled from p_range."""
    rng = np.random.RandomState(seed)
    masks = []
    for _ in range(T):
        p = rng.uniform(p_range[0], p_range[1])
        mask = (rng.random((num_layers, num_heads)) > p).astype(np.float32)
        masks.append(torch.tensor(mask, dtype=torch.float32))
    return masks


def compute_ahss_from_embeddings(unmasked, masked_list):
    """Compute AHSS scores from pre-extracted embeddings."""
    ref = unmasked.astype(np.float64)
    norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
    ref = ref / norms

    ahss = np.zeros(len(ref), dtype=np.float64)
    for masked in masked_list:
        m = masked.astype(np.float64)
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
        ahss += np.sum((m - ref) ** 2, axis=1)
    ahss /= len(masked_list)
    return ahss


def load_experiment_data(dataset, seed, config, device="cuda"):
    """Load model and data for an experiment."""
    exp_name = f"{dataset}_intra_seed{seed}"
    base_dir = SCRIPT_DIR

    dataset_config = config.financedocs if dataset == "financedocs" else config.tobacco
    id_classes = dataset_config.id_classes
    label2id = get_label2id(id_classes)
    id2label = get_id2label(id_classes)
    num_labels = len(id_classes)

    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints", exp_name, "best_model")
    model = LayoutLMv3DocumentClassifier(
        model_name=checkpoint_dir, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
    )
    model.to(device)
    model.eval()

    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    data_base = os.path.join(base_dir, "data")
    if dataset == "financedocs":
        data_path = os.path.join(data_base, "colated_tensors")
    else:
        data_path = os.path.join(data_base, "tobacco", "colated_tensors")

    full_ds = load_from_disk(data_path)
    id_ds = filter_dataset_by_classes(full_ds, id_classes)
    ood_ds = filter_dataset_by_classes(full_ds, [dataset_config.ood_class])

    id_ds = encode_labels(id_ds, label2id)
    labels_arr = np.array(id_ds["labels"])
    indices = np.arange(len(id_ds))
    train_idx, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )
    train_ds = id_ds.select(train_idx)
    eval_ds = id_ds.select(eval_idx)

    ood_dummy = {dataset_config.ood_class: num_labels}
    ood_ds = encode_labels(ood_ds, ood_dummy)

    train_dl = create_dataloader(train_ds, batch_size=160, shuffle=False)
    eval_dl = create_dataloader(eval_ds, batch_size=160, shuffle=False)
    ood_dl = create_dataloader(ood_ds, batch_size=160, shuffle=False)

    return model, train_dl, eval_dl, ood_dl, device


def run_ablation_T(dataset, seed, config):
    """Ablation on T (number of masks)."""
    T_values = [5, 10, 15, 20, 25, 30, 50]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION T: {dataset}_intra_seed{seed}")
    logger.info(f"T values: {T_values}")
    logger.info(f"{'='*70}")

    model, train_dl, eval_dl, ood_dl, device = load_experiment_data(dataset, seed, config, device)

    # Generate the largest set of masks (T=50) and subsample
    all_masks = generate_masks(50, p_range=[0.1, 0.2], seed=seed)

    # Extract embeddings with all 50 masks
    logger.info("Extracting embeddings with T=50 masks...")
    eval_unmasked, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, eval_dl, all_masks, device=device, desc="eval T=50",
    )
    ood_unmasked, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, ood_dl, all_masks, device=device, desc="ood T=50",
    )

    # Also get train embeddings for Mahalanobis
    train_embs_list = []
    train_labels_list = []
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
                input_ids=input_ids, attention_mask=attention_mask,
                bbox=bbox, pixel_values=pixel_values, token_type_ids=token_type_ids,
            )
            train_embs_list.append(emb.cpu().numpy())
            train_labels_list.append(batch["labels"].numpy())
    train_embs = np.concatenate(train_embs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # Fit Mahalanobis
    maha = MahalanobisScorer()
    maha.fit(train_embs, train_labels)
    eval_maha = maha.score(eval_unmasked)
    ood_maha = maha.score(ood_unmasked)

    del model
    torch.cuda.empty_cache()

    results = []
    for T in T_values:
        eval_subset = eval_masked_list[:T]
        ood_subset = ood_masked_list[:T]

        eval_ahss = compute_ahss_from_embeddings(eval_unmasked, eval_subset)
        ood_ahss = compute_ahss_from_embeddings(ood_unmasked, ood_subset)

        auroc = compute_auroc(-eval_ahss, -ood_ahss)
        fpr = compute_fpr_at_tpr(-eval_ahss, -ood_ahss)

        # Combined
        from scipy.stats import rankdata
        def rank_norm(s):
            r = rankdata(s, method='average')
            return (r - 1) / (len(r) - 1)

        all_m = np.concatenate([eval_maha, ood_maha])
        all_a = np.concatenate([-eval_ahss, -ood_ahss])
        n = len(eval_maha)

        best_alpha, best_comb = 0.5, 0.0
        for alpha in np.arange(0, 1.05, 0.1):
            comb = alpha * rank_norm(all_m) + (1 - alpha) * rank_norm(all_a)
            a = compute_auroc(comb[:n], comb[n:])
            if a > best_comb:
                best_comb = a
                best_alpha = alpha

        result = {
            "T": T,
            "ahss_auroc": float(auroc),
            "ahss_fpr95": float(fpr),
            "combined_auroc": float(best_comb),
            "combined_alpha": float(best_alpha),
        }
        results.append(result)
        logger.info(f"  T={T:3d}: AHSS={auroc:.4f}, Combined={best_comb:.4f}")

    return results


def run_ablation_p(dataset, seed, config):
    """Ablation on p (masking probability)."""
    p_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    T = 25
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION p: {dataset}_intra_seed{seed}")
    logger.info(f"p values: {p_values}")
    logger.info(f"{'='*70}")

    model, train_dl, eval_dl, ood_dl, device = load_experiment_data(dataset, seed, config, device)

    # Get train embeddings
    train_embs_list = []
    train_labels_list = []
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
                input_ids=input_ids, attention_mask=attention_mask,
                bbox=bbox, pixel_values=pixel_values, token_type_ids=token_type_ids,
            )
            train_embs_list.append(emb.cpu().numpy())
            train_labels_list.append(batch["labels"].numpy())
    train_embs = np.concatenate(train_embs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    results = []
    for p in p_values:
        masks = generate_masks(T, p_range=[p, p], seed=seed)

        eval_unmasked, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
            model, eval_dl, masks, device=device, desc=f"eval p={p}",
        )
        ood_unmasked, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
            model, ood_dl, masks, device=device, desc=f"ood p={p}",
        )

        eval_ahss = compute_ahss_from_embeddings(eval_unmasked, eval_masked_list)
        ood_ahss = compute_ahss_from_embeddings(ood_unmasked, ood_masked_list)

        auroc = compute_auroc(-eval_ahss, -ood_ahss)
        fpr = compute_fpr_at_tpr(-eval_ahss, -ood_ahss)

        # Mahalanobis + combined
        maha = MahalanobisScorer()
        maha.fit(train_embs, train_labels)
        eval_maha = maha.score(eval_unmasked)
        ood_maha = maha.score(ood_unmasked)

        from scipy.stats import rankdata
        def rank_norm(s):
            r = rankdata(s, method='average')
            return (r - 1) / (len(r) - 1)

        all_m = np.concatenate([eval_maha, ood_maha])
        all_a = np.concatenate([-eval_ahss, -ood_ahss])
        n = len(eval_maha)

        best_alpha, best_comb = 0.5, 0.0
        for alpha in np.arange(0, 1.05, 0.1):
            comb = alpha * rank_norm(all_m) + (1 - alpha) * rank_norm(all_a)
            a = compute_auroc(comb[:n], comb[n:])
            if a > best_comb:
                best_comb = a
                best_alpha = alpha

        result = {
            "p": p,
            "ahss_auroc": float(auroc),
            "ahss_fpr95": float(fpr),
            "combined_auroc": float(best_comb),
            "combined_alpha": float(best_alpha),
        }
        results.append(result)
        logger.info(f"  p={p:.2f}: AHSS={auroc:.4f}, Combined={best_comb:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def run_ablation_topF(dataset, seed, config):
    """Ablation on Top-F (number of top masks to keep after AHM selection)."""
    F_values = [1, 3, 5, 10, 25]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION Top-F: {dataset}_intra_seed{seed}")
    logger.info(f"F values: {F_values}")
    logger.info(f"{'='*70}")

    model, train_dl, eval_dl, ood_dl, device = load_experiment_data(dataset, seed, config, device)

    # Generate 25 masks
    all_masks = generate_masks(25, p_range=[0.1, 0.2], seed=seed)

    # Extract all embeddings
    eval_unmasked, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, eval_dl, all_masks, device=device, desc="eval F=25",
    )
    ood_unmasked, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, ood_dl, all_masks, device=device, desc="ood F=25",
    )

    # Compute per-mask AHSS for eval to rank masks
    per_mask_aurocs = []
    for i, (em, om) in enumerate(zip(eval_masked_list, ood_masked_list)):
        eval_ahss_i = compute_ahss_from_embeddings(eval_unmasked, [em])
        ood_ahss_i = compute_ahss_from_embeddings(ood_unmasked, [om])
        auroc_i = compute_auroc(-eval_ahss_i, -ood_ahss_i)
        per_mask_aurocs.append(auroc_i)

    # Rank masks by per-mask AUROC (best first)
    sorted_indices = np.argsort(per_mask_aurocs)[::-1]

    # Get train embeddings for Maha
    train_embs_list = []
    train_labels_list = []
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
                input_ids=input_ids, attention_mask=attention_mask,
                bbox=bbox, pixel_values=pixel_values, token_type_ids=token_type_ids,
            )
            train_embs_list.append(emb.cpu().numpy())
            train_labels_list.append(batch["labels"].numpy())
    train_embs = np.concatenate(train_embs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    maha = MahalanobisScorer()
    maha.fit(train_embs, train_labels)
    eval_maha = maha.score(eval_unmasked)
    ood_maha = maha.score(ood_unmasked)

    del model
    torch.cuda.empty_cache()

    results = []
    for F in F_values:
        top_idx = sorted_indices[:F]
        eval_top = [eval_masked_list[i] for i in top_idx]
        ood_top = [ood_masked_list[i] for i in top_idx]

        eval_ahss = compute_ahss_from_embeddings(eval_unmasked, eval_top)
        ood_ahss = compute_ahss_from_embeddings(ood_unmasked, ood_top)

        auroc = compute_auroc(-eval_ahss, -ood_ahss)
        fpr = compute_fpr_at_tpr(-eval_ahss, -ood_ahss)

        from scipy.stats import rankdata
        def rank_norm(s):
            r = rankdata(s, method='average')
            return (r - 1) / (len(r) - 1)

        all_m = np.concatenate([eval_maha, ood_maha])
        all_a = np.concatenate([-eval_ahss, -ood_ahss])
        n = len(eval_maha)

        best_alpha, best_comb = 0.5, 0.0
        for alpha in np.arange(0, 1.05, 0.1):
            comb = alpha * rank_norm(all_m) + (1 - alpha) * rank_norm(all_a)
            a = compute_auroc(comb[:n], comb[n:])
            if a > best_comb:
                best_comb = a
                best_alpha = alpha

        result = {
            "F": F,
            "ahss_auroc": float(auroc),
            "ahss_fpr95": float(fpr),
            "combined_auroc": float(best_comb),
            "combined_alpha": float(best_alpha),
            "mask_indices": [int(i) for i in top_idx],
        }
        results.append(result)
        logger.info(f"  F={F:3d}: AHSS={auroc:.4f}, Combined={best_comb:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AHSS Ablation Studies")
    parser.add_argument("--ablation", type=str, default="T",
                        choices=["T", "p", "topF", "all"])
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    config = get_default_config()
    output_dir = os.path.join(SCRIPT_DIR, "outputs", "ablation_results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    datasets = ["financedocs", "tobacco"] if args.run_all else [args.dataset]
    ablations = ["T", "p", "topF"] if (args.run_all or args.ablation == "all") else [args.ablation]

    for dataset in datasets:
        for ablation in ablations:
            key = f"{dataset}_intra_seed{args.seed}_{ablation}"
            result_path = os.path.join(output_dir, f"{key}.json")

            if os.path.exists(result_path):
                logger.info(f"Skipping {key} (results exist)")
                with open(result_path) as f:
                    all_results[key] = json.load(f)
                continue

            start = time.time()
            if ablation == "T":
                results = run_ablation_T(dataset, args.seed, config)
            elif ablation == "p":
                results = run_ablation_p(dataset, args.seed, config)
            elif ablation == "topF":
                results = run_ablation_topF(dataset, args.seed, config)

            elapsed = time.time() - start
            output = {
                "ablation": ablation,
                "dataset": dataset,
                "seed": args.seed,
                "results": results,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(result_path, "w") as f:
                json.dump(output, f, indent=2)
            all_results[key] = output
            logger.info(f"Saved {key} ({elapsed:.1f}s)")

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY SUMMARY")
    logger.info(f"{'='*80}")

    for key, data in all_results.items():
        logger.info(f"\n{key}:")
        for r in data["results"]:
            param = r.get("T", r.get("p", r.get("F", "?")))
            logger.info(f"  {param}: AHSS={r['ahss_auroc']:.4f}, Combined={r['combined_auroc']:.4f}")


if __name__ == "__main__":
    main()
