"""
run_near_ood.py — near-OOD experiments (original construction of Table 4).
The reported document near-OOD values use leave-class-out models instead
(verification_2026-10/code/e1_doc_nearood.py).

Tests whether AHSS provides disproportionate advantage on near-OOD
(semantically similar, hard-to-distinguish distributions) compared
to far-OOD (easy to distinguish).

Near-OOD Experiments:
1. FinanceDocs: sec_form (ID) vs sec_letter (OOD) — same document domain
2. Tobacco: Letter (ID) vs Memo (OOD) — same document type family
3. ViT: CIFAR-10 (ID) vs CIFAR-100 (OOD) — same visual domain

Far-OOD (baseline comparison from existing results):
1. FinanceDocs intra: 9 classes (ID) vs resumes (OOD) — well-separated
2. FinanceDocs cross: FinanceDocs (ID) vs Tobacco (OOD) — different domain
3. ViT: CIFAR-10 (ID) vs SVHN (OOD) — different visual domain

Key Metric: Δ_near = AHSS improvement on near-OOD vs far-OOD
Hypothesis: Δ_near > 0 (AHSS helps more on near-OOD)

Usage:
    python run_near_ood.py --run_all
    python run_near_ood.py --experiment financedocs_near --seed 42
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
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("near_ood")


# Near-OOD experiment definitions
NEAR_OOD_EXPERIMENTS = {
    "financedocs_near": {
        "model_exp": "financedocs_intra",  # Use existing model trained on 9 ID classes
        "dataset": "financedocs",
        "description": "sec_form (ID) vs sec_letter (OOD) — same domain near-OOD",
        # We reuse the model trained on all 9 ID classes, but ONLY evaluate on
        # sec_form as ID and sec_letter as OOD to test near-OOD detection
        "id_eval_classes": ["sec_form"],
        "ood_classes": ["sec_letter"],
        "data_source": "financedocs",
    },
    "tobacco_near": {
        "model_exp": "tobacco_intra",
        "dataset": "tobacco",
        "description": "Letter (ID) vs Memo (OOD) — same type family near-OOD",
        "id_eval_classes": ["Letter"],
        "ood_classes": ["Memo"],
        "data_source": "tobacco",
    },
}


def run_near_ood_experiment(experiment_name, seed, config):
    """Run a near-OOD evaluation experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_info = NEAR_OOD_EXPERIMENTS[experiment_name]
    model_exp = exp_info["model_exp"]
    base_dir = SCRIPT_DIR

    model_seed_name = f"{model_exp}_seed{seed}"
    output_dir = os.path.join(base_dir, "outputs", "near_ood_results",
                               f"{experiment_name}_seed{seed}")
    results_path = os.path.join(output_dir, "near_ood_results.json")

    if os.path.exists(results_path):
        logger.info(f"Results exist: {results_path}, skipping.")
        with open(results_path) as f:
            return json.load(f)

    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints",
                                   model_seed_name, "best_model")
    ahm_dir = os.path.join(base_dir, "outputs", "ahm_masks", model_seed_name)
    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", model_seed_name)

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Model not found: {checkpoint_dir}")
        return None

    logger.info(f"\n{'='*70}")
    logger.info(f"NEAR-OOD: {experiment_name} (seed {seed})")
    logger.info(f"  {exp_info['description']}")
    logger.info(f"  Model from: {model_seed_name}")
    logger.info(f"{'='*70}")

    start = time.time()

    # Load dataset
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    data_base = os.path.join(base_dir, "data")
    if exp_info["data_source"] == "financedocs":
        data_path = os.path.join(data_base, "colated_tensors")
    else:
        data_path = os.path.join(data_base, "tobacco", "colated_tensors")

    full_ds = load_from_disk(data_path)

    # Get the model's full ID class set for loading the model
    dataset_config = config.financedocs if exp_info["dataset"] == "financedocs" else config.tobacco
    model_id_classes = dataset_config.id_classes  # 9 classes used for training
    label2id = get_label2id(model_id_classes)
    id2label = get_id2label(model_id_classes)
    num_labels = len(model_id_classes)

    # Load model
    model = LayoutLMv3DocumentClassifier(
        model_name=checkpoint_dir, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
    )
    model.to(device)
    model.eval()

    masker = AttentionHeadMasker(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )

    # Load masks
    masks = []
    for i in range(100):
        path = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if not os.path.exists(path):
            break
        masks.append(masker.load_mask(path))

    # Add random masks
    rng = np.random.RandomState(seed + 6666)
    for j in range(20):
        p = [0.1, 0.2][j % 2]
        mask = masker.create_mask(masking_percentage=p, seed=int(rng.randint(0, 2**31)))
        masks.append(mask)

    logger.info(f"Using {len(masks)} masks")

    # Get near-OOD ID subset and OOD subset
    id_classes_near = exp_info["id_eval_classes"]
    ood_classes = exp_info["ood_classes"]

    # Filter for ID eval class(es)
    id_near_ds = filter_dataset_by_classes(full_ds, id_classes_near)
    # Filter for OOD class(es)
    ood_near_ds = filter_dataset_by_classes(full_ds, ood_classes)

    logger.info(f"Near-OOD: ID samples={len(id_near_ds)}, OOD samples={len(ood_near_ds)}")

    # Encode labels using the MODEL's label2id (for consistent embedding extraction)
    id_near_ds = encode_labels(id_near_ds, label2id)
    ood_label_map = {lbl: num_labels for lbl in ood_classes}
    ood_near_ds = encode_labels(ood_near_ds, ood_label_map)

    # Split ID into train/eval
    labels_arr = np.array(id_near_ds["labels"])
    if len(np.unique(labels_arr)) > 1:
        _, eval_idx = train_test_split(
            np.arange(len(id_near_ds)), train_size=0.8,
            random_state=seed, stratify=labels_arr,
        )
    else:
        _, eval_idx = train_test_split(
            np.arange(len(id_near_ds)), train_size=0.8,
            random_state=seed,
        )

    eval_ds = id_near_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=64, shuffle=False)
    ood_dl = create_dataloader(ood_near_ds, batch_size=64, shuffle=False)

    # Extract multi-mask embeddings
    logger.info("Extracting eval multi-mask embeddings...")
    eval_unmasked, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, eval_dl, masks, device=device, desc="Near ID eval",
    )
    logger.info("Extracting OOD multi-mask embeddings...")
    ood_unmasked, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, ood_dl, masks, device=device, desc="Near OOD",
    )

    # Load train embeddings for Mahalanobis (from existing results)
    train_embs = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))

    del model
    torch.cuda.empty_cache()

    # Compute scores
    results = {
        "experiment": experiment_name, "seed": seed,
        "description": exp_info["description"],
        "n_id_eval": len(eval_ds), "n_ood": len(ood_near_ds),
    }

    # AHSS
    scorer = AHSSScorer(normalize=True)
    eval_ahss = -scorer.compute_ahss(eval_unmasked, eval_masked_list, normalize=True)
    ood_ahss = -scorer.compute_ahss(ood_unmasked, ood_masked_list, normalize=True)
    ahss_auroc = compute_auroc(eval_ahss, ood_ahss)
    ahss_fpr = compute_fpr_at_tpr(eval_ahss, ood_ahss)
    results["ahss"] = {"auroc": float(ahss_auroc), "fpr_at_95": float(ahss_fpr)}
    logger.info(f"  AHSS: AUROC={ahss_auroc:.4f}")

    # Mahalanobis
    mah = MahalanobisScorer()
    mah.fit(train_embs, train_labels)
    eval_mah = mah.score(eval_unmasked)
    ood_mah = mah.score(ood_unmasked)
    mah_auroc = compute_auroc(eval_mah, ood_mah)
    mah_fpr = compute_fpr_at_tpr(eval_mah, ood_mah)
    results["mahalanobis"] = {"auroc": float(mah_auroc), "fpr_at_95": float(mah_fpr)}
    logger.info(f"  Mahalanobis: AUROC={mah_auroc:.4f}")

    # kNN
    knn = KNNScorer(k=50, metric="euclidean")
    knn.fit(train_embs)
    eval_knn = knn.score(eval_unmasked)
    ood_knn = knn.score(ood_unmasked)
    knn_auroc = compute_auroc(eval_knn, ood_knn)
    knn_fpr = compute_fpr_at_tpr(eval_knn, ood_knn)
    results["knn"] = {"auroc": float(knn_auroc), "fpr_at_95": float(knn_fpr)}
    logger.info(f"  kNN: AUROC={knn_auroc:.4f}")

    # Combined
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

    results["combined"] = {
        "auroc": float(best_comb),
        "best_alpha": float(best_alpha),
    }
    logger.info(f"  Combined: AUROC={best_comb:.4f} (α={best_alpha:.1f})")

    # Delta from Mahalanobis
    delta = best_comb - mah_auroc
    results["delta_from_mah"] = float(delta)
    logger.info(f"  Δ(Combined - Mah): {delta:+.4f}")

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {results_path} ({elapsed:.1f}s)")

    return results


def run_all(config):
    """Run all near-OOD experiments."""
    seeds = config.seeds
    all_results = []

    for exp_name in NEAR_OOD_EXPERIMENTS:
        for seed in seeds:
            try:
                r = run_near_ood_experiment(exp_name, seed, config)
                if r is not None:
                    all_results.append(r)
            except Exception as e:
                logger.error(f"FAILED {exp_name}_seed{seed}: {e}")
                import traceback
                traceback.print_exc()

    # Compare near-OOD vs far-OOD from existing results
    logger.info(f"\n{'='*80}")
    logger.info("NEAR-OOD vs FAR-OOD COMPARISON")
    logger.info(f"{'='*80}")

    for exp_name in NEAR_OOD_EXPERIMENTS:
        exp_results = [r for r in all_results if r.get("experiment") == exp_name]
        if not exp_results:
            continue

        ahss_aurocs = [r["ahss"]["auroc"] for r in exp_results]
        mah_aurocs = [r["mahalanobis"]["auroc"] for r in exp_results]
        comb_aurocs = [r["combined"]["auroc"] for r in exp_results]
        deltas = [r["delta_from_mah"] for r in exp_results]

        logger.info(f"\n{exp_name} ({len(exp_results)} seeds):")
        logger.info(f"  AHSS:      {np.mean(ahss_aurocs):.4f}±{np.std(ahss_aurocs):.4f}")
        logger.info(f"  Maha:      {np.mean(mah_aurocs):.4f}±{np.std(mah_aurocs):.4f}")
        logger.info(f"  Combined:  {np.mean(comb_aurocs):.4f}±{np.std(comb_aurocs):.4f}")
        logger.info(f"  Δ(C-Mah):  {np.mean(deltas):+.4f}±{np.std(deltas):.4f}")

    # Save aggregated
    agg_path = os.path.join(SCRIPT_DIR, "outputs", "near_ood_results", "aggregated_results.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="Near-OOD Advantage Evaluation")
    parser.add_argument("--experiment", type=str, default="financedocs_near",
                        choices=list(NEAR_OOD_EXPERIMENTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    config = get_default_config()

    if args.run_all:
        run_all(config)
    else:
        run_near_ood_experiment(args.experiment, args.seed, config)


if __name__ == "__main__":
    main()
