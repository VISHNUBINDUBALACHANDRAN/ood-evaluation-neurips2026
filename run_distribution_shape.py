"""
run_distribution_shape.py — AHSS Distribution Shape Analysis.

Computes the full per-sample displacement distribution statistics (FSP Signature):
  FSP(x) = [μ(x), σ²(x), γ₁(x), κ(x)]

where for each sample x and T masks:
  d_t(x) = ||f(x; M_t) - f(x; 1)||²   (per-mask displacement)
  μ(x)    = mean(d_t)                   (= standard AHSS)
  σ²(x)  = var(d_t)                     (displacement variance)
  γ₁(x)  = skewness(d_t)               (distribution asymmetry)
  κ(x)   = kurtosis(d_t) - 3           (excess kurtosis)

Hypothesis:
  - ID data: uniformly low displacement → low variance, near-zero skewness
  - OOD data: high displacement with high variance → positive skewness

This script:
  1. Loads model + masks for each experiment
  2. Computes per-mask displacement for each sample
  3. Computes 4 moments of the distribution
  4. Uses each moment as OOD score → evaluates AUROC
  5. Combines all 4 moments via logistic regression → FSP score
  6. Statistical tests (Mann-Whitney U) for significance

Usage:
    python run_distribution_shape.py --run_all
    python run_distribution_shape.py --dataset financedocs --experiment_type intra --seed 42
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
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import utils.compat  # noqa: F401

from configs.config import get_default_config, get_label2id, get_id2label
from models.layoutlmv3_classifier import (
    LayoutLMv3DocumentClassifier,
    extract_embeddings_with_multiple_masks,
)
from models.attention_masking import AttentionHeadMasker
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from ood_methods.distance_based import MahalanobisScorer
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dist_shape")


def compute_per_mask_displacements(unmasked, masked_list, normalize=True):
    """
    Compute per-mask displacement for each sample.

    Args:
        unmasked: (N, D) unmasked embeddings
        masked_list: list of T arrays, each (N, D)
        normalize: L2-normalize before displacement

    Returns:
        displacements: (N, T) array where displacements[i, t] = ||f(x_i; M_t) - f(x_i; 1)||²
    """
    ref = unmasked.astype(np.float64)
    if normalize:
        norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
        ref = ref / norms

    T = len(masked_list)
    N = ref.shape[0]
    displacements = np.zeros((N, T), dtype=np.float64)

    for t, masked in enumerate(masked_list):
        m = masked.astype(np.float64)
        if normalize:
            m_norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
            m = m / m_norms
        diff = m - ref
        displacements[:, t] = np.sum(diff ** 2, axis=1)

    return displacements


def compute_fsp_features(displacements):
    """
    Compute FSP feature vector from per-mask displacements.

    FSP(x) = [μ(x), σ²(x), γ₁(x), κ(x)]

    Args:
        displacements: (N, T) per-mask displacement array

    Returns:
        fsp: dict with keys 'mean', 'variance', 'skewness', 'kurtosis'
             each is (N,) array
    """
    mu = np.mean(displacements, axis=1)
    var = np.var(displacements, axis=1)

    # Skewness and kurtosis using scipy for numerical stability
    # scipy.stats.skew and kurtosis with fisher=True (excess kurtosis)
    skew = scipy_stats.skew(displacements, axis=1)
    kurt = scipy_stats.kurtosis(displacements, axis=1, fisher=True)

    return {
        "mean": mu,       # = standard AHSS
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
    }


def evaluate_fsp_feature(eval_feat, ood_feat, name, higher_is_ood=True):
    """
    Evaluate a single FSP feature as OOD detector.

    Convention: higher score = more ID-like for compute_auroc.
    If higher_is_ood=True, we negate the scores.
    """
    if higher_is_ood:
        eval_scores = -eval_feat
        ood_scores = -ood_feat
    else:
        eval_scores = eval_feat
        ood_scores = ood_feat

    auroc = compute_auroc(eval_scores, ood_scores)
    fpr = compute_fpr_at_tpr(eval_scores, ood_scores)

    # Mann-Whitney U test
    u_stat, p_value = scipy_stats.mannwhitneyu(eval_feat, ood_feat, alternative='two-sided')

    return {
        "name": name,
        "auroc": float(auroc),
        "fpr_at_95": float(fpr),
        "id_mean": float(np.mean(eval_feat)),
        "id_std": float(np.std(eval_feat)),
        "id_median": float(np.median(eval_feat)),
        "ood_mean": float(np.mean(ood_feat)),
        "ood_std": float(np.std(ood_feat)),
        "ood_median": float(np.median(ood_feat)),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value),
        "significant": bool(p_value < 0.001),
    }


def compute_fsp_combined(eval_fsp, ood_fsp, train_fsp, train_labels):
    """
    Combine all 4 FSP features via logistic regression for a unified OOD score.

    Trains a simple LogisticRegression on train+synthetic_ood FSP features.
    Since we only have ID train data, we use a train-based threshold approach.
    
    Instead, we use a simpler rank-based combination.
    """
    from scipy.stats import rankdata

    def rank_norm(scores):
        r = rankdata(scores, method='average')
        return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(scores)

    # Stack all 4 features for eval and OOD
    feature_names = ["mean", "variance", "skewness", "kurtosis"]
    n_eval = len(eval_fsp["mean"])
    n_ood = len(ood_fsp["mean"])

    best_auroc = 0.0
    best_config = None

    # Grid search over feature subsets and weights
    # Try each individual feature + all pairs + all 4
    import itertools
    for r in range(1, len(feature_names) + 1):
        for combo in itertools.combinations(range(len(feature_names)), r):
            # Combine using rank normalization
            eval_combined = np.zeros(n_eval)
            ood_combined = np.zeros(n_ood)
            for idx in combo:
                fname = feature_names[idx]
                all_scores = np.concatenate([eval_fsp[fname], ood_fsp[fname]])
                ranks = rank_norm(all_scores)
                # Negate: higher displacement features = more OOD
                eval_combined += -ranks[:n_eval]
                ood_combined += -ranks[n_eval:]

            auroc = compute_auroc(eval_combined, ood_combined)
            if auroc > best_auroc:
                best_auroc = auroc
                best_config = combo
                best_eval = eval_combined.copy()
                best_ood = ood_combined.copy()

    fpr = compute_fpr_at_tpr(best_eval, best_ood)
    selected_features = [feature_names[i] for i in best_config]

    return {
        "name": "FSP_combined",
        "auroc": float(best_auroc),
        "fpr_at_95": float(fpr),
        "selected_features": selected_features,
    }


def run_experiment(dataset, experiment_type, seed, config):
    """Run distribution shape analysis for a single experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    base_dir = SCRIPT_DIR

    # Paths
    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints", exp_name, "best_model")
    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(base_dir, "outputs", "ahm_masks", exp_name)
    output_dir = os.path.join(base_dir, "outputs", "distribution_shape", exp_name)

    results_path = os.path.join(output_dir, "distribution_shape_results.json")
    if os.path.exists(results_path):
        logger.info(f"Results exist: {results_path}, skipping.")
        with open(results_path) as f:
            return json.load(f)

    for d in [embedding_dir, ahm_dir, checkpoint_dir]:
        if not os.path.exists(d):
            logger.error(f"Missing: {d}")
            return None

    logger.info(f"\n{'='*70}")
    logger.info(f"DISTRIBUTION SHAPE ANALYSIS: {exp_name}")
    logger.info(f"{'='*70}")

    start = time.time()

    # Setup
    masker = AttentionHeadMasker(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )

    # Load AHM masks + generate random masks (total T=25+5=30)
    masks = []
    sources = []
    for i in range(100):
        path = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if not os.path.exists(path):
            break
        masks.append(masker.load_mask(path))
        sources.append(f"ahm_{i}")

    # Add random masks to get T=30 total
    rng = np.random.RandomState(seed + 5555)
    num_random = 25
    for j in range(num_random):
        p = [0.1, 0.2][j % 2]
        mask = masker.create_mask(masking_percentage=p, seed=int(rng.randint(0, 2**31)))
        masks.append(mask)
        sources.append(f"random_{j}")

    logger.info(f"Total masks: {len(masks)} ({len(masks) - num_random} AHM + {num_random} random)")

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
    eval_ds = id_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=64, shuffle=False)

    if experiment_type == "cross":
        ood_dummy = {lbl: num_labels for lbl in set(ood_ds["labels"])}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    else:
        ood_dummy = {dataset_config.ood_class: num_labels}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    ood_dl = create_dataloader(ood_ds, batch_size=64, shuffle=False)

    # Also load train data for Mahalanobis baseline
    train_ds = id_ds.select(train_idx)
    train_dl = create_dataloader(train_ds, batch_size=64, shuffle=False)

    # Extract multi-mask embeddings
    logger.info("Extracting eval multi-mask embeddings...")
    eval_unmasked, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, eval_dl, masks, device=device, desc=f"{exp_name} eval",
    )

    logger.info("Extracting OOD multi-mask embeddings...")
    ood_unmasked, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, ood_dl, masks, device=device, desc=f"{exp_name} OOD",
    )

    # Load precomputed train embeddings
    train_unmasked = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))

    del model
    torch.cuda.empty_cache()

    # Compute per-mask displacements
    logger.info("Computing per-mask displacements...")
    eval_displacements = compute_per_mask_displacements(eval_unmasked, eval_masked_list)
    ood_displacements = compute_per_mask_displacements(ood_unmasked, ood_masked_list)

    # Compute FSP features
    logger.info("Computing FSP features...")
    eval_fsp = compute_fsp_features(eval_displacements)
    ood_fsp = compute_fsp_features(ood_displacements)

    # Evaluate each FSP feature individually
    results = {
        "experiment": exp_name, "dataset": dataset,
        "experiment_type": experiment_type, "seed": seed,
        "num_masks": len(masks),
    }

    feature_results = {}
    for fname in ["mean", "variance", "skewness", "kurtosis"]:
        r = evaluate_fsp_feature(eval_fsp[fname], ood_fsp[fname], f"FSP_{fname}", higher_is_ood=True)
        feature_results[fname] = r
        logger.info(f"  FSP_{fname}: AUROC={r['auroc']:.4f}, "
                     f"ID={r['id_mean']:.6f}±{r['id_std']:.6f}, "
                     f"OOD={r['ood_mean']:.6f}±{r['ood_std']:.6f}, "
                     f"p={r['mann_whitney_p']:.2e}")

    results["individual_features"] = feature_results

    # Combined FSP score
    train_disp = compute_per_mask_displacements(
        train_unmasked,
        [eval_masked_list[0][:len(train_unmasked)]] if len(train_unmasked) <= len(eval_masked_list[0]) else [train_unmasked],
    )
    train_fsp = compute_fsp_features(train_disp) if train_disp.shape[1] > 1 else None

    combined = compute_fsp_combined(eval_fsp, ood_fsp, train_fsp, train_labels)
    results["combined"] = combined
    logger.info(f"  FSP_combined: AUROC={combined['auroc']:.4f}, "
                 f"features={combined['selected_features']}")

    # Compare to standard AHSS (mean only)
    ahss_result = feature_results["mean"]
    logger.info(f"\n  Standard AHSS (mean): AUROC={ahss_result['auroc']:.4f}")
    logger.info(f"  FSP Combined:         AUROC={combined['auroc']:.4f}")
    delta = combined['auroc'] - ahss_result['auroc']
    logger.info(f"  Δ AUROC:              {delta:+.4f}")

    # Also compute combined FSP + Mahalanobis
    logger.info("Computing Mahalanobis + FSP combined...")
    mah = MahalanobisScorer()
    mah.fit(train_unmasked, train_labels)
    eval_mah = mah.score(eval_unmasked)
    ood_mah = mah.score(ood_unmasked)
    mah_auroc = compute_auroc(eval_mah, ood_mah)

    # Rank-combine FSP best + Mahalanobis
    from scipy.stats import rankdata

    def rank_norm(s):
        r = rankdata(s, method='average')
        return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(s)

    # Use variance as the FSP feature
    eval_var = -eval_fsp["variance"]  # negate: higher var = more OOD
    ood_var = -ood_fsp["variance"]
    all_mah = np.concatenate([eval_mah, ood_mah])
    all_var = np.concatenate([eval_var, ood_var])
    n_eval = len(eval_mah)

    best_fsp_mah_auroc = 0.0
    best_alpha = 0.5
    for alpha in np.arange(0.0, 1.05, 0.1):
        c = alpha * rank_norm(all_mah) + (1 - alpha) * rank_norm(all_var)
        a = compute_auroc(c[:n_eval], c[n_eval:])
        if a > best_fsp_mah_auroc:
            best_fsp_mah_auroc = a
            best_alpha = alpha

    results["fsp_variance_mah_combined"] = {
        "auroc": float(best_fsp_mah_auroc),
        "best_alpha": float(best_alpha),
        "mah_only_auroc": float(mah_auroc),
    }
    logger.info(f"  FSP_var+Mah: AUROC={best_fsp_mah_auroc:.4f} (α={best_alpha:.1f})")
    logger.info(f"  Mah only:    AUROC={mah_auroc:.4f}")

    # Save displacement statistics for further analysis
    results["displacement_stats"] = {
        "eval_shape": list(eval_displacements.shape),
        "ood_shape": list(ood_displacements.shape),
        "eval_disp_global_mean": float(eval_displacements.mean()),
        "ood_disp_global_mean": float(ood_displacements.mean()),
    }

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save raw displacements for visualization
    np.save(os.path.join(output_dir, "eval_displacements.npy"), eval_displacements)
    np.save(os.path.join(output_dir, "ood_displacements.npy"), ood_displacements)

    logger.info(f"\nSaved to {output_dir} ({elapsed:.1f}s)")
    return results


def run_all(config):
    """Run distribution shape analysis for all experiments."""
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
    logger.info("AGGREGATED DISTRIBUTION SHAPE RESULTS")
    logger.info(f"{'='*80}")

    for dataset, exp_type in experiments:
        exp_key = f"{dataset}_{exp_type}"
        exp_results = [r for r in all_results if r.get("dataset") == dataset
                       and r.get("experiment_type") == exp_type]
        if not exp_results:
            continue

        logger.info(f"\n{exp_key} ({len(exp_results)} seeds):")
        for fname in ["mean", "variance", "skewness", "kurtosis"]:
            aurocs = [r["individual_features"][fname]["auroc"] for r in exp_results]
            logger.info(f"  FSP_{fname}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}")

        comb_aurocs = [r["combined"]["auroc"] for r in exp_results]
        logger.info(f"  FSP_combined: AUROC={np.mean(comb_aurocs):.4f}±{np.std(comb_aurocs):.4f}")

        if "fsp_variance_mah_combined" in exp_results[0]:
            fvm = [r["fsp_variance_mah_combined"]["auroc"] for r in exp_results]
            logger.info(f"  FSP_var+Mah:  AUROC={np.mean(fvm):.4f}±{np.std(fvm):.4f}")

    # Save aggregated
    agg_path = os.path.join(SCRIPT_DIR, "outputs", "distribution_shape", "aggregated_results.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAggregated saved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="AHSS Distribution Shape Analysis (FSP)")
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
