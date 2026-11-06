"""
run_phase1_all_seeds.py — Phase 1: Full AHSS + MC-Dropout evaluation across all seeds.

Runs all 4 experiments × 5 seeds:
  - financedocs intra, financedocs cross, tobacco intra, tobacco cross
  - seeds: 42, 123, 456, 789, 1024

For each experiment:
  1. AHSS scoring (25 masks: 5 AHM + 20 random)
  2. MC-Dropout baseline (25 stochastic passes)
  3. Mahalanobis + kNN baselines
  4. Combined scoring
  5. Save per-seed + aggregated results

Usage:
    python run_phase1_all_seeds.py
    python run_phase1_all_seeds.py --skip_mc_dropout   # skip MC-Dropout (faster)
    python run_phase1_all_seeds.py --seeds 42 123      # specific seeds
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

# Fix datasets 3.6.0 compatibility: 'List' was renamed to 'Sequence'
import utils.compat  # noqa: F401

from configs.config import get_default_config, get_label2id, get_id2label
from models.layoutlmv3_classifier import (
    LayoutLMv3DocumentClassifier,
    extract_embeddings_with_multiple_masks,
    extract_mc_dropout_embeddings,
)
from models.attention_masking import AttentionHeadMasker
from ood_methods.sensitivity_based import AHSSScorer, MCDropoutScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
LOG_DIR = os.path.join(MAIN_DIR, "logs")


def load_masks(ahm_dir, masker, num_extra=20, masking_percentages=None, seed=42):
    """Load AHM masks + generate random masks."""
    if masking_percentages is None:
        masking_percentages = [0.1, 0.2]
    masks, sources = [], []

    for i in range(100):
        path = os.path.join(ahm_dir, f"best_mask_{i}.npy")
        if os.path.exists(path):
            masks.append(masker.load_mask(path))
            sources.append(f"ahm_best_{i}")
        else:
            break

    if num_extra > 0:
        rng = np.random.RandomState(seed + 9999)
        per_pct = num_extra // len(masking_percentages)
        remainder = num_extra % len(masking_percentages)
        for p_idx, p in enumerate(masking_percentages):
            count = per_pct + (1 if p_idx < remainder else 0)
            for j in range(count):
                masks.append(masker.create_mask(p, seed=int(rng.randint(0, 2**31))))
                sources.append(f"random_p{p}_{j}")

    return masks, sources


def rank_normalize(eval_s, ood_s):
    """Rank-based normalization on concatenated eval+OOD scores."""
    all_s = np.concatenate([eval_s, ood_s])
    ranks = np.argsort(np.argsort(all_s)).astype(np.float64)
    ranks /= (len(ranks) - 1 + 1e-10)
    return ranks[:len(eval_s)], ranks[len(eval_s):]


def run_single_seed(dataset, experiment_type, seed, config, args):
    """Run full AHSS + MC-Dropout evaluation for one experiment/seed."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    logger.info(f"\n{'='*70}\n  {exp_name}\n{'='*70}")

    # Skip if results already exist
    existing_results_path = os.path.join(BASE_DIR, "outputs", "ahss_results", exp_name, "phase1_results.json")
    if os.path.exists(existing_results_path):
        logger.info(f"Results already exist at {existing_results_path}, loading and skipping.")
        with open(existing_results_path, "r") as f:
            return json.load(f)

    # Paths
    embedding_dir = os.path.join(BASE_DIR, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(BASE_DIR, "outputs", "ahm_masks", exp_name)
    model_dir = os.path.join(BASE_DIR, "outputs", "checkpoints", exp_name, "best_model")

    if not os.path.exists(embedding_dir):
        logger.error(f"Missing embeddings: {embedding_dir}")
        return None
    if not os.path.exists(model_dir):
        logger.error(f"Missing model: {model_dir}")
        return None

    # Load precomputed unmasked embeddings
    eval_unmasked = np.load(os.path.join(embedding_dir, "eval_embeddings.npy"))
    ood_unmasked = np.load(os.path.join(embedding_dir, "ood_embeddings.npy"))
    train_unmasked = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    train_labels = np.load(os.path.join(embedding_dir, "train_labels.npy"))
    eval_labels = np.load(os.path.join(embedding_dir, "eval_labels.npy"))

    logger.info(f"Embeddings: train={train_unmasked.shape}, eval={eval_unmasked.shape}, ood={ood_unmasked.shape}")

    # Setup dataset config
    dataset_config = config.financedocs if dataset == "financedocs" else config.tobacco
    if experiment_type == "intra":
        id_classes = dataset_config.id_classes
    else:
        id_classes = dataset_config.all_classes

    label2id = get_label2id(id_classes)
    id2label = get_id2label(id_classes)
    num_labels = len(id_classes)

    # Load model
    model = LayoutLMv3DocumentClassifier(
        model_name=model_dir, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
    )
    model.to(device)
    model.eval()

    # Load masks
    masker = AttentionHeadMasker(config.model.num_layers, config.model.num_heads)
    masks, mask_sources = load_masks(
        ahm_dir, masker, num_extra=args.num_extra_masks,
        masking_percentages=config.ahss.masking_percentages, seed=seed,
    )
    logger.info(f"Using {len(masks)} masks ({len([s for s in mask_sources if s.startswith('ahm')])} AHM + {len([s for s in mask_sources if s.startswith('random')])} random)")

    # Load data for model-based extraction
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    if dataset == "financedocs":
        data_path = os.path.join(BASE_DIR, "data", "colated_tensors")
    else:
        data_path = os.path.join(BASE_DIR, "data", "tobacco", "colated_tensors")

    full_ds = load_from_disk(data_path)

    if experiment_type == "intra":
        id_ds = filter_dataset_by_classes(full_ds, id_classes)
        ood_ds = filter_dataset_by_classes(full_ds, [dataset_config.ood_class])
    else:
        id_ds = filter_dataset_by_classes(full_ds, id_classes)
        if dataset == "financedocs":
            ood_path = os.path.join(BASE_DIR, "data", "tobacco", "colated_tensors")
        else:
            ood_path = os.path.join(BASE_DIR, "data", "colated_tensors")
        ood_ds = load_from_disk(ood_path)

    id_ds = encode_labels(id_ds, label2id)
    labels_arr = np.array(id_ds["labels"])
    indices = np.arange(len(id_ds))
    train_idx, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )
    eval_ds = id_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=32, shuffle=False)

    ood_dummy_labels = {lbl: num_labels for lbl in set(ood_ds["labels"])}
    ood_ds_encoded = encode_labels(ood_ds, ood_dummy_labels)
    ood_dl = create_dataloader(ood_ds_encoded, batch_size=32, shuffle=False)

    # ========== AHSS: Extract embeddings under masks ==========
    logger.info("--- AHSS: Multi-mask extraction ---")
    t0 = time.time()
    eval_unmasked_new, eval_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, eval_dl, masks, device=device, desc=f"{exp_name} eval",
    )
    _, ood_masked_list, _ = extract_embeddings_with_multiple_masks(
        model, ood_dl, masks, device=device,
        include_unmasked=False, desc=f"{exp_name} OOD",
    )
    # We need unmasked OOD too
    ood_unmasked_list = []
    model.set_attention_head_mask(None)
    all_ood_embs = []
    with torch.no_grad():
        for batch in ood_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            emb = model.extract_embeddings(input_ids=input_ids, attention_mask=attention_mask,
                                            bbox=bbox, pixel_values=pixel_values)
            all_ood_embs.append(emb.cpu().numpy())
    ood_unmasked_new = np.concatenate(all_ood_embs, axis=0)

    ahss_time = time.time() - t0
    logger.info(f"AHSS extraction done in {ahss_time:.1f}s")

    # Use freshly extracted embeddings
    eval_unmasked = eval_unmasked_new
    ood_unmasked = ood_unmasked_new

    # Compute AHSS scores
    scorer = AHSSScorer(normalize=config.ahss.normalize_embeddings)
    eval_ahss_raw = scorer.score_raw(eval_unmasked, eval_masked_list)
    ood_ahss_raw = scorer.score_raw(ood_unmasked, ood_masked_list)
    eval_ahss = scorer.score(eval_unmasked, eval_masked_list)
    ood_ahss = scorer.score(ood_unmasked, ood_masked_list)

    ahss_auroc = compute_auroc(eval_ahss, ood_ahss)
    ahss_fpr = compute_fpr_at_tpr(eval_ahss, ood_ahss)
    sep_ratio = float(ood_ahss_raw.mean() / (eval_ahss_raw.mean() + 1e-10))
    logger.info(f"AHSS: AUROC={ahss_auroc:.4f}, FPR@95={ahss_fpr:.4f}, sep={sep_ratio:.2f}x")

    # ========== MC-Dropout ==========
    # Free GPU memory from AHSS extraction before MC-Dropout
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    mc_auroc, mc_fpr = 0.0, 1.0
    if not args.skip_mc_dropout:
        logger.info("--- MC-Dropout: Stochastic extraction ---")
        t1 = time.time()
        eval_mc_samples, _ = extract_mc_dropout_embeddings(
            model, eval_dl, n_samples=args.mc_samples,
            device=device, desc=f"{exp_name} eval MC",
        )
        ood_mc_samples, _ = extract_mc_dropout_embeddings(
            model, ood_dl, n_samples=args.mc_samples,
            device=device, desc=f"{exp_name} OOD MC",
        )
        mc_time = time.time() - t1
        logger.info(f"MC-Dropout extraction done in {mc_time:.1f}s")

        mc_scorer = MCDropoutScorer(normalize=True)
        eval_mc = mc_scorer.score(eval_mc_samples)
        ood_mc = mc_scorer.score(ood_mc_samples)
        mc_auroc = compute_auroc(eval_mc, ood_mc)
        mc_fpr = compute_fpr_at_tpr(eval_mc, ood_mc)
        logger.info(f"MC-Dropout: AUROC={mc_auroc:.4f}, FPR@95={mc_fpr:.4f}")

    # ========== Mahalanobis + kNN baselines ==========
    mah = MahalanobisScorer()
    mah.fit(train_unmasked, train_labels)
    eval_mah = mah.score(eval_unmasked)
    ood_mah = mah.score(ood_unmasked)
    mah_auroc = compute_auroc(eval_mah, ood_mah)
    mah_fpr = compute_fpr_at_tpr(eval_mah, ood_mah)

    knn = KNNScorer(k=config.ood.knn_k, metric="euclidean")
    knn.fit(train_unmasked)
    eval_knn = knn.score(eval_unmasked)
    ood_knn = knn.score(ood_unmasked)
    knn_auroc = compute_auroc(eval_knn, ood_knn)
    knn_fpr = compute_fpr_at_tpr(eval_knn, ood_knn)

    logger.info(f"Mahalanobis: AUROC={mah_auroc:.4f}, FPR@95={mah_fpr:.4f}")
    logger.info(f"kNN: AUROC={knn_auroc:.4f}, FPR@95={knn_fpr:.4f}")

    # ========== Combined scoring (rank-normalized) ==========
    eval_ahss_rn, ood_ahss_rn = rank_normalize(eval_ahss, ood_ahss)
    eval_mah_rn, ood_mah_rn = rank_normalize(eval_mah, ood_mah)
    eval_knn_rn, ood_knn_rn = rank_normalize(eval_knn, ood_knn)

    best_mah_combo = {"alpha": 0, "auroc": 0, "fpr": 1}
    best_knn_combo = {"alpha": 0, "auroc": 0, "fpr": 1}

    for alpha in config.ahss.alpha_grid:
        ec = alpha * eval_mah_rn + (1 - alpha) * eval_ahss_rn
        oc = alpha * ood_mah_rn + (1 - alpha) * ood_ahss_rn
        a = compute_auroc(ec, oc)
        f = compute_fpr_at_tpr(ec, oc)
        if a > best_mah_combo["auroc"]:
            best_mah_combo = {"alpha": alpha, "auroc": a, "fpr": f}

        ec2 = alpha * eval_knn_rn + (1 - alpha) * eval_ahss_rn
        oc2 = alpha * ood_knn_rn + (1 - alpha) * ood_ahss_rn
        a2 = compute_auroc(ec2, oc2)
        f2 = compute_fpr_at_tpr(ec2, oc2)
        if a2 > best_knn_combo["auroc"]:
            best_knn_combo = {"alpha": alpha, "auroc": a2, "fpr": f2}

    # Clean up GPU
    del model
    torch.cuda.empty_cache()

    # ========== Save results ==========
    results = {
        "dataset": dataset,
        "experiment_type": experiment_type,
        "seed": seed,
        "num_masks": len(masks),
        "ahss": {"auroc": ahss_auroc, "fpr": ahss_fpr,
                 "id_mean": float(eval_ahss_raw.mean()), "ood_mean": float(ood_ahss_raw.mean()),
                 "sep_ratio": sep_ratio},
        "mc_dropout": {"auroc": mc_auroc, "fpr": mc_fpr, "n_samples": args.mc_samples},
        "mahalanobis": {"auroc": mah_auroc, "fpr": mah_fpr},
        "knn": {"auroc": knn_auroc, "fpr": knn_fpr},
        "ahss_mah_combo": best_mah_combo,
        "ahss_knn_combo": best_knn_combo,
    }

    out_dir = os.path.join(BASE_DIR, "outputs", "ahss_results", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    return results


def aggregate_results(all_results, config):
    """Aggregate results across seeds and produce final tables."""
    experiments = {}
    for r in all_results:
        key = f"{r['dataset']}_{r['experiment_type']}"
        if key not in experiments:
            experiments[key] = []
        experiments[key].append(r)

    logger.info("\n" + "=" * 90)
    logger.info("PHASE 1 — AGGREGATED RESULTS (mean ± std across seeds)")
    logger.info("=" * 90)

    header = f"{'Experiment':<25} {'AHSS':>14} {'MC-Drop':>14} {'Maha':>14} {'kNN':>14} {'AHSS+Mah':>14}"
    logger.info(header)
    logger.info("-" * 95)

    summary = {}
    for exp_key, results_list in sorted(experiments.items()):
        ahss_aurocs = [r["ahss"]["auroc"] for r in results_list]
        mc_aurocs = [r["mc_dropout"]["auroc"] for r in results_list]
        mah_aurocs = [r["mahalanobis"]["auroc"] for r in results_list]
        knn_aurocs = [r["knn"]["auroc"] for r in results_list]
        combo_aurocs = [r["ahss_mah_combo"]["auroc"] for r in results_list]

        def fmt(vals):
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        logger.info(f"{exp_key:<25} {fmt(ahss_aurocs):>14} {fmt(mc_aurocs):>14} "
                     f"{fmt(mah_aurocs):>14} {fmt(knn_aurocs):>14} {fmt(combo_aurocs):>14}")

        summary[exp_key] = {
            "ahss": {"mean": float(np.mean(ahss_aurocs)), "std": float(np.std(ahss_aurocs))},
            "mc_dropout": {"mean": float(np.mean(mc_aurocs)), "std": float(np.std(mc_aurocs))},
            "mahalanobis": {"mean": float(np.mean(mah_aurocs)), "std": float(np.std(mah_aurocs))},
            "knn": {"mean": float(np.mean(knn_aurocs)), "std": float(np.std(knn_aurocs))},
            "ahss_mah_combo": {"mean": float(np.mean(combo_aurocs)), "std": float(np.std(combo_aurocs))},
            "n_seeds": len(results_list),
        }

    logger.info("-" * 95)

    # FPR@95 table
    logger.info("\nFPR@95 (lower is better):")
    logger.info(f"{'Experiment':<25} {'AHSS':>14} {'MC-Drop':>14} {'Maha':>14} {'kNN':>14}")
    logger.info("-" * 75)
    for exp_key, results_list in sorted(experiments.items()):
        ahss_fprs = [r["ahss"]["fpr"] for r in results_list]
        mc_fprs = [r["mc_dropout"]["fpr"] for r in results_list]
        mah_fprs = [r["mahalanobis"]["fpr"] for r in results_list]
        knn_fprs = [r["knn"]["fpr"] for r in results_list]

        def fmt(vals):
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        logger.info(f"{exp_key:<25} {fmt(ahss_fprs):>14} {fmt(mc_fprs):>14} "
                     f"{fmt(mah_fprs):>14} {fmt(knn_fprs):>14}")

        summary[exp_key]["ahss_fpr"] = {"mean": float(np.mean(ahss_fprs)), "std": float(np.std(ahss_fprs))}
        summary[exp_key]["mc_fpr"] = {"mean": float(np.mean(mc_fprs)), "std": float(np.std(mc_fprs))}
        summary[exp_key]["mah_fpr"] = {"mean": float(np.mean(mah_fprs)), "std": float(np.std(mah_fprs))}
        summary[exp_key]["knn_fpr"] = {"mean": float(np.mean(knn_fprs)), "std": float(np.std(knn_fprs))}

    logger.info("-" * 75)

    # Save aggregated
    agg_dir = os.path.join(BASE_DIR, "outputs", "ahss_results")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, "phase1_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({"summary": summary, "all_results": all_results}, f, indent=2)
    logger.info(f"\nAggregated results saved: {agg_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Full AHSS + MC-Dropout evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: all 5)")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help="Experiments to run, e.g. 'financedocs_intra tobacco_cross'")
    parser.add_argument("--num_extra_masks", type=int, default=20)
    parser.add_argument("--mc_samples", type=int, default=25,
                        help="Number of MC-Dropout forward passes")
    parser.add_argument("--skip_mc_dropout", action="store_true")
    args = parser.parse_args()

    config = get_default_config()
    seeds = args.seeds or config.seeds

    all_experiments = [
        ("financedocs", "intra"),
        ("financedocs", "cross"),
        ("tobacco", "intra"),
        ("tobacco", "cross"),
    ]

    if args.experiments:
        filtered = []
        for exp_str in args.experiments:
            parts = exp_str.split("_")
            filtered.append((parts[0], parts[1]))
        all_experiments = filtered

    all_results = []
    total = len(all_experiments) * len(seeds)
    done = 0

    for dataset, exp_type in all_experiments:
        for seed in seeds:
            done += 1
            logger.info(f"\n[{done}/{total}] Running {dataset}_{exp_type} seed={seed}")
            t0 = time.time()
            try:
                result = run_single_seed(dataset, exp_type, seed, config, args)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"FAILED {dataset}_{exp_type} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
            elapsed = time.time() - t0
            logger.info(f"  [{done}/{total}] Done in {elapsed:.0f}s, {total-done} remaining")

    # Aggregate
    if all_results:
        aggregate_results(all_results, config)

    logger.info("\n=== PHASE 1 COMPLETE ===")


if __name__ == "__main__":
    main()
