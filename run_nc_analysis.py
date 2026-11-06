"""
run_nc_analysis.py — Phase 3: Neural Collapse Analysis.

Measures NC metrics (NC1-NC4) on all experiments with and without masks.
Tests the theoretical connection between NC degree and AHSS discriminability.

Usage:
    python run_nc_analysis.py --seed 42
    python run_nc_analysis.py --run_all
"""

import argparse
import os
import sys
import logging
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_default_config
from evaluation.nc_metrics import compute_all_nc_metrics, compute_nc_under_masks
from models.attention_masking import AttentionHeadMasker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nc_analysis")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_nc_single(dataset, experiment_type, seed, config):
    """Run NC analysis for one experiment using saved embeddings."""
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    logger.info(f"\n{'='*60}\n  NC Analysis: {exp_name}\n{'='*60}")

    emb_dir = os.path.join(BASE_DIR, "outputs", "embeddings", exp_name)
    ahm_dir = os.path.join(BASE_DIR, "outputs", "ahm_masks", exp_name)

    if not os.path.exists(emb_dir):
        logger.error(f"Missing: {emb_dir}")
        return None

    # Load embeddings
    train_emb = np.load(os.path.join(emb_dir, "train_embeddings.npy"))
    eval_emb = np.load(os.path.join(emb_dir, "eval_embeddings.npy"))
    train_labels = np.load(os.path.join(emb_dir, "train_labels.npy"))
    eval_labels = np.load(os.path.join(emb_dir, "eval_labels.npy"))

    # Load classifier weights (for NC3)
    clf_w_path = os.path.join(emb_dir, "classifier_weight.npy")
    clf_b_path = os.path.join(emb_dir, "classifier_bias.npy")
    clf_weight = np.load(clf_w_path) if os.path.exists(clf_w_path) else None
    clf_bias = np.load(clf_b_path) if os.path.exists(clf_b_path) else None

    # NC on training embeddings (unmasked)
    logger.info("Computing NC metrics on training embeddings...")
    train_nc = compute_all_nc_metrics(train_emb, train_labels, clf_weight, clf_bias)
    logger.info(f"  Train NC1={train_nc['nc1']:.4f}, NC2={train_nc['nc2']:.4f}, "
                f"NC3={train_nc.get('nc3', 'N/A')}, NC4={train_nc['nc4']:.4f}")

    # NC on eval embeddings (unmasked)
    eval_nc = compute_all_nc_metrics(eval_emb, eval_labels, clf_weight, clf_bias)
    logger.info(f"  Eval  NC1={eval_nc['nc1']:.4f}, NC2={eval_nc['nc2']:.4f}, "
                f"NC3={eval_nc.get('nc3', 'N/A')}, NC4={eval_nc['nc4']:.4f}")

    # NC under AHM masks (if AHSS results exist with per-mask embeddings)
    # For now, use the AHM ensemble embedding as a proxy
    ahm_eval_path = os.path.join(ahm_dir, "eval_ahm_embeddings.npy")
    nc_masked = None
    if os.path.exists(ahm_eval_path):
        ahm_eval = np.load(ahm_eval_path)
        ahm_nc = compute_all_nc_metrics(ahm_eval, eval_labels, clf_weight, clf_bias)
        logger.info(f"  AHM   NC1={ahm_nc['nc1']:.4f}, NC2={ahm_nc['nc2']:.4f}, "
                    f"NC4={ahm_nc['nc4']:.4f}")
        nc_masked = ahm_nc

    results = {
        "experiment": exp_name,
        "train_nc": train_nc,
        "eval_nc": eval_nc,
        "ahm_nc": nc_masked,
        "observations": {
            "nc1_train_vs_eval": train_nc["nc1"] - eval_nc["nc1"],
            "nc4_accuracy_train": train_nc["nc4"],
            "nc4_accuracy_eval": eval_nc["nc4"],
        },
    }

    # Save
    out_dir = os.path.join(BASE_DIR, "outputs", "nc_results", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nc_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Neural Collapse analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    config = get_default_config()
    experiments = [
        ("financedocs", "intra"), ("financedocs", "cross"),
        ("tobacco", "intra"), ("tobacco", "cross"),
    ]

    all_results = []
    seeds = config.seeds if args.run_all else [args.seed]

    for ds, et in experiments:
        for seed in seeds:
            r = run_nc_single(ds, et, seed, config)
            if r:
                all_results.append(r)

    # Summary table
    if all_results:
        logger.info("\n" + "=" * 80)
        logger.info("NC ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Experiment':<30} {'NC1':>8} {'NC2':>8} {'NC3':>8} {'NC4':>8}")
        logger.info("-" * 64)
        for r in all_results:
            nc = r["eval_nc"]
            nc3_str = f"{nc['nc3']:.4f}" if nc.get("nc3") is not None else "N/A"
            logger.info(f"{r['experiment']:<30} {nc['nc1']:>8.4f} {nc['nc2']:>8.4f} "
                        f"{nc3_str:>8} {nc['nc4']:>8.4f}")

    # Save aggregated
    if all_results:
        agg_path = os.path.join(BASE_DIR, "outputs", "nc_results", "nc_summary.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nSaved aggregated: {agg_path}")


if __name__ == "__main__":
    main()
