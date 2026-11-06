"""
run_dahm_optimization.py — Phase 2: Differentiable Attention Head Mask Optimization.

Learns optimal masks for AHSS via gradient descent on ID data.
The objective is to find masks where ID embeddings are MOST STABLE,
which implicitly makes OOD embeddings MOST UNSTABLE under the same masks.

Usage:
    python run_dahm_optimization.py --dataset financedocs --experiment_type intra --seed 42
    python run_dahm_optimization.py --run_all
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
from models.layoutlmv3_classifier import LayoutLMv3DocumentClassifier
from models.differentiable_masking import DifferentiableAHMOptimizer
from models.attention_masking import AttentionHeadMasker
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dahm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_dahm_single(dataset, experiment_type, seed, config, args):
    """Run D-AHM optimization for a single experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    logger.info(f"\n{'='*70}\n  D-AHM Optimization: {exp_name}\n{'='*70}")

    model_dir = os.path.join(BASE_DIR, "outputs", "checkpoints", exp_name, "best_model")
    if not os.path.exists(model_dir):
        logger.error(f"Missing model: {model_dir}")
        return None

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

    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False

    # Load ID eval data
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    if dataset == "financedocs":
        data_path = os.path.join(BASE_DIR, "data", "colated_tensors")
    else:
        data_path = os.path.join(BASE_DIR, "data", "tobacco", "colated_tensors")

    full_ds = load_from_disk(data_path)
    if experiment_type == "intra":
        id_ds = filter_dataset_by_classes(full_ds, id_classes)
    else:
        id_ds = filter_dataset_by_classes(full_ds, id_classes)

    id_ds = encode_labels(id_ds, label2id)
    labels_arr = np.array(id_ds["labels"])
    indices = np.arange(len(id_ds))
    _, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )
    eval_ds = id_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=4, shuffle=True)

    # Run D-AHM optimization
    optimizer = DifferentiableAHMOptimizer(
        model=model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        lr=args.lr,
        num_steps=args.num_steps,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        target_masking_rate=args.target_masking_rate,
        sparsity_lambda=args.sparsity_lambda,
        device=device,
    )

    t0 = time.time()
    masks, metadata = optimizer.optimize(eval_dl, num_masks_to_produce=args.num_masks)
    elapsed = time.time() - t0
    logger.info(f"D-AHM optimization completed in {elapsed:.1f}s")

    # Save masks and metadata
    out_dir = os.path.join(BASE_DIR, "outputs", "dahm_masks", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    masker = AttentionHeadMasker(config.model.num_layers, config.model.num_heads)
    for i, mask in enumerate(masks):
        masker.save_mask(mask, os.path.join(out_dir, f"dahm_mask_{i}.npy"))

    # Save metadata
    meta_path = os.path.join(out_dir, "dahm_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {len(masks)} D-AHM masks + metadata to {out_dir}")

    del model
    torch.cuda.empty_cache()

    return {"exp_name": exp_name, "num_masks": len(masks),
            "elapsed": elapsed, "metadata_path": meta_path}


def main():
    parser = argparse.ArgumentParser(description="D-AHM: Differentiable mask optimization")
    parser.add_argument("--dataset", type=str, default="financedocs")
    parser.add_argument("--experiment_type", type=str, default="intra")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end", type=float, default=0.1)
    parser.add_argument("--target_masking_rate", type=float, default=0.15)
    parser.add_argument("--sparsity_lambda", type=float, default=1.0)
    parser.add_argument("--num_masks", type=int, default=5)
    args = parser.parse_args()

    config = get_default_config()

    if args.run_all:
        experiments = [
            ("financedocs", "intra"), ("financedocs", "cross"),
            ("tobacco", "intra"), ("tobacco", "cross"),
        ]
        for dataset, exp_type in experiments:
            for seed in config.seeds:
                run_dahm_single(dataset, exp_type, seed, config, args)
    else:
        run_dahm_single(args.dataset, args.experiment_type, args.seed, config, args)


if __name__ == "__main__":
    main()
