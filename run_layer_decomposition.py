"""
run_layer_decomposition.py — Per-Layer Sensitivity Decomposition.

Decomposes AHSS sensitivity across the 12 transformer layers by creating
single-layer masks: mask heads in ONLY layer l, keep all other layers intact.

This produces a 12-layer sensitivity profile for each sample:
  S_l(x) = (1/T_l) Σ_t ||f(x; M_{l,t}) - f(x; 1)||²

where M_{l,t} masks heads only in layer l.

Outputs:
  - 12-dimensional heatmap showing per-layer OOD sensitivity
  - AUROC per layer (which layers matter most for OOD detection?)
  - Layer-weighted AHSS: combine per-layer scores optimally

Usage:
    python run_layer_decomposition.py --run_all
    python run_layer_decomposition.py --dataset financedocs --experiment_type intra --seed 42
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
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layer_decomp")


def generate_single_layer_masks(num_layers=12, num_heads=12,
                                 masks_per_layer=5, masking_pct=0.5, seed=42):
    """
    Generate masks that only mask heads in a SINGLE layer.

    For each of 12 layers, generate masks_per_layer masks that mask
    masking_pct of heads in THAT layer only, keeping all other layers at 1.

    Returns:
        layer_masks: dict mapping layer_idx → list of (num_layers, num_heads) tensors
    """
    rng = np.random.RandomState(seed)
    layer_masks = {}

    for layer in range(num_layers):
        masks = []
        n_to_mask = max(1, int(num_heads * masking_pct))
        for _ in range(masks_per_layer):
            mask = np.ones((num_layers, num_heads), dtype=np.float32)
            heads_to_mask = rng.choice(num_heads, size=n_to_mask, replace=False)
            mask[layer, heads_to_mask] = 0.0
            masks.append(torch.tensor(mask, dtype=torch.float32))
        layer_masks[layer] = masks

    return layer_masks


def run_experiment(dataset, experiment_type, seed, config):
    """Run per-layer decomposition for a single experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{dataset}_{experiment_type}_seed{seed}"
    base_dir = SCRIPT_DIR
    num_layers = config.model.num_layers
    num_heads = config.model.num_heads

    checkpoint_dir = os.path.join(base_dir, "outputs", "checkpoints", exp_name, "best_model")
    embedding_dir = os.path.join(base_dir, "outputs", "embeddings", exp_name)
    output_dir = os.path.join(base_dir, "outputs", "layer_decomposition", exp_name)

    results_path = os.path.join(output_dir, "layer_decomposition_results.json")
    if os.path.exists(results_path):
        logger.info(f"Results exist: {results_path}, skipping.")
        with open(results_path) as f:
            return json.load(f)

    for d in [embedding_dir, checkpoint_dir]:
        if not os.path.exists(d):
            logger.error(f"Missing: {d}")
            return None

    logger.info(f"\n{'='*70}")
    logger.info(f"LAYER DECOMPOSITION: {exp_name}")
    logger.info(f"{'='*70}")

    start = time.time()

    # Generate single-layer masks
    masks_per_layer = 5
    layer_masks = generate_single_layer_masks(
        num_layers=num_layers, num_heads=num_heads,
        masks_per_layer=masks_per_layer, masking_pct=0.5, seed=seed,
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
    _, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=seed, stratify=labels_arr,
    )
    eval_ds = id_ds.select(eval_idx)
    eval_dl = create_dataloader(eval_ds, batch_size=160, shuffle=False)

    if experiment_type == "cross":
        ood_dummy = {lbl: num_labels for lbl in set(ood_ds["labels"])}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    else:
        ood_dummy = {dataset_config.ood_class: num_labels}
        ood_ds = encode_labels(ood_ds, ood_dummy)
    ood_dl = create_dataloader(ood_ds, batch_size=160, shuffle=False)

    # Extract unmasked embeddings first
    logger.info("Extracting unmasked embeddings...")
    model.set_attention_head_mask(None)

    eval_embs_list = []
    with torch.no_grad():
        for batch in eval_dl:
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
            eval_embs_list.append(emb.cpu().numpy())
    eval_unmasked = np.concatenate(eval_embs_list, axis=0)

    ood_embs_list = []
    with torch.no_grad():
        for batch in ood_dl:
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
            ood_embs_list.append(emb.cpu().numpy())
    ood_unmasked = np.concatenate(ood_embs_list, axis=0)

    # Per-layer extraction and AHSS
    per_layer_results = {}

    for layer_idx in range(num_layers):
        masks = layer_masks[layer_idx]
        logger.info(f"Layer {layer_idx}: extracting under {len(masks)} masks...")

        # Extract eval under masks for this layer
        _, eval_masked, _ = extract_embeddings_with_multiple_masks(
            model, eval_dl, masks, device=device,
            desc=f"L{layer_idx} eval",
        )

        # Extract OOD under masks for this layer
        _, ood_masked, _ = extract_embeddings_with_multiple_masks(
            model, ood_dl, masks, device=device,
            desc=f"L{layer_idx} OOD",
        )

        # Compute AHSS for this layer
        ref_eval = eval_unmasked.astype(np.float64)
        ref_ood = ood_unmasked.astype(np.float64)
        norms_e = np.linalg.norm(ref_eval, axis=1, keepdims=True) + 1e-8
        norms_o = np.linalg.norm(ref_ood, axis=1, keepdims=True) + 1e-8
        ref_eval = ref_eval / norms_e
        ref_ood = ref_ood / norms_o

        eval_ahss = np.zeros(len(ref_eval), dtype=np.float64)
        ood_ahss = np.zeros(len(ref_ood), dtype=np.float64)

        for masked_e, masked_o in zip(eval_masked, ood_masked):
            me = masked_e.astype(np.float64)
            mo = masked_o.astype(np.float64)
            me = me / (np.linalg.norm(me, axis=1, keepdims=True) + 1e-8)
            mo = mo / (np.linalg.norm(mo, axis=1, keepdims=True) + 1e-8)
            eval_ahss += np.sum((me - ref_eval) ** 2, axis=1)
            ood_ahss += np.sum((mo - ref_ood) ** 2, axis=1)

        eval_ahss /= len(masks)
        ood_ahss /= len(masks)

        # AUROC for this layer
        auroc = compute_auroc(-eval_ahss, -ood_ahss)
        fpr = compute_fpr_at_tpr(-eval_ahss, -ood_ahss)

        per_layer_results[layer_idx] = {
            "layer": layer_idx,
            "auroc": float(auroc),
            "fpr_at_95": float(fpr),
            "id_mean_ahss": float(eval_ahss.mean()),
            "id_std_ahss": float(eval_ahss.std()),
            "ood_mean_ahss": float(ood_ahss.mean()),
            "ood_std_ahss": float(ood_ahss.std()),
            "separation_ratio": float(ood_ahss.mean() / (eval_ahss.mean() + 1e-10)),
        }

        logger.info(f"  Layer {layer_idx}: AUROC={auroc:.4f}, Sep={ood_ahss.mean()/(eval_ahss.mean()+1e-10):.2f}x")

    del model
    torch.cuda.empty_cache()

    # Find best and worst layers
    sorted_layers = sorted(per_layer_results.values(), key=lambda x: x["auroc"], reverse=True)

    results = {
        "experiment": exp_name, "dataset": dataset,
        "experiment_type": experiment_type, "seed": seed,
        "num_layers": num_layers,
        "masks_per_layer": masks_per_layer,
        "per_layer": {str(k): v for k, v in per_layer_results.items()},
        "best_layer": sorted_layers[0]["layer"],
        "best_layer_auroc": sorted_layers[0]["auroc"],
        "worst_layer": sorted_layers[-1]["layer"],
        "worst_layer_auroc": sorted_layers[-1]["auroc"],
        "layer_auroc_profile": [per_layer_results[l]["auroc"] for l in range(num_layers)],
        "layer_separation_profile": [per_layer_results[l]["separation_ratio"] for l in range(num_layers)],
    }

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"\nBest layer: {sorted_layers[0]['layer']} (AUROC={sorted_layers[0]['auroc']:.4f})")
    logger.info(f"Worst layer: {sorted_layers[-1]['layer']} (AUROC={sorted_layers[-1]['auroc']:.4f})")

    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {results_path} ({elapsed:.1f}s)")

    return results


def run_all(config):
    """Run layer decomposition for all experiments."""
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

    # Aggregate per-layer profiles
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATED LAYER DECOMPOSITION")
    logger.info(f"{'='*80}")

    for dataset, exp_type in experiments:
        exp_key = f"{dataset}_{exp_type}"
        exp_results = [r for r in all_results if r.get("dataset") == dataset
                       and r.get("experiment_type") == exp_type]
        if not exp_results:
            continue

        profiles = np.array([r["layer_auroc_profile"] for r in exp_results])
        mean_profile = profiles.mean(axis=0)
        std_profile = profiles.std(axis=0)

        logger.info(f"\n{exp_key} ({len(exp_results)} seeds):")
        logger.info(f"  {'Layer':>6} {'AUROC':>12}")
        for l in range(12):
            logger.info(f"  {l:>6} {mean_profile[l]:.4f}±{std_profile[l]:.4f}")
        logger.info(f"  Best: Layer {np.argmax(mean_profile)} ({mean_profile.max():.4f})")

    agg_path = os.path.join(SCRIPT_DIR, "outputs", "layer_decomposition", "aggregated_results.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-Layer Sensitivity Decomposition")
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
