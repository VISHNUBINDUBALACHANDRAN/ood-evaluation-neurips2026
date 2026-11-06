"""
run_mahavgavg_ahm_crossarch.py — mahAvgAvg_AHM on ViT-B/16 and RoBERTa-base.

Fills the cross-architecture gap for mahAvgAvg_AHM (currently only on LayoutLMv3).

mahAvgAvg_AHM = Mahalanobis distance on multi-layer-averaged, multi-mask-ensembled embeddings:
    ψ(x) = (1/T) Σ_t [ (1/L) Σ_l z_l^CLS(x; M_t) ]
    Score(x) = -min_c (ψ(x) - μ_c)^T Σ^{-1} (ψ(x) - μ_c)

Usage:
    python run_mahavgavg_ahm_crossarch.py --run_all
    python run_mahavgavg_ahm_crossarch.py --architecture vit_cifar10 --ood_source intra --seed 42
"""

import argparse
import os
import sys
import logging
import json
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ood_methods.distance_based import MahalanobisScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Mask Generation (deterministic, matches existing scripts)
# ============================================================================

def generate_random_masks(num_masks, num_layers=12, num_heads=12, masking_pcts=(0.1, 0.2), seed=42):
    """Generate random binary attention head masks — identical to existing scripts."""
    rng = np.random.RandomState(seed)
    masks = []
    for i in range(num_masks):
        pct = masking_pcts[i % len(masking_pcts)]
        mask = np.ones((num_layers, num_heads), dtype=np.float32)
        n_to_mask = int(num_layers * num_heads * pct)
        flat = mask.flatten()
        indices = rng.choice(len(flat), size=n_to_mask, replace=False)
        flat[indices] = 0.0
        mask = flat.reshape(num_layers, num_heads)
        masks.append(torch.tensor(mask, dtype=torch.float32))
    return masks


# ============================================================================
# Data Loading — Functions imported from original scripts at runtime:
#   ViT CIFAR-10: from run_vit_extended_eval import load_cifar10_splits, load_cifar10_cross
#   ViT CIFAR-100: from run_cifar100_benchmark import load_cifar100_data, load_ood_dataset
#   RoBERTa: from run_roberta_experiments import load_sst2_data, load_ood_text_dataset
# This ensures EXACT same data splits as original experiments.
# ============================================================================


# ============================================================================
# Model Loading
# ============================================================================

def load_vit_model(checkpoint_dir, num_labels, device):
    from models.vit_classifier import ViTClassifier
    from utils.model_io import load_model_weights
    model = ViTClassifier(num_labels=num_labels)
    state = load_model_weights(os.path.join(checkpoint_dir, "best_model"), "model")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_roberta_model(checkpoint_dir, device):
    """Load RoBERTaClassifier from checkpoint (same class as fine-tuning script)."""
    from run_roberta_experiments import RoBERTaClassifier
    from utils.model_io import load_model_weights
    model = RoBERTaClassifier(num_labels=2)
    state = load_model_weights(os.path.join(checkpoint_dir, "best_model"), "model")
    # Handle version differences (position_ids may be missing in newer transformers)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Extraction: All-Layer CLS Under Mask (Core of mahAvgAvg_AHM)
# ============================================================================

def extract_all_layer_avgavg_vit(model, dataloader, device="cuda", desc="AvgAvg"):
    """
    Extract sequence-mean from all 12 transformer layers, average across layers.
    Faithful to Chen et al. 2022 AvgAvg: mean over sequence positions, then mean over layers.
    Returns: (N, 768) averaged embeddings, (N,) labels
    """
    model.eval()
    all_avgavg = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pixel_values)
            hidden_states = outputs["hidden_states"]
            # Skip embedding layer (index 0), use layers 1-12
            # Sequence-mean per layer (faithful Chen et al. 2022 AvgAvg)
            seqmean_stack = torch.stack([hs.mean(dim=1) for hs in hidden_states[1:]], dim=0)  # (12, B, 768)
            avgavg = seqmean_stack.mean(dim=0)  # (B, 768)
            all_avgavg.append(avgavg.cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    embeddings = np.concatenate(all_avgavg, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return embeddings, labels


def extract_all_layer_avgavg_roberta(model, dataloader, device="cuda", desc="AvgAvg"):
    """
    Extract sequence-mean from all 12 transformer layers for RoBERTa, average across layers.
    Faithful to Chen et al. 2022 AvgAvg: mean over sequence positions, then mean over layers.
    RoBERTaClassifier.forward() returns dict with 'hidden_states' key.
    Returns: (N, 768) averaged embeddings, (N,) labels
    """
    model.eval()
    all_avgavg = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs["hidden_states"]  # dict access for RoBERTaClassifier
            # Skip embedding layer (index 0), use layers 1-12
            # Sequence-mean per layer (faithful Chen et al. 2022 AvgAvg)
            seqmean_stack = torch.stack([hs.mean(dim=1) for hs in hidden_states[1:]], dim=0)  # (12, B, 768)
            avgavg = seqmean_stack.mean(dim=0)  # (B, 768)
            all_avgavg.append(avgavg.cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    embeddings = np.concatenate(all_avgavg, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return embeddings, labels


# ============================================================================
# Mask Management — Both ViT and RoBERTa use model.set_attention_head_mask()
# ============================================================================

def set_mask(model, mask):
    """Set attention head mask on model (works for both ViT and RoBERTa)."""
    model.set_attention_head_mask(mask)


def clear_mask(model):
    """Clear attention head mask."""
    model.set_attention_head_mask(None)


# ============================================================================
# mahAvgAvg_AHM Computation
# ============================================================================

def compute_mahavgavg_ahm(
    model,
    train_loader,
    eval_loader,
    ood_loader,
    masks,
    train_labels_arr,
    arch_type,  # "vit" or "roberta"
    device="cuda",
    exp_name="",
):
    """
    Compute mahAvgAvg_AHM: Mahalanobis on AvgAvg embeddings under multiple masks.

    Following the same convention as existing mahAHM on ViT/RoBERTa:
    - Train: extracted UNMASKED (single pass), used to fit Mahalanobis
    - Eval/OOD: extracted under T masks, averaged across masks

    This is consistent with run_vit_extended_eval.py mahAHM which fits
    on unmasked train and scores masked eval/OOD.
    """
    T = len(masks)
    extract_fn = extract_all_layer_avgavg_vit if arch_type == "vit" else extract_all_layer_avgavg_roberta

    # Step 1: Extract UNMASKED AvgAvg for train (single pass)
    logger.info(f"  [{exp_name}] Extracting unmasked train AvgAvg embeddings...")
    clear_mask(model)
    train_avgavg, _ = extract_fn(model, train_loader, device=device,
                                  desc="Train AvgAvg (unmasked)")
    logger.info(f"  [{exp_name}] Train AvgAvg shape: {train_avgavg.shape}")

    # Step 2: Extract masked AvgAvg for eval and OOD, average across masks
    eval_sum = None
    ood_sum = None

    for mask_idx, mask in enumerate(masks):
        logger.info(f"  [{exp_name}] Mask {mask_idx+1}/{T}")
        set_mask(model, mask)

        # Extract AvgAvg for eval
        eval_avgavg, _ = extract_fn(model, eval_loader, device=device,
                                     desc=f"Eval mask {mask_idx+1}/{T}")
        if eval_sum is None:
            eval_sum = eval_avgavg.copy()
        else:
            eval_sum += eval_avgavg

        # Extract AvgAvg for OOD
        ood_avgavg, _ = extract_fn(model, ood_loader, device=device,
                                    desc=f"OOD mask {mask_idx+1}/{T}")
        if ood_sum is None:
            ood_sum = ood_avgavg.copy()
        else:
            ood_sum += ood_avgavg

        del eval_avgavg, ood_avgavg

    clear_mask(model)

    # Average across masks
    eval_avgavg_ahm = eval_sum / T
    ood_avgavg_ahm = ood_sum / T

    # Fit Mahalanobis on UNMASKED train AvgAvg (matches existing mahAHM convention)
    maha = MahalanobisScorer()
    maha.fit(train_avgavg, train_labels_arr)

    # Score
    eval_scores = maha.score(eval_avgavg_ahm)
    ood_scores = maha.score(ood_avgavg_ahm)

    auroc = compute_auroc(eval_scores, ood_scores)
    fpr = compute_fpr_at_tpr(eval_scores, ood_scores)

    return {
        "auroc": float(auroc),
        "fpr_at_95": float(fpr),
        "n_masks": T,
        "train_shape": list(train_avgavg.shape),
        "eval_shape": list(eval_avgavg_ahm.shape),
        "ood_shape": list(ood_avgavg_ahm.shape),
        "train_fitting": "unmasked_avgavg",
        "eval_ood_scoring": "masked_avgavg_averaged",
    }


# ============================================================================
# Run Single Experiment
# ============================================================================

def run_experiment(architecture, ood_source, seed, base_output_dir, num_classes=100):
    """Run mahAvgAvg_AHM for a single (architecture, ood_source, seed)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine experiment name and output paths
    if architecture == "vit_cifar10":
        if ood_source == "intra":
            exp_config = "cifar10_intra"
            checkpoint_dir = os.path.join(base_output_dir, "vit_checkpoints", f"cifar10_intra_seed{seed}")
        elif ood_source == "svhn":
            exp_config = "cifar10_svhn"
            checkpoint_dir = os.path.join(base_output_dir, "vit_checkpoints", f"cifar10_svhn_seed{seed}")
        else:
            raise ValueError(f"Unknown ood_source for vit_cifar10: {ood_source}")
        result_dir = os.path.join(base_output_dir, "vit_results", f"{exp_config}_seed{seed}")
    elif architecture == "vit_cifar100":
        # ctag matches the naming used by run_cifar100_benchmark.py so the
        # class-count ablation reads the right subset checkpoint. Empty for
        # num_classes=100, which preserves the original paths exactly.
        ctag = "" if num_classes >= 100 else f"_c{num_classes}"
        exp_config = f"cifar100{ctag}_{ood_source}"
        checkpoint_dir = os.path.join(base_output_dir, "cifar100_checkpoints", f"cifar100{ctag}_seed{seed}")
        result_dir = os.path.join(base_output_dir, "cifar100_results", f"{exp_config}_seed{seed}")
    elif architecture == "roberta":
        exp_config = f"sst2_{ood_source}"
        checkpoint_dir = os.path.join(base_output_dir, "roberta_checkpoints", f"sst2_seed{seed}")
        result_dir = os.path.join(base_output_dir, "roberta_results", f"{exp_config}_seed{seed}")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    exp_name = f"{exp_config}_seed{seed}"
    result_path = os.path.join(result_dir, "mahavgavg_ahm_results.json")

    # Check if already done
    if os.path.exists(result_path):
        logger.info(f"SKIPPING (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint not found: {checkpoint_dir}")
        return None

    logger.info(f"\n{'='*70}")
    logger.info(f"mahAvgAvg_AHM: {exp_name} | arch={architecture}")
    logger.info(f"{'='*70}")
    start_time = time.time()

    # Load data and model
    if architecture == "vit_cifar10":
        # Import data loading from original ViT eval script for exact reproducibility
        from run_vit_extended_eval import load_cifar10_splits as vit_load_cifar10_intra
        from run_vit_extended_eval import load_cifar10_cross as vit_load_cifar10_cross
        if ood_source == "intra":
            train_ds, eval_ds, ood_ds, _, _ = vit_load_cifar10_intra(seed, ood_class=0)
            num_labels = 9
        else:
            train_ds, eval_ds, ood_ds, _, _ = vit_load_cifar10_cross(seed, ood_source="svhn")
            num_labels = 10
        model = load_vit_model(checkpoint_dir, num_labels, device)
        batch_size = 512
        num_workers = 4
        arch_type = "vit"

    elif architecture == "vit_cifar100":
        # Import data loading from CIFAR-100 benchmark script
        from run_cifar100_benchmark import load_cifar100_data, load_ood_dataset
        train_ds, eval_ds, test_raw, transform = load_cifar100_data(seed, num_classes=num_classes)
        ood_ds = load_ood_dataset(ood_source, transform)
        num_labels = num_classes
        model = load_vit_model(checkpoint_dir, num_labels, device)
        batch_size = 512
        num_workers = 4
        arch_type = "vit"

    elif architecture == "roberta":
        from run_roberta_experiments import load_sst2_data as roberta_load_sst2
        from run_roberta_experiments import load_ood_text_dataset as roberta_load_ood
        from transformers import RobertaTokenizer
        cache_dir = os.path.join(SCRIPT_DIR, "data", "hf_cache")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=cache_dir)
        train_ds, eval_ds = roberta_load_sst2(seed, tokenizer)
        ood_ds = roberta_load_ood(ood_source, tokenizer)
        model = load_roberta_model(checkpoint_dir, device)
        batch_size = 64
        num_workers = 2
        arch_type = "roberta"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    # Get train labels
    train_labels = []
    for batch in train_loader:
        train_labels.append(batch["labels"].numpy())
    train_labels_arr = np.concatenate(train_labels, axis=0)

    # Generate masks (deterministic, same as AHSS/mahAHM experiments)
    masks = generate_random_masks(25, seed=seed)
    logger.info(f"Generated 25 random masks (seed={seed})")

    # Compute mahAvgAvg_AHM
    result = compute_mahavgavg_ahm(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        ood_loader=ood_loader,
        masks=masks,
        train_labels_arr=train_labels_arr,
        arch_type=arch_type,
        device=device,
        exp_name=exp_name,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - start_time
    result.update({
        "experiment": exp_config,
        "seed": seed,
        "architecture": architecture,
        "method": "mahAvgAvg_AHM",
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Save
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  mahAvgAvg_AHM: AUROC={result['auroc']:.4f}, FPR@95={result['fpr_at_95']:.4f}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info(f"  Saved: {result_path}")

    return result


# ============================================================================
# Run All Experiments
# ============================================================================

ALL_CONFIGS = [
    # (architecture, ood_source)
    ("vit_cifar10", "intra"),
    ("vit_cifar10", "svhn"),
    ("vit_cifar100", "svhn"),
    ("vit_cifar100", "dtd"),
    ("roberta", "mnli"),
    ("roberta", "agnews"),
    ("roberta", "20news"),
]

SEEDS = [42, 123, 456, 789, 1024]


def run_all(base_output_dir):
    """Run all 35 experiments (7 configs × 5 seeds)."""
    all_results = []
    total = len(ALL_CONFIGS) * len(SEEDS)
    done = 0

    logger.info(f"Starting mahAvgAvg_AHM cross-architecture evaluation")
    logger.info(f"Total experiments: {total}")
    logger.info(f"Configs: {len(ALL_CONFIGS)}, Seeds: {len(SEEDS)}")
    logger.info(f"Output dir: {base_output_dir}")

    for arch, ood_source in ALL_CONFIGS:
        for seed in SEEDS:
            done += 1
            logger.info(f"\n[{done}/{total}] {arch} / {ood_source} / seed={seed}")
            result = run_experiment(arch, ood_source, seed, base_output_dir)
            if result:
                all_results.append(result)

    # Aggregate
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATION: mahAvgAvg_AHM Cross-Architecture Results")
    logger.info(f"{'='*70}")

    summary = {}
    for arch, ood_source in ALL_CONFIGS:
        exp_config = get_exp_config(arch, ood_source)
        aurocs = []
        fprs = []
        for seed in SEEDS:
            for r in all_results:
                if r.get("experiment") == exp_config and r.get("seed") == seed:
                    aurocs.append(r["auroc"])
                    fprs.append(r["fpr_at_95"])

        if aurocs:
            summary[exp_config] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "mean_fpr": float(np.mean(fprs)),
                "std_fpr": float(np.std(fprs)),
                "per_seed_auroc": aurocs,
                "per_seed_fpr": fprs,
                "n_seeds": len(aurocs),
            }
            logger.info(f"  {exp_config:<25} AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
                        f"FPR@95={np.mean(fprs):.4f}±{np.std(fprs):.4f}")

    # Save aggregated
    agg_dir = os.path.join(base_output_dir, "mahavgavg_ahm_results")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nAggregated results saved: {agg_path}")

    # Save all per-seed results
    all_path = os.path.join(agg_dir, "all_per_seed_results.json")
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Per-seed results saved: {all_path}")

    return summary


def get_exp_config(arch, ood_source):
    if arch == "vit_cifar10":
        return f"cifar10_{ood_source}"
    elif arch == "vit_cifar100":
        return f"cifar100_{ood_source}"
    elif arch == "roberta":
        return f"sst2_{ood_source}"
    return f"{arch}_{ood_source}"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="mahAvgAvg_AHM Cross-Architecture Evaluation")
    parser.add_argument("--architecture", type=str, choices=["vit_cifar10", "vit_cifar100", "roberta"])
    parser.add_argument("--ood_source", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="vit_cifar100 only: ID class subset (100 = original run)")
    parser.add_argument("--base_output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "outputs"))
    args = parser.parse_args()

    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.run_all:
        run_all(args.base_output_dir)
    elif args.architecture and args.ood_source and args.seed:
        run_experiment(args.architecture, args.ood_source, args.seed, args.base_output_dir,
                       num_classes=args.num_classes)
    else:
        parser.error("Either --run_all or (--architecture, --ood_source, --seed) required")


if __name__ == "__main__":
    main()
