"""
run_xmaha_baseline.py — X-Mahalanobis Baseline for AHSS NeurIPS 2026 Paper.

X-Maha: multi-layer feature fusion with adaptive importance weights +
Mahalanobis distance. Reference: Zhang et al., NeurIPS 2025.

    z_fused(x) = Σ_{l=1}^{L} w_l · z_l(x)
    Score = -min_c (z_fused - μ_c)^T Σ^{-1} (z_fused - μ_c)

For LayoutLMv3: uses pre-computed layer embeddings from outputs/embeddings/
For ViT: extracts per-layer CLS from checkpoints (not pre-saved)

Usage:
    python run_xmaha_baseline.py --run_all
    python run_xmaha_baseline.py --architecture vit --experiment cifar10_svhn --seed 42
"""

import argparse
import os
import sys
import logging
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import minimize
from sklearn.covariance import ShrunkCovariance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.model_io import load_model_weights, model_weights_exist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# X-Mahalanobis scorer
# ============================================================================

class XMahalanobisScorer:
    """
    X-Mahalanobis: Multi-layer feature fusion with Mahalanobis distance.
    
    z_fused(x) = Σ_{l} w_l * z_l(x)  (weighted combination)
    Score = -min_c (z_fused - μ_c)^T Σ^{-1} (z_fused - μ_c)
    """
    
    def __init__(self):
        self.weights = None
        self.class_means = {}
        self.precision_matrix = None
    
    def _fuse(self, layer_embs, weights):
        """Fuse layer embeddings: layer_embs shape (L, N, D)."""
        return np.tensordot(weights, layer_embs, axes=([0], [0]))  # (N, D)
    
    def _fit_mahalanobis(self, fused, labels):
        """Fit class-conditional Gaussians on fused features."""
        classes = np.unique(labels)
        self.class_means = {}
        for c in classes:
            mask = labels == c
            self.class_means[c] = fused[mask].mean(axis=0)
        
        centered = np.zeros_like(fused)
        for c in classes:
            mask = labels == c
            centered[mask] = fused[mask] - self.class_means[c]
        
        cov_est = ShrunkCovariance(shrinkage=0.01)
        cov_est.fit(centered)
        self.precision_matrix = cov_est.precision_
    
    def _score(self, fused):
        """Compute Mahalanobis distance scores (higher = more ID)."""
        N = fused.shape[0]
        num_classes = len(self.class_means)
        distances = np.zeros((N, num_classes))
        for idx, (c, mean) in enumerate(self.class_means.items()):
            diff = fused - mean
            distances[:, idx] = np.sum(diff @ self.precision_matrix * diff, axis=1)
        return -distances.min(axis=1)
    
    def fit(self, train_layers, train_labels, eval_layers=None, ood_layers=None, 
            optimize=True, max_train_for_opt=5000):
        """
        Fit X-Maha. train_layers: (L, N_train, D) array.
        If optimize=True and eval/ood provided, optimize weights via Nelder-Mead.
        Subsamples training set during optimization for speed when N_train is large.
        """
        L = train_layers.shape[0]
        
        if optimize and eval_layers is not None and ood_layers is not None:
            logger.info(f"Optimizing X-Maha weights over {L} layers...")
            
            # Subsample training set for optimization speed
            N_train = train_layers.shape[1]
            if N_train > max_train_for_opt:
                rng = np.random.RandomState(42)
                idx = rng.choice(N_train, max_train_for_opt, replace=False)
                opt_train_layers = train_layers[:, idx, :]
                opt_train_labels = train_labels[idx]
                logger.info(f"  Subsampled {N_train} -> {max_train_for_opt} for optimization")
            else:
                opt_train_layers = train_layers
                opt_train_labels = train_labels
            
            def objective(raw_w):
                w = np.exp(raw_w) / np.exp(raw_w).sum()
                fused_train = self._fuse(opt_train_layers, w)
                self._fit_mahalanobis(fused_train, opt_train_labels)
                fused_eval = self._fuse(eval_layers, w)
                fused_ood = self._fuse(ood_layers, w)
                auroc = compute_auroc(self._score(fused_eval), self._score(fused_ood))
                return -auroc
            
            result = minimize(objective, np.zeros(L), method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5})
            raw_w = result.x
            self.weights = np.exp(raw_w) / np.exp(raw_w).sum()
            logger.info(f"Optimized weights: {np.array2string(self.weights, precision=4)}")
            logger.info(f"Best validation AUROC: {-result.fun:.4f}")
        else:
            self.weights = np.ones(L) / L
        
        # Final fit uses ALL training data
        fused_train = self._fuse(train_layers, self.weights)
        self._fit_mahalanobis(fused_train, train_labels)
    
    def score(self, layer_embs):
        """Score: layer_embs shape (L, N, D). Returns (N,) scores."""
        fused = self._fuse(layer_embs, self.weights)
        return self._score(fused)


# ============================================================================
# LayoutLMv3 experiments (pre-computed embeddings)
# ============================================================================

def run_xmaha_layoutlmv3(experiment, seed, output_base):
    """Run X-Maha on LayoutLMv3 using pre-computed layer embeddings."""
    
    exp_name = f"{experiment}_seed{seed}"
    emb_dir = os.path.join(output_base, "embeddings", exp_name)
    result_dir = os.path.join(output_base, "xmaha_results", exp_name)
    result_path = os.path.join(result_dir, "xmaha_results.json")
    
    if os.path.exists(result_path):
        logger.info(f"Skipping (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)
    
    # Check layer embeddings exist
    for split in ["train", "eval", "ood"]:
        path = os.path.join(emb_dir, f"{split}_layer_embeddings.npy")
        if not os.path.exists(path):
            logger.error(f"Layer embeddings not found: {path}")
            return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"X-MAHA: {exp_name} (LayoutLMv3)")
    logger.info(f"{'='*60}")
    start = time.time()
    
    # Load pre-computed layer embeddings: shape (12, N, 768)
    train_layers = np.load(os.path.join(emb_dir, "train_layer_embeddings.npy"))
    eval_layers = np.load(os.path.join(emb_dir, "eval_layer_embeddings.npy"))
    ood_layers = np.load(os.path.join(emb_dir, "ood_layer_embeddings.npy"))
    train_labels = np.load(os.path.join(emb_dir, "train_labels.npy"))
    
    logger.info(f"Train layers: {train_layers.shape}, Eval: {eval_layers.shape}, OOD: {ood_layers.shape}")
    
    results = {"experiment": experiment, "seed": seed, "architecture": "LayoutLMv3-base"}
    
    # X-Maha (uniform)
    logger.info("X-Maha (uniform weights)...")
    xmu = XMahalanobisScorer()
    xmu.fit(train_layers, train_labels, optimize=False)
    eval_scores_u = xmu.score(eval_layers)
    ood_scores_u = xmu.score(ood_layers)
    auroc_u = compute_auroc(eval_scores_u, ood_scores_u)
    fpr_u = compute_fpr_at_tpr(eval_scores_u, ood_scores_u)
    results["xmaha_uniform"] = {"auroc": float(auroc_u), "fpr_at_95": float(fpr_u)}
    logger.info(f"  Uniform: AUROC={auroc_u:.4f}, FPR@95={fpr_u:.4f}")
    
    # X-Maha (optimized)
    logger.info("X-Maha (optimized weights)...")
    xmo = XMahalanobisScorer()
    xmo.fit(train_layers, train_labels, eval_layers=eval_layers, ood_layers=ood_layers, optimize=True)
    eval_scores_o = xmo.score(eval_layers)
    ood_scores_o = xmo.score(ood_layers)
    auroc_o = compute_auroc(eval_scores_o, ood_scores_o)
    fpr_o = compute_fpr_at_tpr(eval_scores_o, ood_scores_o)
    results["xmaha_optimized"] = {
        "auroc": float(auroc_o), "fpr_at_95": float(fpr_o),
        "weights": xmo.weights.tolist()
    }
    logger.info(f"  Optimized: AUROC={auroc_o:.4f}, FPR@95={fpr_o:.4f}")
    
    # Single-layer Mahalanobis reference (last layer)
    from ood_methods.distance_based import MahalanobisScorer
    logger.info("Mahalanobis (last layer, reference)...")
    maha = MahalanobisScorer()
    maha.fit(train_layers[-1], train_labels)
    eval_maha = maha.score(eval_layers[-1])
    ood_maha = maha.score(ood_layers[-1])
    auroc_m = compute_auroc(eval_maha, ood_maha)
    fpr_m = compute_fpr_at_tpr(eval_maha, ood_maha)
    results["mahalanobis_last_layer"] = {"auroc": float(auroc_m), "fpr_at_95": float(fpr_m)}
    logger.info(f"  Maha (last): AUROC={auroc_m:.4f}, FPR@95={fpr_m:.4f}")
    
    # Per-layer Mahalanobis (shows which layers are best)
    logger.info("Per-layer Mahalanobis analysis...")
    per_layer = {}
    for l in range(train_layers.shape[0]):
        m = MahalanobisScorer()
        m.fit(train_layers[l], train_labels)
        a = compute_auroc(m.score(eval_layers[l]), m.score(ood_layers[l]))
        f = compute_fpr_at_tpr(m.score(eval_layers[l]), m.score(ood_layers[l]))
        per_layer[f"layer_{l+1}"] = {"auroc": float(a), "fpr_at_95": float(f)}
    results["per_layer_mahalanobis"] = per_layer
    
    results["elapsed_seconds"] = time.time() - start
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {result_path}")
    return results


# ============================================================================
# ViT experiments (extract from checkpoints)
# ============================================================================

def extract_vit_all_layer_cls(model, dataloader, device="cuda"):
    """Extract CLS token from all 12 transformer layers of ViT."""
    model.eval()
    model.to(device)
    
    layer_embs = [[] for _ in range(13)]  # 0=embedding layer, 1-12=transformer layers
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting layers"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pixel_values)
            hidden_states = outputs["hidden_states"]  # tuple of (batch, seq, 768)
            
            for l in range(len(hidden_states)):
                cls = hidden_states[l][:, 0, :].cpu().numpy()
                layer_embs[l].append(cls)
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    # Stack: shape (13, N, 768), skip embedding layer -> (12, N, 768)
    result = np.stack([np.concatenate(layer_embs[l]) for l in range(1, 13)])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return result, labels


def run_xmaha_vit(experiment, seed, output_base):
    """Run X-Maha on ViT-B/16."""
    from models.vit_classifier import ViTClassifier
    from torch.utils.data import DataLoader
    
    exp_name = f"{experiment}_seed{seed}"
    ckpt_dir = os.path.join(output_base, "vit_checkpoints", exp_name, "best_model")
    result_dir = os.path.join(output_base, "xmaha_results", f"vit_{exp_name}")
    result_path = os.path.join(result_dir, "xmaha_results.json")
    
    if os.path.exists(result_path):
        logger.info(f"Skipping (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)
    
    if not model_weights_exist(ckpt_dir, "model"):
        logger.error(f"Checkpoint not found: {ckpt_dir}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"X-MAHA: vit_{exp_name}")
    logger.info(f"{'='*60}")
    start = time.time()
    
    # Load data using existing helpers
    from run_vit_extended_eval import load_cifar10_splits, load_cifar10_cross
    
    if experiment == "cifar10_intra":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_splits(seed, ood_class=0)
        num_labels = 9
    elif experiment == "cifar10_svhn":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_cross(seed, ood_source="svhn")
        num_labels = 10
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTClassifier(num_labels=num_labels)
    state = load_model_weights(ckpt_dir, "model")
    model.load_state_dict(state)
    model.to(device)
    
    # Extract per-layer CLS embeddings
    train_layers, train_labels = extract_vit_all_layer_cls(model, train_loader, device)
    eval_layers, eval_labels = extract_vit_all_layer_cls(model, eval_loader, device)
    ood_layers, _ = extract_vit_all_layer_cls(model, ood_loader, device)
    
    del model
    torch.cuda.empty_cache()
    
    logger.info(f"Train layers: {train_layers.shape}, Eval: {eval_layers.shape}, OOD: {ood_layers.shape}")
    
    # Save extracted layer embeddings for future use
    layer_emb_dir = os.path.join(result_dir, "layer_embeddings")
    os.makedirs(layer_emb_dir, exist_ok=True)
    np.save(os.path.join(layer_emb_dir, "train_layer_embeddings.npy"), train_layers)
    np.save(os.path.join(layer_emb_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(layer_emb_dir, "eval_layer_embeddings.npy"), eval_layers)
    np.save(os.path.join(layer_emb_dir, "ood_layer_embeddings.npy"), ood_layers)
    
    results = {"experiment": experiment, "seed": seed, "architecture": "ViT-B/16"}
    
    # X-Maha (uniform)
    logger.info("X-Maha (uniform)...")
    xmu = XMahalanobisScorer()
    xmu.fit(train_layers, train_labels, optimize=False)
    eval_scores_u = xmu.score(eval_layers)
    ood_scores_u = xmu.score(ood_layers)
    auroc_u = compute_auroc(eval_scores_u, ood_scores_u)
    fpr_u = compute_fpr_at_tpr(eval_scores_u, ood_scores_u)
    results["xmaha_uniform"] = {"auroc": float(auroc_u), "fpr_at_95": float(fpr_u)}
    logger.info(f"  Uniform: AUROC={auroc_u:.4f}, FPR@95={fpr_u:.4f}")
    
    # X-Maha (optimized)
    logger.info("X-Maha (optimized)...")
    xmo = XMahalanobisScorer()
    xmo.fit(train_layers, train_labels, eval_layers=eval_layers, ood_layers=ood_layers, optimize=True)
    eval_scores_o = xmo.score(eval_layers)
    ood_scores_o = xmo.score(ood_layers)
    auroc_o = compute_auroc(eval_scores_o, ood_scores_o)
    fpr_o = compute_fpr_at_tpr(eval_scores_o, ood_scores_o)
    results["xmaha_optimized"] = {
        "auroc": float(auroc_o), "fpr_at_95": float(fpr_o),
        "weights": xmo.weights.tolist()
    }
    logger.info(f"  Optimized: AUROC={auroc_o:.4f}, FPR@95={fpr_o:.4f}")
    
    # Single-layer reference
    from ood_methods.distance_based import MahalanobisScorer
    logger.info("Mahalanobis (last layer, reference)...")
    maha = MahalanobisScorer()
    maha.fit(train_layers[-1], train_labels)
    eval_maha = maha.score(eval_layers[-1])
    ood_maha = maha.score(ood_layers[-1])
    auroc_m = compute_auroc(eval_maha, ood_maha)
    fpr_m = compute_fpr_at_tpr(eval_maha, ood_maha)
    results["mahalanobis_last_layer"] = {"auroc": float(auroc_m), "fpr_at_95": float(fpr_m)}
    logger.info(f"  Maha (last): AUROC={auroc_m:.4f}, FPR@95={fpr_m:.4f}")
    
    # Per-layer analysis
    logger.info("Per-layer Mahalanobis analysis...")
    per_layer = {}
    for l in range(train_layers.shape[0]):
        m = MahalanobisScorer()
        m.fit(train_layers[l], train_labels)
        a = compute_auroc(m.score(eval_layers[l]), m.score(ood_layers[l]))
        f = compute_fpr_at_tpr(m.score(eval_layers[l]), m.score(ood_layers[l]))
        per_layer[f"layer_{l+1}"] = {"auroc": float(a), "fpr_at_95": float(f)}
    results["per_layer_mahalanobis"] = per_layer
    
    results["elapsed_seconds"] = time.time() - start
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {result_path}")
    return results


# ============================================================================
# Aggregation & Comparison
# ============================================================================

def aggregate_and_compare(output_base, seeds):
    """Aggregate X-Maha results and compare with existing AHSS methods."""
    
    summary = {}
    
    # LayoutLMv3
    for exp in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        results_list = []
        for s in seeds:
            path = os.path.join(output_base, "xmaha_results", f"{exp}_seed{s}", "xmaha_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results_list.append(json.load(f))
        if results_list:
            summary[f"layoutlmv3_{exp}"] = _aggregate_method_results(results_list)
    
    # ViT
    for exp in ["cifar10_intra", "cifar10_svhn"]:
        results_list = []
        for s in seeds:
            path = os.path.join(output_base, "xmaha_results", f"vit_{exp}_seed{s}", "xmaha_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results_list.append(json.load(f))
        if results_list:
            summary[f"vit_{exp}"] = _aggregate_method_results(results_list)
    
    # Save
    out_dir = os.path.join(output_base, "xmaha_results", "aggregated")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "xmaha_aggregated.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    logger.info(f"\n{'='*90}")
    logger.info("X-MAHA RESULTS SUMMARY (mean +/- std across seeds)")
    logger.info(f"{'='*90}")
    logger.info(f"{'Experiment':<30} {'Method':<22} {'AUROC':>14} {'FPR@95':>14}")
    logger.info("-" * 82)
    for exp_name, methods in sorted(summary.items()):
        for method, stats in sorted(methods.items()):
            auroc_str = f"{stats['auroc_mean']:.4f}+/-{stats['auroc_std']:.4f}"
            fpr_str = f"{stats['fpr_mean']:.4f}+/-{stats['fpr_std']:.4f}"
            logger.info(f"{exp_name:<30} {method:<22} {auroc_str:>14} {fpr_str:>14}")
        logger.info("-" * 82)
    
    return summary


def _aggregate_method_results(results_list):
    """Aggregate results from multiple seeds."""
    agg = {}
    methods = ["xmaha_uniform", "xmaha_optimized", "mahalanobis_last_layer"]
    for method in methods:
        aurocs = [r[method]["auroc"] for r in results_list if method in r]
        fprs = [r[method]["fpr_at_95"] for r in results_list if method in r]
        if aurocs:
            agg[method] = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std": float(np.std(fprs)),
                "n_seeds": len(aurocs),
            }
    return agg


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="X-Mahalanobis Baseline")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--architecture", choices=["layoutlmv3", "vit", "both"], default="both")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs")
    
    seeds = [args.seed] if args.seed else args.seeds
    all_results = []
    
    if args.run_all or args.architecture in ["layoutlmv3", "both"]:
        exps = ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]
        if args.experiment and args.experiment in exps:
            exps = [args.experiment]
        for exp in exps:
            for s in seeds:
                try:
                    r = run_xmaha_layoutlmv3(exp, s, args.output_dir)
                    if r:
                        all_results.append(r)
                except Exception as e:
                    logger.error(f"Failed: {exp} seed={s}: {e}")
                    import traceback; traceback.print_exc()
    
    if args.run_all or args.architecture in ["vit", "both"]:
        exps = ["cifar10_intra", "cifar10_svhn"]
        if args.experiment and args.experiment in exps:
            exps = [args.experiment]
        for exp in exps:
            for s in seeds:
                try:
                    r = run_xmaha_vit(exp, s, args.output_dir)
                    if r:
                        all_results.append(r)
                except Exception as e:
                    logger.error(f"Failed: vit {exp} seed={s}: {e}")
                    import traceback; traceback.print_exc()
    
    if len(all_results) > 1:
        aggregate_and_compare(args.output_dir, seeds)
    
    logger.info(f"\nDone! {len(all_results)} experiment(s) completed.")


if __name__ == "__main__":
    main()
