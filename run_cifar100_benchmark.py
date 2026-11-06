"""
run_cifar100_benchmark.py — CIFAR-100 Standard OOD Detection Benchmark.

Fine-tunes ViT-B/16 on CIFAR-100 (100 classes ID), evaluates OOD detection
with AHSS + 17 baseline methods against:
  - SVHN (far OOD)
  - DTD/Describable Textures (near OOD)
  - LSUN-C (far OOD)

This is the standard OpenOOD benchmark for cross-dataset OOD detection.

Usage:
    python run_cifar100_benchmark.py --run_all
    python run_cifar100_benchmark.py --seed 42 --ood_source svhn
"""

import argparse
import os
import sys
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.vit_classifier import (
    ViTClassifier,
    extract_vit_embeddings_with_multiple_masks,
    extract_vit_mc_dropout_embeddings,
    extract_vit_embeddings_from_dataloader,
)
from ood_methods.sensitivity_based import AHSSScorer, MCDropoutScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from ood_methods.confidence_based import MSPScorer, MaxLogitScorer, EnergyScorer, KLDivergenceScorer
from ood_methods.feature_based import VIMScorer, ResidualScorer, NECOScorer, ReActScorer, ASHScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.model_io import load_model_weights, save_model_weights, model_weights_exist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Helpers
# ============================================================================

def get_vit_transform():
    """Standard ViT preprocessing: 224×224, ImageNet normalization."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class WrappedDataset(Dataset):
    """Wraps torchvision dataset with indices and transform."""
    def __init__(self, base_dataset, indices, transform, label_offset=0):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
        self.label_offset = label_offset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.base[self.indices[idx]]
        image = self.transform(image)
        return {"pixel_values": image, "labels": label + self.label_offset}


class OODWrappedDataset(Dataset):
    """Wraps OOD dataset with dummy label."""
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.base[self.indices[idx]]
        image = self.transform(image)
        return {"pixel_values": image, "labels": -1}


def load_cifar100_data(seed, data_dir=None, num_classes=100):
    """Load CIFAR-100: `num_classes` classes as ID, 90/10 train/eval split.

    num_classes=100 (default) reproduces the original behaviour exactly. Smaller
    values keep the first `num_classes` label ids, which supports the class-count
    ablation. Labels are already 0..num_classes-1 in
    that case, so no remapping is needed.
    """
    from torchvision.datasets import CIFAR100

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data", "cifar100")

    transform = get_vit_transform()

    train_raw = CIFAR100(root=data_dir, train=True, download=True)
    test_raw = CIFAR100(root=data_dir, train=False, download=True)

    if num_classes >= 100:
        all_indices = list(range(len(train_raw)))
    else:
        keep = set(range(num_classes))
        all_indices = [i for i, t in enumerate(train_raw.targets) if t in keep]
        if not all_indices:
            raise RuntimeError(f"class subset num_classes={num_classes} selected 0 training images")

    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    train_indices = all_indices[:split]
    eval_indices = all_indices[split:]

    train_ds = WrappedDataset(train_raw, train_indices, transform)
    eval_ds = WrappedDataset(train_raw, eval_indices, transform)

    logger.info(f"CIFAR-100 ({num_classes} classes): train={len(train_ds)}, eval={len(eval_ds)}")
    return train_ds, eval_ds, test_raw, transform


def load_ood_dataset(ood_source, transform, data_dir=None, max_samples=10000):
    """Load an OOD test dataset."""
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data")
    
    if ood_source == "svhn":
        from torchvision.datasets import SVHN
        ood_raw = SVHN(root=os.path.join(data_dir, "svhn"), split="test", download=True)
        indices = list(range(min(max_samples, len(ood_raw))))
    elif ood_source == "dtd":
        from torchvision.datasets import DTD
        ood_raw = DTD(root=os.path.join(data_dir, "dtd"), split="test", download=True)
        indices = list(range(len(ood_raw)))
    elif ood_source == "lsun_c":
        # Use LSUN-C (cropped) from OpenOOD or torchvision
        try:
            from torchvision.datasets import LSUN
            ood_raw = LSUN(root=os.path.join(data_dir, "lsun"), classes="test")
            indices = list(range(min(max_samples, len(ood_raw))))
        except Exception:
            logger.warning("LSUN not available, falling back to SVHN")
            return load_ood_dataset("svhn", transform, data_dir, max_samples)
    else:
        raise ValueError(f"Unknown OOD source: {ood_source}")
    
    ood_ds = OODWrappedDataset(ood_raw, indices, transform)
    logger.info(f"OOD ({ood_source}): {len(ood_ds)} samples")
    return ood_ds


# ============================================================================
# Fine-tuning
# ============================================================================

def finetune_cifar100(train_ds, eval_ds, seed, output_dir, epochs=10, lr=2e-5, batch_size=64, num_labels=100):
    """Fine-tune ViT-B/16 on CIFAR-100 (num_labels=100 reproduces original behaviour)."""
    checkpoint_path = os.path.join(output_dir, "best_model")
    
    if model_weights_exist(checkpoint_path, "model"):
        logger.info(f"Loading existing checkpoint from {checkpoint_path}")
        model = ViTClassifier(num_labels=num_labels)
        state = load_model_weights(checkpoint_path, "model")
        model.load_state_dict(state)
        return model
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ViTClassifier(num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
        scheduler.step()
        train_acc = train_correct / train_total
        
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in eval_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model.model(pixel_values=pixel_values)
                preds = outputs.logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += len(labels)
        
        eval_acc = eval_correct / eval_total
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/train_total:.4f}, train_acc={train_acc:.4f}, eval_acc={eval_acc:.4f}")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best eval_acc={best_acc:.4f}")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    save_model_weights(best_state, os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(best_state)
    logger.info(f"Saved best model (eval_acc={best_acc:.4f}) to {checkpoint_path}")
    
    return model


# ============================================================================
# Mask Generation
# ============================================================================

def generate_random_masks(num_masks, num_layers=12, num_heads=12, masking_pcts=(0.1, 0.2), seed=42):
    """Generate random binary attention head masks."""
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
# Score extraction helpers
# ============================================================================

def extract_logits(model, dataloader, device="cuda"):
    """Extract logits, last-layer embeddings, and labels."""
    model.eval()
    model.to(device)
    
    all_logits = []
    all_embs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pixel_values)
            
            all_logits.append(outputs["logits"].cpu().numpy())
            cls_emb = outputs["hidden_states"][-1][:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    logits = np.concatenate(all_logits)
    embs = np.concatenate(all_embs)
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return logits, embs, labels


def get_classifier_weights(model):
    """Get classifier head weights and bias."""
    state = model.state_dict()
    for key in ["model.classifier.weight", "classifier.weight"]:
        if key in state:
            W = state[key].cpu().numpy()
            b_key = key.replace("weight", "bias")
            b = state[b_key].cpu().numpy() if b_key in state else np.zeros(W.shape[0])
            return W, b
    return None, None


def rank_normalize(scores):
    """Rank-based normalization to [0,1]."""
    ranks = rankdata(scores, method='average')
    return (ranks - 1) / max(1, len(ranks) - 1)


# ============================================================================
# Multi-layer extraction for X-Maha
# ============================================================================

def extract_all_layer_cls(model, dataloader, device="cuda"):
    """Extract CLS from all 12 transformer layers. Returns (12, N, 768)."""
    model.eval()
    model.to(device)
    
    layer_embs = [[] for _ in range(13)]
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting all layers"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pixel_values)
            hidden_states = outputs["hidden_states"]
            
            for l in range(len(hidden_states)):
                layer_embs[l].append(hidden_states[l][:, 0, :].cpu().numpy())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    result = np.stack([np.concatenate(layer_embs[l]) for l in range(1, 13)])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return result, labels


# ============================================================================
# X-Maha scorer (reuse from run_xmaha_baseline)
# ============================================================================

class XMahalanobisScorer:
    """X-Maha: multi-layer fusion + Mahalanobis."""
    
    def __init__(self):
        self.weights = None
        self.class_means = {}
        self.precision_matrix = None
    
    def _fuse(self, layer_embs, weights):
        return np.tensordot(weights, layer_embs, axes=([0], [0]))
    
    def _fit_mahalanobis(self, fused, labels):
        from sklearn.covariance import ShrunkCovariance
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
        N = fused.shape[0]
        distances = np.zeros((N, len(self.class_means)))
        for idx, (c, mean) in enumerate(self.class_means.items()):
            diff = fused - mean
            distances[:, idx] = np.sum(diff @ self.precision_matrix * diff, axis=1)
        return -distances.min(axis=1)
    
    def fit(self, train_layers, train_labels, eval_layers=None, ood_layers=None, optimize=True, max_train_for_opt=5000):
        L = train_layers.shape[0]
        if optimize and eval_layers is not None and ood_layers is not None:
            from scipy.optimize import minimize
            # Subsample training data for optimization speed
            N_train = train_layers.shape[1]
            if N_train > max_train_for_opt:
                rng = np.random.RandomState(42)
                idx = rng.choice(N_train, max_train_for_opt, replace=False)
                opt_train = train_layers[:, idx, :]
                opt_labels = train_labels[idx]
                logger.info(f"  Subsampled {N_train} -> {max_train_for_opt} for optimization")
            else:
                opt_train = train_layers
                opt_labels = train_labels
            def objective(raw_w):
                w = np.exp(raw_w) / np.exp(raw_w).sum()
                self._fit_mahalanobis(self._fuse(opt_train, w), opt_labels)
                auroc = compute_auroc(self._score(self._fuse(eval_layers, w)),
                                     self._score(self._fuse(ood_layers, w)))
                return -auroc
            result = minimize(objective, np.zeros(L), method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5})
            self.weights = np.exp(result.x) / np.exp(result.x).sum()
        else:
            self.weights = np.ones(L) / L
        # Final fit on ALL training data
        self._fit_mahalanobis(self._fuse(train_layers, self.weights), train_labels)
    
    def score(self, layer_embs):
        return self._score(self._fuse(layer_embs, self.weights))


# ============================================================================
# Full OOD Evaluation Pipeline
# ============================================================================

def run_cifar100_experiment(seed, ood_source, base_output_dir, num_classes=100):
    """Run complete CIFAR-100 OOD detection experiment."""
    
    tag = "" if num_classes >= 100 else f"_c{num_classes}"
    exp_name = f"cifar100{tag}_{ood_source}_seed{seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_dir = os.path.join(base_output_dir, "cifar100_checkpoints", f"cifar100{tag}_seed{seed}")
    result_dir = os.path.join(base_output_dir, "cifar100_results", exp_name)
    result_path = os.path.join(result_dir, "full_results.json")
    
    if os.path.exists(result_path):
        logger.info(f"Skipping (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CIFAR-100 BENCHMARK: {exp_name}")
    logger.info(f"{'='*70}")
    start = time.time()
    
    # Step 1: Load CIFAR-100 data
    train_ds, eval_ds, test_raw, transform = load_cifar100_data(seed, num_classes=num_classes)
    
    # Step 2: Load OOD data
    ood_ds = load_ood_dataset(ood_source, transform)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Step 3: Fine-tune or load model
    logger.info("Step 1: Fine-tuning ViT on CIFAR-100...")
    model = finetune_cifar100(train_ds, eval_ds, seed, checkpoint_dir, epochs=10, lr=2e-5, batch_size=64, num_labels=num_classes)
    model.to(device)
    model.eval()
    
    results = {"experiment": f"cifar100_{ood_source}", "seed": seed, "architecture": "ViT-B/16"}
    
    # Step 4: Extract logits + embeddings
    logger.info("Step 2: Extracting logits & embeddings...")
    train_logits, train_embs, train_labels = extract_logits(model, train_loader, device)
    eval_logits, eval_embs, eval_labels = extract_logits(model, eval_loader, device)
    ood_logits, ood_embs, _ = extract_logits(model, ood_loader, device)
    
    W, b = get_classifier_weights(model)
    
    # Step 5: Generate masks & extract multi-mask embeddings
    logger.info("Step 3: Extracting multi-mask embeddings for AHSS...")
    masks = generate_random_masks(25, seed=seed)
    
    eval_unmasked, eval_masked_list, _ = extract_vit_embeddings_with_multiple_masks(
        model, eval_loader, masks, device=device, desc="eval masks"
    )
    ood_unmasked, ood_masked_list, _ = extract_vit_embeddings_with_multiple_masks(
        model, ood_loader, masks, device=device, desc="OOD masks"
    )
    
    # Step 6: Extract MC-Dropout embeddings
    logger.info("Step 4: MC-Dropout embeddings...")
    eval_mc_list, _ = extract_vit_mc_dropout_embeddings(model, eval_loader, n_forward=25, device=device)
    ood_mc_list, _ = extract_vit_mc_dropout_embeddings(model, ood_loader, n_forward=25, device=device)
    
    # Step 7: Extract all-layer embeddings for X-Maha
    logger.info("Step 5: All-layer embeddings for X-Maha...")
    train_layers, _ = extract_all_layer_cls(model, train_loader, device)
    eval_layers, _ = extract_all_layer_cls(model, eval_loader, device)
    ood_layers, _ = extract_all_layer_cls(model, ood_loader, device)
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    
    # ========================================
    # Score computation
    # ========================================
    
    def evaluate(eval_scores, ood_scores, method_name):
        auroc = compute_auroc(eval_scores, ood_scores)
        fpr = compute_fpr_at_tpr(eval_scores, ood_scores)
        results[method_name] = {"auroc": float(auroc), "fpr_at_95": float(fpr)}
        logger.info(f"  {method_name:<25} AUROC={auroc:.4f}  FPR@95={fpr:.4f}")
    
    logger.info("\n--- OOD Detection Scores ---")
    
    # 1. AHSS
    ahss = AHSSScorer()
    eval_ahss = -ahss.compute_ahss(eval_unmasked, eval_masked_list, normalize=True)
    ood_ahss = -ahss.compute_ahss(ood_unmasked, ood_masked_list, normalize=True)
    evaluate(eval_ahss, ood_ahss, "AHSS")
    
    # 2. MC-Dropout
    mc = MCDropoutScorer()
    eval_mc = -mc.compute_mc_uncertainty(eval_mc_list, normalize=True)
    ood_mc = -mc.compute_mc_uncertainty(ood_mc_list, normalize=True)
    evaluate(eval_mc, ood_mc, "MC-Dropout")
    
    # 3. Mahalanobis
    maha = MahalanobisScorer()
    maha.fit(train_embs, train_labels)
    evaluate(maha.score(eval_embs), maha.score(ood_embs), "Mahalanobis")
    
    # 4. kNN
    knn = KNNScorer(k=10)
    knn.fit(train_embs)
    evaluate(knn.score(eval_embs), knn.score(ood_embs), "kNN")
    
    # 5. MSP
    msp = MSPScorer()
    evaluate(msp.score(eval_logits), msp.score(ood_logits), "MSP")
    
    # 6. MaxLogit
    ml = MaxLogitScorer()
    evaluate(ml.score(eval_logits), ml.score(ood_logits), "MaxLogit")
    
    # 7. Energy
    energy = EnergyScorer()
    evaluate(energy.score(eval_logits), energy.score(ood_logits), "Energy")
    
    # 8. KL Divergence
    kl = KLDivergenceScorer()
    evaluate(kl.score(eval_logits), kl.score(ood_logits), "KL")
    
    # 9. VIM
    try:
        vim = VIMScorer()
        vim.fit(train_embs, W, b)
        evaluate(vim.score(eval_embs), vim.score(ood_embs), "VIM")
    except Exception as e:
        logger.warning(f"VIM failed: {e}")
    
    # 10. Residual
    try:
        res = ResidualScorer()
        res.fit(train_embs)
        evaluate(res.score(eval_embs), res.score(ood_embs), "Residual")
    except Exception as e:
        logger.warning(f"Residual failed: {e}")
    
    # 11. NECO
    try:
        neco = NECOScorer()
        neco.fit(train_embs, train_labels)
        evaluate(neco.score(eval_embs), neco.score(ood_embs), "NECO")
    except Exception as e:
        logger.warning(f"NECO failed: {e}")
    
    # 12. ReAct
    try:
        react = ReActScorer()
        react.fit(train_embs)
        evaluate(react.score(eval_embs, eval_logits), react.score(ood_embs, ood_logits), "ReAct")
    except Exception as e:
        logger.warning(f"ReAct failed: {e}")
    
    # 13. ASH
    try:
        ash = ASHScorer()
        evaluate(ash.score(eval_embs, eval_logits), ash.score(ood_embs, ood_logits), "ASH")
    except Exception as e:
        logger.warning(f"ASH failed: {e}")
    
    # 14. X-Maha (uniform)
    xmu = XMahalanobisScorer()
    xmu.fit(train_layers, train_labels, optimize=False)
    evaluate(xmu.score(eval_layers), xmu.score(ood_layers), "X-Maha_uniform")
    
    # 15. X-Maha (optimized)
    xmo = XMahalanobisScorer()
    xmo.fit(train_layers, train_labels, eval_layers=eval_layers, ood_layers=ood_layers, optimize=True)
    evaluate(xmo.score(eval_layers), xmo.score(ood_layers), "X-Maha_optimized")
    results["X-Maha_optimized"]["weights"] = xmo.weights.tolist()
    
    # 16. mahAHM (Mahalanobis on masked embeddings average)
    logger.info("Computing AHM variants...")
    maha_ahm = MahalanobisScorer()
    # Average masked embeddings
    eval_avg_masked = np.mean(eval_masked_list, axis=0)
    ood_avg_masked = np.mean(ood_masked_list, axis=0)
    maha_ahm.fit(train_embs, train_labels)
    evaluate(maha_ahm.score(eval_avg_masked), maha_ahm.score(ood_avg_masked), "mahAHM")
    
    # 17. knnAHM
    knn_ahm = KNNScorer(k=10)
    knn_ahm.fit(train_embs)
    evaluate(knn_ahm.score(eval_avg_masked), knn_ahm.score(ood_avg_masked), "knnAHM")
    
    # 18. Combined (rank normalization + alpha sweep)
    logger.info("Computing Combined scoring...")
    best_comb_auroc = 0
    best_alpha = 0
    eval_maha_scores = maha.score(eval_embs)
    ood_maha_scores = maha.score(ood_embs)
    
    all_ahss = np.concatenate([eval_ahss, ood_ahss])
    all_maha = np.concatenate([eval_maha_scores, ood_maha_scores])
    all_ahss_rn = rank_normalize(all_ahss)
    all_maha_rn = rank_normalize(all_maha)
    n_eval = len(eval_ahss)
    
    for alpha_10 in range(11):
        alpha = alpha_10 / 10.0
        all_comb = alpha * all_ahss_rn + (1 - alpha) * all_maha_rn
        comb_auroc = compute_auroc(all_comb[:n_eval], all_comb[n_eval:])
        
        if comb_auroc > best_comb_auroc:
            best_comb_auroc = comb_auroc
            best_alpha = alpha
    
    # Recompute best
    all_ahss = np.concatenate([eval_ahss, ood_ahss])
    all_maha = np.concatenate([eval_maha_scores, ood_maha_scores])
    all_comb = best_alpha * rank_normalize(all_ahss) + (1 - best_alpha) * rank_normalize(all_maha)
    n_eval = len(eval_ahss)
    comb_auroc = compute_auroc(all_comb[:n_eval], all_comb[n_eval:])
    comb_fpr = compute_fpr_at_tpr(all_comb[:n_eval], all_comb[n_eval:])
    results["Combined"] = {"auroc": float(comb_auroc), "fpr_at_95": float(comb_fpr), "best_alpha": best_alpha}
    logger.info(f"  {'Combined':<25} AUROC={comb_auroc:.4f}  FPR@95={comb_fpr:.4f}  alpha={best_alpha}")
    
    results["elapsed_seconds"] = time.time() - start
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["n_train"] = len(train_ds)
    results["n_eval"] = len(eval_ds)
    results["n_ood"] = len(ood_ds)
    
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {result_path}")
    
    # Save layer embeddings
    layer_dir = os.path.join(result_dir, "layer_embeddings")
    os.makedirs(layer_dir, exist_ok=True)
    np.save(os.path.join(layer_dir, "train_layer_embeddings.npy"), train_layers)
    np.save(os.path.join(layer_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(layer_dir, "eval_layer_embeddings.npy"), eval_layers)
    np.save(os.path.join(layer_dir, "ood_layer_embeddings.npy"), ood_layers)
    
    return results


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_cifar100_results(base_output_dir, seeds, ood_sources):
    """Aggregate CIFAR-100 results across seeds."""
    
    summary = {}
    
    for ood_src in ood_sources:
        results_list = []
        for s in seeds:
            path = os.path.join(base_output_dir, "cifar100_results",
                              f"cifar100_{ood_src}_seed{s}", "full_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results_list.append(json.load(f))
        
        if not results_list:
            continue
        
        agg = {}
        # Get all method names from first result
        methods = [k for k in results_list[0] if isinstance(results_list[0][k], dict) and "auroc" in results_list[0][k]]
        
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
        
        summary[f"cifar100_{ood_src}"] = agg
    
    # Save
    agg_dir = os.path.join(base_output_dir, "cifar100_results", "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    with open(os.path.join(agg_dir, "cifar100_aggregated.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print table
    logger.info(f"\n{'='*90}")
    logger.info("CIFAR-100 OOD DETECTION BENCHMARK (mean +/- std across seeds)")
    logger.info(f"{'='*90}")
    
    for exp_name, methods in sorted(summary.items()):
        logger.info(f"\n--- {exp_name} ---")
        logger.info(f"{'Method':<25} {'AUROC':>14} {'FPR@95':>14}")
        logger.info("-" * 55)
        for method, stats in sorted(methods.items(), key=lambda x: -x[1]["auroc_mean"]):
            auroc_str = f"{stats['auroc_mean']:.4f}+/-{stats['auroc_std']:.4f}"
            fpr_str = f"{stats['fpr_mean']:.4f}+/-{stats['fpr_std']:.4f}"
            logger.info(f"{method:<25} {auroc_str:>14} {fpr_str:>14}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 OOD Benchmark")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--ood_source", type=str, default=None, choices=["svhn", "dtd"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--num_classes", type=int, default=100,
                        help="ID class subset size for the class-count ablation (100 = original)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs")
    
    seeds = [args.seed] if args.seed else args.seeds
    ood_sources = [args.ood_source] if args.ood_source else ["svhn", "dtd"]
    
    all_results = []
    
    for ood_src in ood_sources:
        for s in seeds:
            try:
                r = run_cifar100_experiment(s, ood_src, args.output_dir, num_classes=args.num_classes)
                if r:
                    all_results.append(r)
            except Exception as e:
                logger.error(f"Failed: cifar100_{ood_src} seed={s}: {e}")
                import traceback; traceback.print_exc()
    
    if len(all_results) > 1:
        aggregate_cifar100_results(args.output_dir, seeds, ood_sources)
    
    logger.info(f"\nDone! {len(all_results)} experiment(s) completed.")


if __name__ == "__main__":
    main()
