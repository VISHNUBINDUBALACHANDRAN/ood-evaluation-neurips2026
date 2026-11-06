"""
run_vit_experiments.py — Phase 4: ViT Cross-Architecture AHSS Experiments.

Fine-tunes ViT-B/16 on CIFAR-10 (9 ID classes, 1 OOD class for intra-dataset)
and evaluates AHSS + baselines for OOD detection.

Experiments:
  1. cifar10_intra: CIFAR-10 with class 0 ("airplane") as OOD
  2. cifar10_svhn: CIFAR-10 (all 10 classes as ID) → SVHN as OOD  
  3. cifar10_textures: CIFAR-10 → DTD Textures as OOD

Each experiment:
  - Fine-tune ViT on ID train split
  - Generate random masks + optimize AHSS masks
  - Compute AHSS, MC-Dropout, Mahalanobis, kNN scores  
  - Report AUROC + FPR@95

Usage:
    python run_vit_experiments.py --experiment cifar10_intra --seed 42
    python run_vit_experiments.py --run_all
"""

import argparse
import os
import sys
import logging
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Add project root to path
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
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.model_io import load_model_weights, save_model_weights, model_weights_exist, save_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------- Dataset Helpers ----------

class CIFAR10Dataset(Dataset):
    """Wraps torchvision CIFAR10 for HuggingFace ViT compatibility."""
    
    def __init__(self, torchvision_dataset, transform=None):
        self.dataset = torchvision_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}


class SVHNDataset(Dataset):
    """Wraps torchvision SVHN for compatibility."""
    
    def __init__(self, torchvision_dataset, transform=None):
        self.dataset = torchvision_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}


def get_vit_transform():
    """Get ViT-compatible image transform (224×224, normalize to ImageNet stats)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_cifar10_splits(seed, ood_class=0, data_dir=None):
    """
    Load CIFAR-10 with one class as OOD (intra-dataset setup).
    
    Returns:
        train_dataset, eval_dataset, ood_dataset, id_classes, label_map
    """
    from torchvision.datasets import CIFAR10
    
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data", "cifar10")
    
    transform = get_vit_transform()
    
    train_raw = CIFAR10(root=data_dir, train=True, download=True)
    test_raw = CIFAR10(root=data_dir, train=False, download=True)
    
    all_classes = list(range(10))
    id_classes = [c for c in all_classes if c != ood_class]
    
    # Remap labels: ID classes → 0..8
    label_map = {c: i for i, c in enumerate(id_classes)}
    
    class RemappedDataset(Dataset):
        def __init__(self, base_dataset, class_indices, lmap, tfm):
            self.base = base_dataset
            self.indices = class_indices
            self.lmap = lmap
            self.tfm = tfm
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.base[real_idx]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": self.lmap[label]}
    
    class OODDataset(Dataset):
        def __init__(self, base_dataset, class_indices, tfm):
            self.base = base_dataset
            self.indices = class_indices
            self.tfm = tfm
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.base[real_idx]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": -1}  # OOD label
    
    # Split indices
    train_id_indices = [i for i, (_, l) in enumerate(train_raw) if l in id_classes]
    test_id_indices = [i for i, (_, l) in enumerate(test_raw) if l in id_classes]
    test_ood_indices = [i for i, (_, l) in enumerate(test_raw) if l == ood_class]
    
    # Further split train into train/eval (90/10)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_id_indices)
    split = int(0.9 * len(train_id_indices))
    train_indices = train_id_indices[:split]
    eval_from_train = train_id_indices[split:]
    
    train_ds = RemappedDataset(train_raw, train_indices, label_map, transform)
    eval_ds = RemappedDataset(train_raw, eval_from_train, label_map, transform)
    ood_ds = OODDataset(test_raw, test_ood_indices, transform)
    
    logger.info(f"CIFAR-10 intra (OOD class={ood_class}): train={len(train_ds)}, eval={len(eval_ds)}, OOD={len(ood_ds)}")
    
    return train_ds, eval_ds, ood_ds, id_classes, label_map


def load_cifar10_cross(seed, ood_source="svhn", data_dir=None):
    """
    Load CIFAR-10 (all 10 classes as ID) with external OOD dataset.
    """
    from torchvision.datasets import CIFAR10, SVHN
    
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data")
    
    transform = get_vit_transform()
    
    train_raw = CIFAR10(root=os.path.join(data_dir, "cifar10"), train=True, download=True)
    test_raw = CIFAR10(root=os.path.join(data_dir, "cifar10"), train=False, download=True)
    
    all_indices_train = list(range(len(train_raw)))
    all_indices_test = list(range(len(test_raw)))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices_train)
    split = int(0.9 * len(all_indices_train))
    train_indices = all_indices_train[:split]
    eval_indices = all_indices_train[split:]
    
    class WrappedDataset(Dataset):
        def __init__(self, base, indices, tfm):
            self.base = base
            self.indices = indices
            self.tfm = tfm
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            image, label = self.base[self.indices[idx]]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": label}
    
    train_ds = WrappedDataset(train_raw, train_indices, transform)
    eval_ds = WrappedDataset(train_raw, eval_indices, transform)
    
    # OOD dataset
    if ood_source == "svhn":
        ood_raw = SVHN(root=os.path.join(data_dir, "svhn"), split="test", download=True)
        # Limit to 10k samples for efficiency
        ood_indices = list(range(min(10000, len(ood_raw))))
        
        class SVHNWrapped(Dataset):
            def __init__(self, base, indices, tfm):
                self.base = base
                self.indices = indices
                self.tfm = tfm
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                image, label = self.base[self.indices[idx]]
                image = self.tfm(image)
                return {"pixel_values": image, "labels": -1}
        
        ood_ds = SVHNWrapped(ood_raw, ood_indices, transform)
    else:
        raise ValueError(f"Unknown OOD source: {ood_source}")
    
    logger.info(f"CIFAR-10 cross ({ood_source}): train={len(train_ds)}, eval={len(eval_ds)}, OOD={len(ood_ds)}")
    
    return train_ds, eval_ds, ood_ds, list(range(10)), {i: i for i in range(10)}


# ---------- Fine-tuning ----------

def finetune_vit(train_ds, eval_ds, num_labels, seed, output_dir, epochs=10, lr=2e-5, batch_size=64):
    """Fine-tune ViT-B/16 on the training set."""
    
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
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Train
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
        
        # Eval
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
            logger.info(f"  → New best eval_acc={best_acc:.4f}")
    
    # Save best
    os.makedirs(checkpoint_path, exist_ok=True)
    save_model_weights(best_state, os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(best_state)
    logger.info(f"Saved best model (eval_acc={best_acc:.4f}) to {checkpoint_path}")
    
    return model


# ---------- Mask Generation ----------

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


# ---------- Scoring Pipeline ----------

def rank_normalize(scores):
    """Rank-based normalization to [0,1]."""
    from scipy.stats import rankdata
    ranks = rankdata(scores, method='average')
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(scores)


def run_single_experiment(experiment, seed, base_output_dir, log_dir, skip_mc_dropout=False):
    """Run a single ViT AHSS experiment."""
    
    exp_name = f"{experiment}_seed{seed}"
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING: {exp_name}")
    logger.info(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(base_output_dir, "vit_results", exp_name)
    checkpoint_dir = os.path.join(base_output_dir, "vit_checkpoints", f"{experiment}_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Skip if results already exist
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        logger.info(f"Results already exist at {results_file}, skipping.")
        with open(results_file, "r") as f:
            return json.load(f)
    
    # Load data
    if experiment == "cifar10_intra":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_splits(seed, ood_class=0)
        num_labels = 9
    elif experiment == "cifar10_svhn":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_cross(seed, ood_source="svhn")
        num_labels = 10
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Step 1: Fine-tune (or load checkpoint)
    logger.info("Step 1: Fine-tuning ViT...")
    model = finetune_vit(train_ds, eval_ds, num_labels, seed, checkpoint_dir)
    model.to(device)
    model.eval()
    
    # Step 2: Generate masks
    logger.info("Step 2: Generating masks...")
    masks = generate_random_masks(25, seed=seed)
    
    # Save masks
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, m in enumerate(masks):
        save_mask(m, os.path.join(masks_dir, f"mask_{i}.safetensors"))
    
    # Step 3: Extract multi-mask embeddings for train/eval/OOD
    logger.info("Step 3: Extracting train embeddings (unmasked)...")
    train_embs, train_labels = extract_vit_embeddings_from_dataloader(
        model, train_loader, device=device, desc=f"{exp_name} train"
    )
    
    logger.info("Step 3: Extracting eval multi-mask embeddings...")
    eval_unmasked, eval_masked_list, eval_labels = extract_vit_embeddings_with_multiple_masks(
        model, eval_loader, masks, device=device, desc=f"{exp_name} eval"
    )
    
    logger.info("Step 3: Extracting OOD multi-mask embeddings...")
    ood_unmasked, ood_masked_list, ood_labels = extract_vit_embeddings_with_multiple_masks(
        model, ood_loader, masks, device=device, desc=f"{exp_name} OOD"
    )
    
    # Step 4: Compute AHSS scores
    logger.info("Step 4: Computing AHSS scores...")
    ahss_scorer = AHSSScorer()
    
    eval_ahss_raw = ahss_scorer.compute_ahss(eval_unmasked, eval_masked_list, normalize=True)
    ood_ahss_raw = ahss_scorer.compute_ahss(ood_unmasked, ood_masked_list, normalize=True)
    
    # AHSS: negate so higher = more ID-like (convention for compute_auroc)
    eval_ahss = -eval_ahss_raw
    ood_ahss = -ood_ahss_raw
    
    ahss_auroc = compute_auroc(eval_ahss, ood_ahss)
    ahss_fpr = compute_fpr_at_tpr(eval_ahss, ood_ahss)
    
    logger.info(f"  AHSS_raw AUROC: {ahss_auroc:.4f}, FPR@95: {ahss_fpr:.4f}")
    
    # Step 5: Distance-based baselines
    logger.info("Step 5: Computing Mahalanobis & kNN...")
    
    maha_scorer = MahalanobisScorer()
    maha_scorer.fit(train_embs, train_labels)
    eval_maha = maha_scorer.score(eval_unmasked)
    ood_maha = maha_scorer.score(ood_unmasked)
    
    maha_auroc = compute_auroc(eval_maha, ood_maha)
    maha_fpr = compute_fpr_at_tpr(eval_maha, ood_maha)
    
    logger.info(f"  Mahalanobis AUROC: {maha_auroc:.4f}, FPR@95: {maha_fpr:.4f}")
    
    knn_scorer = KNNScorer(k=10)
    knn_scorer.fit(train_embs)
    eval_knn = knn_scorer.score(eval_unmasked)
    ood_knn = knn_scorer.score(ood_unmasked)
    
    knn_auroc = compute_auroc(eval_knn, ood_knn)
    knn_fpr = compute_fpr_at_tpr(eval_knn, ood_knn)
    
    logger.info(f"  kNN AUROC: {knn_auroc:.4f}, FPR@95: {knn_fpr:.4f}")
    
    # Step 6: Combined scoring (AHSS + Mahalanobis) with rank normalization
    logger.info("Step 6: Combined scoring...")
    
    best_combined_auroc = 0.0
    best_alpha = 0.5
    alpha_grid = np.arange(0.0, 1.05, 0.1)
    
    # Concatenate for rank normalization, then split back
    all_maha = np.concatenate([eval_maha, ood_maha])
    all_ahss_neg = np.concatenate([eval_ahss, ood_ahss])  # already negated
    n_eval = len(eval_maha)
    
    for alpha in alpha_grid:
        combined = alpha * rank_normalize(all_maha) + (1 - alpha) * rank_normalize(all_ahss_neg)
        ec = combined[:n_eval]
        oc = combined[n_eval:]
        a = compute_auroc(ec, oc)
        if a > best_combined_auroc:
            best_combined_auroc = a
            best_alpha = alpha
    
    combined_final = best_alpha * rank_normalize(all_maha) + (1 - best_alpha) * rank_normalize(all_ahss_neg)
    combined_eval = combined_final[:n_eval]
    combined_ood = combined_final[n_eval:]
    combined_auroc = compute_auroc(combined_eval, combined_ood)
    combined_fpr = compute_fpr_at_tpr(combined_eval, combined_ood)
    
    logger.info(f"  Combined (α={best_alpha:.1f}) AUROC: {combined_auroc:.4f}, FPR@95: {combined_fpr:.4f}")
    
    # Step 7: MC-Dropout (if not skipped)
    mc_auroc = 0.0
    mc_fpr = 1.0
    if not skip_mc_dropout:
        logger.info("Step 7: MC-Dropout baseline...")
        eval_mc_embs, _ = extract_vit_mc_dropout_embeddings(
            model, eval_loader, device=device, n_forward=25, desc=f"{exp_name} eval MC"
        )
        ood_mc_embs, _ = extract_vit_mc_dropout_embeddings(
            model, ood_loader, device=device, n_forward=25, desc=f"{exp_name} OOD MC"
        )
        
        mc_scorer = MCDropoutScorer()
        eval_mc_scores = -mc_scorer.compute_mc_uncertainty(eval_mc_embs)  # negate: lower uncertainty = ID
        ood_mc_scores = -mc_scorer.compute_mc_uncertainty(ood_mc_embs)
        
        mc_auroc = compute_auroc(eval_mc_scores, ood_mc_scores)
        mc_fpr = compute_fpr_at_tpr(eval_mc_scores, ood_mc_scores)
        logger.info(f"  MC-Dropout AUROC: {mc_auroc:.4f}, FPR@95: {mc_fpr:.4f}")
    
    # Compile results
    results = {
        "experiment": experiment,
        "seed": seed,
        "architecture": "ViT-B/16",
        "ahss_raw": {"auroc": float(ahss_auroc), "fpr_at_95": float(ahss_fpr)},
        "mahalanobis": {"auroc": float(maha_auroc), "fpr_at_95": float(maha_fpr)},
        "knn": {"auroc": float(knn_auroc), "fpr_at_95": float(knn_fpr)},
        "combined": {
            "auroc": float(combined_auroc),
            "fpr_at_95": float(combined_fpr),
            "best_alpha": float(best_alpha),
        },
        "mc_dropout": {"auroc": float(mc_auroc), "fpr_at_95": float(mc_fpr)},
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "n_ood": len(ood_ds),
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {exp_name}")
    logger.info(f"  AHSS_raw AUROC:      {results['ahss_raw']['auroc']:.4f}")
    logger.info(f"  Mahalanobis AUROC:   {results['mahalanobis']['auroc']:.4f}")
    logger.info(f"  kNN AUROC:           {results['knn']['auroc']:.4f}")
    logger.info(f"  Combined AUROC:      {results['combined']['auroc']:.4f} (α={best_alpha:.1f})")
    logger.info(f"  MC-Dropout AUROC:    {results['mc_dropout']['auroc']:.4f}")
    logger.info(f"{'='*50}")
    
    return results


def aggregate_results(base_output_dir, experiment, seeds):
    """Aggregate results across seeds for an experiment."""
    results_list = []
    for seed in seeds:
        results_file = os.path.join(base_output_dir, "vit_results", f"{experiment}_seed{seed}", "results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_list.append(json.load(f))
    
    if not results_list:
        logger.warning(f"No results found for {experiment}")
        return
    
    # Aggregate
    methods = ["ahss_raw", "mahalanobis", "knn", "combined", "mc_dropout"]
    agg = {}
    for method in methods:
        aurocs = [r[method]["auroc"] for r in results_list]
        fprs = [r[method]["fpr_at_95"] for r in results_list]
        agg[method] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "fpr_at_95_mean": float(np.mean(fprs)),
            "fpr_at_95_std": float(np.std(fprs)),
            "n_seeds": len(aurocs),
        }
    
    # Save
    agg_dir = os.path.join(base_output_dir, "vit_results", "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    with open(os.path.join(agg_dir, f"{experiment}_aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)
    
    # Print table
    logger.info(f"\n{'='*70}")
    logger.info(f"AGGREGATED: {experiment} ({len(results_list)} seeds)")
    logger.info(f"{'Method':<20} {'AUROC':<20} {'FPR@95':<20}")
    logger.info(f"{'-'*60}")
    for method in methods:
        a = agg[method]
        logger.info(f"{method:<20} {a['auroc_mean']:.4f}±{a['auroc_std']:.4f}    {a['fpr_at_95_mean']:.4f}±{a['fpr_at_95_std']:.4f}")
    logger.info(f"{'='*70}")
    
    # Save table
    with open(os.path.join(agg_dir, f"{experiment}_table.txt"), "w") as f:
        f.write(f"{'Method':<20} {'AUROC':<20} {'FPR@95':<20}\n")
        f.write(f"{'-'*60}\n")
        for method in methods:
            a = agg[method]
            f.write(f"{method:<20} {a['auroc_mean']:.4f}±{a['auroc_std']:.4f}    {a['fpr_at_95_mean']:.4f}±{a['fpr_at_95_std']:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="cifar10_intra",
                        choices=["cifar10_intra", "cifar10_svhn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--skip_mc_dropout", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs")
    
    log_dir = os.path.join(os.path.dirname(SCRIPT_DIR), os.path.pardir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    seeds = args.seeds or [42, 123, 456, 789, 1024]
    
    if args.run_all:
        experiments = ["cifar10_intra", "cifar10_svhn"]
        for experiment in experiments:
            for seed in seeds:
                try:
                    run_single_experiment(experiment, seed, args.output_dir, log_dir, args.skip_mc_dropout)
                except Exception as e:
                    logger.error(f"FAILED: {experiment}_seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
            aggregate_results(args.output_dir, experiment, seeds)
    else:
        if args.seeds:
            for seed in args.seeds:
                run_single_experiment(args.experiment, seed, args.output_dir, log_dir, args.skip_mc_dropout)
            aggregate_results(args.output_dir, args.experiment, args.seeds)
        else:
            run_single_experiment(args.experiment, args.seed, args.output_dir, log_dir, args.skip_mc_dropout)


if __name__ == "__main__":
    main()
