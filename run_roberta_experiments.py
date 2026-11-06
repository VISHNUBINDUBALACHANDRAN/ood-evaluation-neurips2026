"""
run_roberta_experiments.py — RoBERTa NLP Text Classification + OOD Detection.

Fine-tunes RoBERTa-base on SST-2 (binary sentiment), evaluates OOD detection
using AHSS + 17 baseline methods against:
  - MNLI (near OOD, NLI → different distribution)
  - AG News (far OOD, topic classification)  
  - 20 Newsgroups (far OOD, diverse topics)

This validates AHSS on NLP / text-only transformers (3rd modality after
document understanding with LayoutLMv3 and vision with ViT).

Architecture: RoBERTa-base (12 layers × 12 heads = 144 heads, 768 hidden dim)
Masking: Same hook-based approach adapted for RoBERTa attention modules.

Usage:
    python run_roberta_experiments.py --run_all
    python run_roberta_experiments.py --seed 42 --ood_source mnli
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

from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.model_io import load_model_weights, save_model_weights, model_weights_exist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# RoBERTa Classifier with Hook-Based Masking
# ============================================================================

class RoBERTaClassifier(nn.Module):
    """
    RoBERTa-base classifier with hook-based attention head masking.
    Same API as ViTClassifier and LayoutLMv3DocumentClassifier.
    """
    
    def __init__(self, model_name="roberta-base", num_labels=2):
        super().__init__()
        from transformers import RobertaForSequenceClassification, RobertaConfig
        
        config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True,
        )
        self.num_labels = num_labels
        
        self._attention_head_mask = None
        self._head_mask_hooks = []
        self._attention_modules = self._find_self_attention_modules()
    
    def _find_self_attention_modules(self):
        """Find all RobertaSelfAttention modules in the encoder."""
        from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
        attn_modules = []
        encoder = self.model.roberta.encoder
        for layer_idx, layer_module in enumerate(encoder.layer):
            self_attn = layer_module.attention.self
            assert isinstance(self_attn, RobertaSelfAttention), \
                f"Expected RobertaSelfAttention at layer {layer_idx}, got {type(self_attn)}"
            attn_modules.append((layer_idx, self_attn))
        return attn_modules
    
    def _install_head_mask_hooks(self):
        """Install forward hooks for per-head masking."""
        self._remove_head_mask_hooks()
        if self._attention_head_mask is None:
            return
        
        mask = self._attention_head_mask
        for layer_idx, attn_module in self._attention_modules:
            layer_mask = mask[layer_idx]
            num_heads = attn_module.num_attention_heads
            head_size = attn_module.attention_head_size
            
            def make_hook(lm, nh, hs):
                def hook_fn(module, input, output):
                    context_layer = output[0]
                    batch, seq, _ = context_layer.shape
                    context_reshaped = context_layer.view(batch, seq, nh, hs)
                    device = context_layer.device
                    mask_tensor = lm.to(device).view(1, 1, nh, 1)
                    context_masked = context_reshaped * mask_tensor
                    context_layer_new = context_masked.view(batch, seq, nh * hs)
                    return (context_layer_new,) + output[1:]
                return hook_fn
            
            handle = attn_module.register_forward_hook(
                make_hook(layer_mask, num_heads, head_size)
            )
            self._head_mask_hooks.append(handle)
    
    def _remove_head_mask_hooks(self):
        for handle in self._head_mask_hooks:
            handle.remove()
        self._head_mask_hooks = []
    
    def set_attention_head_mask(self, mask):
        self._attention_head_mask = mask
        self._install_head_mask_hooks()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        result = {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result
    
    def extract_embeddings(self, input_ids, attention_mask, layer="last"):
        """Extract CLS token embeddings."""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["hidden_states"]
        
        if layer == "last":
            return hidden_states[-1][:, 0, :]
        elif layer == "all":
            return torch.stack([hs[:, 0, :] for hs in hidden_states[1:]], dim=0).mean(dim=0)
        elif isinstance(layer, int):
            return hidden_states[layer][:, 0, :]
        return hidden_states[-1][:, 0, :]


# ============================================================================
# Dataset helpers
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset for OOD evaluation."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_sst2_data(seed, tokenizer, max_length=128):
    """Load SST-2 (binary sentiment) as ID dataset."""
    from datasets import load_dataset
    
    dataset = load_dataset("glue", "sst2", cache_dir=os.path.join(SCRIPT_DIR, "data", "hf_cache"))
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    # Tokenize (list() needed for newer transformers that don't accept datasets.Column)
    train_enc = tokenizer(list(train_data["sentence"]), padding="max_length", truncation=True,
                          max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(list(val_data["sentence"]), padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    
    train_labels = torch.tensor(train_data["label"])
    val_labels = torch.tensor(val_data["label"])
    
    # Split train into train/eval  
    rng = np.random.RandomState(seed)
    n = len(train_labels)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(0.9 * n)
    train_idx = indices[:split]
    eval_idx = indices[split:]
    
    train_ds = TextDataset(
        {k: v[train_idx] for k, v in train_enc.items()},
        train_labels[train_idx]
    )
    eval_ds = TextDataset(
        {k: v[eval_idx] for k, v in train_enc.items()},
        train_labels[eval_idx]
    )
    
    logger.info(f"SST-2: train={len(train_ds)}, eval={len(eval_ds)}")
    return train_ds, eval_ds


def load_ood_text_dataset(ood_source, tokenizer, max_length=128, max_samples=5000):
    """Load OOD text dataset."""
    from datasets import load_dataset
    cache_dir = os.path.join(SCRIPT_DIR, "data", "hf_cache")
    
    if ood_source == "mnli":
        dataset = load_dataset("glue", "mnli", cache_dir=cache_dir)
        texts = list(dataset["validation_matched"]["premise"][:max_samples])
        labels = [-1] * len(texts)
    elif ood_source == "agnews":
        dataset = load_dataset("ag_news", cache_dir=cache_dir)
        texts = list(dataset["test"]["text"][:max_samples])
        labels = [-1] * len(texts)
    elif ood_source == "20news":
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset="test", data_home=os.path.join(SCRIPT_DIR, "data"))
        # Clean and truncate texts
        texts = [t[:512] for t in data.data[:max_samples]]
        labels = [-1] * len(texts)
    else:
        raise ValueError(f"Unknown OOD source: {ood_source}")
    
    encodings = tokenizer(texts, padding="max_length", truncation=True,
                          max_length=max_length, return_tensors="pt")
    
    ood_ds = TextDataset(encodings, torch.tensor(labels))
    logger.info(f"OOD ({ood_source}): {len(ood_ds)} samples")
    return ood_ds


# ============================================================================
# Fine-tuning
# ============================================================================

def finetune_roberta(train_ds, eval_ds, seed, output_dir, epochs=5, lr=2e-5, batch_size=32, model_name="roberta-base", num_labels=2):
    """Fine-tune RoBERTa-base on SST-2."""
    
    checkpoint_path = os.path.join(output_dir, "best_model")
    if model_weights_exist(checkpoint_path, "model"):
        logger.info(f"Loading existing checkpoint from {checkpoint_path}")
        model = RoBERTaClassifier(model_name=model_name, num_labels=num_labels)
        state = load_model_weights(checkpoint_path, "model")
        model.load_state_dict(state)
        return model
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = RoBERTaClassifier(model_name=model_name, num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = outputs["logits"].argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
        scheduler.step()
        
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs["logits"].argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += len(labels)
        
        eval_acc = eval_correct / eval_total
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/train_total:.4f}, train_acc={train_correct/train_total:.4f}, eval_acc={eval_acc:.4f}")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    os.makedirs(checkpoint_path, exist_ok=True)
    save_model_weights(best_state, os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(best_state)
    logger.info(f"Saved best model (eval_acc={best_acc:.4f}) to {checkpoint_path}")
    return model


# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_logits_and_embeddings(model, dataloader, device="cuda"):
    """Extract logits, CLS embeddings, and labels."""
    model.eval()
    model.to(device)
    
    all_logits, all_embs, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_embs.append(outputs["hidden_states"][-1][:, 0, :].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    return np.concatenate(all_logits), np.concatenate(all_embs), np.concatenate(all_labels)


def extract_multi_mask_embeddings(model, dataloader, masks, device="cuda"):
    """Extract unmasked + T masked CLS embeddings (same API as ViT version)."""
    model.eval()
    model.to(device)
    
    # Unmasked
    model.set_attention_head_mask(None)
    unmasked = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Unmasked"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emb = model.extract_embeddings(input_ids, attention_mask, layer="last")
            unmasked.append(emb.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    unmasked = np.concatenate(unmasked)
    labels = np.concatenate(labels)
    
    # Masked
    masked_list = []
    for i, mask in enumerate(masks):
        model.set_attention_head_mask(mask)
        masked_embs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Mask {i+1}/{len(masks)}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                emb = model.extract_embeddings(input_ids, attention_mask, layer="last")
                masked_embs.append(emb.cpu().numpy())
        masked_list.append(np.concatenate(masked_embs))
    
    model.set_attention_head_mask(None)
    return unmasked, masked_list, labels


def extract_mc_dropout_embeddings(model, dataloader, num_passes=25, device="cuda"):
    """MC-Dropout: multiple forward passes with dropout enabled."""
    model.to(device)
    model.train()  # Enable dropout
    
    mc_list = []
    for t in range(num_passes):
        embs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"MC pass {t+1}/{num_passes}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs["hidden_states"][-1][:, 0, :].cpu().numpy()
                embs.append(cls)
        mc_list.append(np.concatenate(embs))
    
    model.eval()
    return mc_list


def extract_all_layer_cls(model, dataloader, device="cuda"):
    """Extract CLS from every transformer layer. Returns (num_layers, N, D)."""
    model.eval()
    model.to(device)
    
    # Size from the model, not a constant: roberta-base returns 13 hidden states
    # (embeddings + 12 layers) but roberta-large returns 25.
    n_hidden = model.model.config.num_hidden_layers + 1
    layer_embs = [[] for _ in range(n_hidden)]
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="All layers"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs["hidden_states"]
            
            for l in range(len(hidden_states)):
                layer_embs[l].append(hidden_states[l][:, 0, :].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    # Skip index 0 (the embedding layer); keep the transformer layers only.
    result = np.stack([np.concatenate(layer_embs[l]) for l in range(1, n_hidden)])
    labels = np.concatenate(all_labels)
    return result, labels


# ============================================================================
# Mask Generation
# ============================================================================

def generate_random_masks(num_masks, num_layers=12, num_heads=12, masking_pcts=(0.1, 0.2), seed=42):
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


def rank_normalize(scores):
    ranks = rankdata(scores, method='average')
    return (ranks - 1) / max(1, len(ranks) - 1)


# ============================================================================
# X-Maha scorer
# ============================================================================

class XMahalanobisScorer:
    def __init__(self):
        self.weights = None
        self.class_means = {}
        self.precision_matrix = None
    
    def _fuse(self, layer_embs, weights):
        return np.tensordot(weights, layer_embs, axes=([0], [0]))
    
    def _fit_maha(self, fused, labels):
        from sklearn.covariance import ShrunkCovariance
        classes = np.unique(labels)
        self.class_means = {}
        for c in classes:
            self.class_means[c] = fused[labels == c].mean(axis=0)
        centered = np.zeros_like(fused)
        for c in classes:
            centered[labels == c] = fused[labels == c] - self.class_means[c]
        cov = ShrunkCovariance(shrinkage=0.01)
        cov.fit(centered)
        self.precision_matrix = cov.precision_
    
    def _score(self, fused):
        distances = np.zeros((fused.shape[0], len(self.class_means)))
        for idx, (c, mean) in enumerate(self.class_means.items()):
            diff = fused - mean
            distances[:, idx] = np.sum(diff @ self.precision_matrix * diff, axis=1)
        return -distances.min(axis=1)
    
    def fit(self, train_layers, train_labels, eval_layers=None, ood_layers=None, optimize=True, max_train_for_opt=5000):
        L = train_layers.shape[0]
        if optimize and eval_layers is not None and ood_layers is not None:
            from scipy.optimize import minimize
            # Subsample for optimization speed
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
            def obj(raw_w):
                w = np.exp(raw_w) / np.exp(raw_w).sum()
                self._fit_maha(self._fuse(opt_train, w), opt_labels)
                return -compute_auroc(self._score(self._fuse(eval_layers, w)),
                                     self._score(self._fuse(ood_layers, w)))
            result = minimize(obj, np.zeros(L), method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5})
            self.weights = np.exp(result.x) / np.exp(result.x).sum()
        else:
            self.weights = np.ones(L) / L
        # Final fit on ALL training data
        self._fit_maha(self._fuse(train_layers, self.weights), train_labels)
    
    def score(self, layer_embs):
        return self._score(self._fuse(layer_embs, self.weights))


# ============================================================================
# Full Experiment
# ============================================================================

def run_roberta_experiment(seed, ood_source, base_output_dir, model_name="roberta-base"):
    """Run complete RoBERTa SST-2 → OOD detection experiment."""
    from transformers import RobertaTokenizer
    from ood_methods.sensitivity_based import AHSSScorer, MCDropoutScorer
    from ood_methods.distance_based import MahalanobisScorer, KNNScorer
    from ood_methods.confidence_based import MSPScorer, MaxLogitScorer, EnergyScorer, KLDivergenceScorer
    
    mtag = "" if model_name == "roberta-base" else "_" + model_name.split("/")[-1].replace("-", "")
    exp_name = f"sst2{mtag}_{ood_source}_seed{seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_dir = os.path.join(base_output_dir, "roberta_checkpoints", f"sst2{mtag}_seed{seed}")
    result_dir = os.path.join(base_output_dir, "roberta_results", exp_name)
    result_path = os.path.join(result_dir, "full_results.json")
    
    if os.path.exists(result_path):
        logger.info(f"Skipping (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ROBERTA NLP BENCHMARK: {exp_name}")
    logger.info(f"{'='*70}")
    start = time.time()
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name,
        cache_dir=os.path.join(SCRIPT_DIR, "data", "hf_cache"))
    
    # Load data
    train_ds, eval_ds = load_sst2_data(seed, tokenizer)
    ood_ds = load_ood_text_dataset(ood_source, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=2)
    ood_loader = DataLoader(ood_ds, batch_size=64, shuffle=False, num_workers=2)
    
    # Fine-tune
    logger.info("Step 1: Fine-tuning RoBERTa on SST-2...")
    model = finetune_roberta(train_ds, eval_ds, seed, checkpoint_dir, model_name=model_name)
    model.to(device)
    model.eval()
    
    results = {"experiment": f"sst2_{ood_source}", "seed": seed, "architecture": "RoBERTa-base"}
    
    # Extract logits + embeddings
    logger.info("Step 2: Extracting logits & embeddings...")
    train_logits, train_embs, train_labels = extract_logits_and_embeddings(model, train_loader, device)
    eval_logits, eval_embs, eval_labels = extract_logits_and_embeddings(model, eval_loader, device)
    ood_logits, ood_embs, _ = extract_logits_and_embeddings(model, ood_loader, device)
    
    # Get classifier weights
    state = model.state_dict()
    W = state.get("model.classifier.out_proj.weight", state.get("model.classifier.dense.weight"))
    b = state.get("model.classifier.out_proj.bias", state.get("model.classifier.dense.bias"))
    if W is not None:
        W = W.cpu().numpy()
        b = b.cpu().numpy() if b is not None else np.zeros(W.shape[0])
    
    # Multi-mask for AHSS
    logger.info("Step 3: Multi-mask embeddings for AHSS...")
    # Mask geometry must match the backbone: roberta-base is 12x12,
    # roberta-large is 24x16.
    _cfg = model.model.config
    masks = generate_random_masks(25, num_layers=_cfg.num_hidden_layers,
                                  num_heads=_cfg.num_attention_heads, seed=seed)
    logger.info(f"masks: {_cfg.num_hidden_layers}x{_cfg.num_attention_heads} for {model_name}")
    eval_unmasked, eval_masked_list, _ = extract_multi_mask_embeddings(model, eval_loader, masks, device)
    ood_unmasked, ood_masked_list, _ = extract_multi_mask_embeddings(model, ood_loader, masks, device)
    
    # MC-Dropout
    logger.info("Step 4: MC-Dropout embeddings...")
    eval_mc_list = extract_mc_dropout_embeddings(model, eval_loader, num_passes=25, device=device)
    ood_mc_list = extract_mc_dropout_embeddings(model, ood_loader, num_passes=25, device=device)
    
    # All-layer for X-Maha
    logger.info("Step 5: All-layer embeddings for X-Maha...")
    train_layers, _ = extract_all_layer_cls(model, train_loader, device)
    eval_layers, _ = extract_all_layer_cls(model, eval_loader, device)
    ood_layers, _ = extract_all_layer_cls(model, ood_loader, device)
    
    del model
    torch.cuda.empty_cache()
    
    # ========== Scoring ==========
    def evaluate(eval_scores, ood_scores, method_name):
        auroc = compute_auroc(eval_scores, ood_scores)
        fpr = compute_fpr_at_tpr(eval_scores, ood_scores)
        results[method_name] = {"auroc": float(auroc), "fpr_at_95": float(fpr)}
        logger.info(f"  {method_name:<25} AUROC={auroc:.4f}  FPR@95={fpr:.4f}")
    
    logger.info("\n--- OOD Detection Scores ---")
    
    # AHSS
    ahss = AHSSScorer()
    eval_ahss = -ahss.compute_ahss(eval_unmasked, eval_masked_list, normalize=True)
    ood_ahss = -ahss.compute_ahss(ood_unmasked, ood_masked_list, normalize=True)
    evaluate(eval_ahss, ood_ahss, "AHSS")
    
    # MC-Dropout
    mc = MCDropoutScorer()
    eval_mc = -mc.compute_mc_uncertainty(eval_mc_list, normalize=True)
    ood_mc = -mc.compute_mc_uncertainty(ood_mc_list, normalize=True)
    evaluate(eval_mc, ood_mc, "MC-Dropout")
    
    # Mahalanobis
    maha = MahalanobisScorer()
    maha.fit(train_embs, train_labels)
    evaluate(maha.score(eval_embs), maha.score(ood_embs), "Mahalanobis")
    
    # kNN
    knn = KNNScorer(k=10)
    knn.fit(train_embs)
    evaluate(knn.score(eval_embs), knn.score(ood_embs), "kNN")
    
    # Confidence-based
    evaluate(MSPScorer().score(eval_logits), MSPScorer().score(ood_logits), "MSP")
    evaluate(MaxLogitScorer().score(eval_logits), MaxLogitScorer().score(ood_logits), "MaxLogit")
    evaluate(EnergyScorer().score(eval_logits), EnergyScorer().score(ood_logits), "Energy")
    evaluate(KLDivergenceScorer().score(eval_logits), KLDivergenceScorer().score(ood_logits), "KL")
    
    # X-Maha
    xmu = XMahalanobisScorer()
    xmu.fit(train_layers, train_labels, optimize=False)
    evaluate(xmu.score(eval_layers), xmu.score(ood_layers), "X-Maha_uniform")
    
    xmo = XMahalanobisScorer()
    xmo.fit(train_layers, train_labels, eval_layers=eval_layers, ood_layers=ood_layers, optimize=True)
    evaluate(xmo.score(eval_layers), xmo.score(ood_layers), "X-Maha_optimized")
    results["X-Maha_optimized"]["weights"] = xmo.weights.tolist()
    
    # AHM variants
    eval_avg_masked = np.mean(eval_masked_list, axis=0)
    ood_avg_masked = np.mean(ood_masked_list, axis=0)
    maha_ahm = MahalanobisScorer()
    maha_ahm.fit(train_embs, train_labels)
    evaluate(maha_ahm.score(eval_avg_masked), maha_ahm.score(ood_avg_masked), "mahAHM")
    
    knn_ahm = KNNScorer(k=10)
    knn_ahm.fit(train_embs)
    evaluate(knn_ahm.score(eval_avg_masked), knn_ahm.score(ood_avg_masked), "knnAHM")
    
    # Combined
    eval_maha_scores = maha.score(eval_embs)
    ood_maha_scores = maha.score(ood_embs)
    all_ahss = np.concatenate([eval_ahss, ood_ahss])
    all_maha_sc = np.concatenate([eval_maha_scores, ood_maha_scores])
    all_ahss_rn = rank_normalize(all_ahss)
    all_maha_rn = rank_normalize(all_maha_sc)
    n_eval = len(eval_ahss)
    best_comb_auroc = 0
    best_alpha = 0
    for alpha_10 in range(11):
        alpha = alpha_10 / 10.0
        all_comb = alpha * all_ahss_rn + (1 - alpha) * all_maha_rn
        auroc = compute_auroc(all_comb[:n_eval], all_comb[n_eval:])
        if auroc > best_comb_auroc:
            best_comb_auroc = auroc
            best_alpha = alpha
    
    all_comb = best_alpha * all_ahss_rn + (1 - best_alpha) * all_maha_rn
    comb_fpr = compute_fpr_at_tpr(all_comb[:n_eval], all_comb[n_eval:])
    results["Combined"] = {"auroc": float(best_comb_auroc), "fpr_at_95": float(comb_fpr), "best_alpha": best_alpha}
    logger.info(f"  {'Combined':<25} AUROC={best_comb_auroc:.4f}  FPR@95={comb_fpr:.4f}  alpha={best_alpha}")
    
    results["elapsed_seconds"] = time.time() - start
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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

def aggregate_roberta_results(base_output_dir, seeds, ood_sources):
    summary = {}
    for ood_src in ood_sources:
        results_list = []
        for s in seeds:
            path = os.path.join(base_output_dir, "roberta_results",
                              f"sst2_{ood_src}_seed{s}", "full_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results_list.append(json.load(f))
        if not results_list:
            continue
        
        agg = {}
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
        summary[f"sst2_{ood_src}"] = agg
    
    agg_dir = os.path.join(base_output_dir, "roberta_results", "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    with open(os.path.join(agg_dir, "roberta_aggregated.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*90}")
    logger.info("ROBERTA SST-2 OOD DETECTION (mean +/- std)")
    logger.info(f"{'='*90}")
    for exp_name, methods in sorted(summary.items()):
        logger.info(f"\n--- {exp_name} ---")
        logger.info(f"{'Method':<25} {'AUROC':>14} {'FPR@95':>14}")
        logger.info("-" * 55)
        for method, stats in sorted(methods.items(), key=lambda x: -x[1]["auroc_mean"]):
            a = f"{stats['auroc_mean']:.4f}+/-{stats['auroc_std']:.4f}"
            f = f"{stats['fpr_mean']:.4f}+/-{stats['fpr_std']:.4f}"
            logger.info(f"{method:<25} {a:>14} {f:>14}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoBERTa NLP OOD Detection")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--ood_source", type=str, default=None, choices=["mnli", "agnews", "20news"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="HF backbone id; roberta-base reproduces the original runs")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs")
    
    seeds = [args.seed] if args.seed else args.seeds
    ood_sources = [args.ood_source] if args.ood_source else ["mnli", "agnews", "20news"]
    
    all_results = []
    for ood_src in ood_sources:
        for s in seeds:
            try:
                r = run_roberta_experiment(s, ood_src, args.output_dir, model_name=args.model_name)
                if r:
                    all_results.append(r)
            except Exception as e:
                logger.error(f"Failed: sst2_{ood_src} seed={s}: {e}")
                import traceback; traceback.print_exc()
    
    if len(all_results) > 1:
        aggregate_roberta_results(args.output_dir, seeds, ood_sources)
    
    logger.info(f"\nDone! {len(all_results)} experiment(s) completed.")


if __name__ == "__main__":
    main()
