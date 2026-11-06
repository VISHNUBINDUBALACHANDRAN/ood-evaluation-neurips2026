"""
ViT (Vision Transformer) Classifier for AHSS Cross-Architecture Experiments.

Mirrors the LayoutLMv3DocumentClassifier API:
- Hook-based attention head masking
- CLS/mean-pool embedding extraction
- Multi-mask embedding extraction for AHSS
- MC-Dropout embedding extraction

Architecture: ViT-B/16 (12 layers × 12 heads = 144 heads, 768 hidden dim)
Target datasets: CIFAR-10 (9 ID + 1 OOD class), with SVHN/Textures as near/far OOD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTConfig
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class ViTClassifier(nn.Module):
    """
    ViT-based image classifier with hook-based attention head masking.
    
    Supports:
    - Standard classification (forward → logits)
    - CLS token embedding extraction
    - Multi-mask embedding extraction for AHSS
    - MC-Dropout embedding extraction
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 9,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        config = ViTConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        
        if label2id is not None:
            config.label2id = label2id
        if id2label is not None:
            config.id2label = id2label
        
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        
        # Attention head masking state
        self._attention_head_mask = None
        self._head_mask_hooks = []
        
        # Cache attention module references
        self._attention_modules = self._find_self_attention_modules()

    def _find_self_attention_modules(self):
        """Find all ViTSelfAttention modules in the encoder, ordered by layer."""
        from transformers.models.vit.modeling_vit import ViTSelfAttention
        attn_modules = []
        encoder = self.model.vit.encoder
        for layer_idx, layer_module in enumerate(encoder.layer):
            self_attn = layer_module.attention.attention
            assert isinstance(self_attn, ViTSelfAttention), \
                f"Expected ViTSelfAttention at layer {layer_idx}, got {type(self_attn)}"
            attn_modules.append((layer_idx, self_attn))
        return attn_modules

    def _install_head_mask_hooks(self):
        """Install forward hooks for per-head masking on ViTSelfAttention modules."""
        self._remove_head_mask_hooks()
        
        if self._attention_head_mask is None:
            return
        
        mask = self._attention_head_mask  # (num_layers, num_heads)
        
        for layer_idx, attn_module in self._attention_modules:
            layer_mask = mask[layer_idx]  # (num_heads,)
            num_heads = attn_module.num_attention_heads
            head_size = attn_module.attention_head_size
            
            def make_hook(lm, nh, hs):
                def hook_fn(module, input, output):
                    # output is a tuple: (context_layer, attention_probs) or just (context_layer,)
                    context_layer = output[0]  # (batch, seq, all_head_size)
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

    def set_attention_head_mask(self, mask: Optional[torch.Tensor]):
        self._attention_head_mask = mask
        self._install_head_mask_hooks()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            pixel_values=pixel_values,
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

    def extract_embeddings(
        self,
        pixel_values: torch.Tensor,
        layer: str = "last",
    ) -> torch.Tensor:
        """Extract CLS token embeddings from specified layer."""
        with torch.no_grad():
            outputs = self.forward(pixel_values=pixel_values)
        
        hidden_states = outputs["hidden_states"]
        
        if layer == "last":
            emb = hidden_states[-1][:, 0, :]  # CLS token from last layer
        elif layer == "all":
            # Average CLS across all transformer layers (skip embedding layer)
            emb = torch.stack([hs[:, 0, :] for hs in hidden_states[1:]], dim=0).mean(dim=0)
        elif isinstance(layer, int):
            emb = hidden_states[layer][:, 0, :]
        else:
            emb = hidden_states[-1][:, 0, :]
        
        return emb


def extract_vit_all_layer_cls(
    model: ViTClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    desc: str = "Extracting all layers",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS token from all 12 transformer layers (skip embedding layer 0).
    
    Returns:
        layer_embeddings: (12, N, 768) — CLS from each transformer layer
        labels: (N,)
    """
    model.eval()
    model.to(device)

    layer_embs = [[] for _ in range(13)]
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pixel_values)
            hidden_states = outputs["hidden_states"]

            for l in range(len(hidden_states)):
                layer_embs[l].append(hidden_states[l][:, 0, :].cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
            elif "label" in batch:
                all_labels.append(batch["label"].cpu().numpy())

    result = np.stack([np.concatenate(layer_embs[l]) for l in range(1, 13)])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return result, labels


def extract_vit_embeddings_with_multiple_masks(
    model: ViTClassifier,
    dataloader: torch.utils.data.DataLoader,
    masks: List[torch.Tensor],
    device: str = "cuda",
    layer: str = "last",
    include_unmasked: bool = True,
    desc: str = "Extracting multi-mask embeddings",
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Extract ViT embeddings under multiple attention head masks.
    API matches extract_embeddings_with_multiple_masks for LayoutLMv3.
    """
    model.eval()
    model.to(device)

    unmasked_embs = None
    labels_arr = None

    if include_unmasked:
        model.set_attention_head_mask(None)
        all_embs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [unmasked]"):
                pixel_values = batch["pixel_values"].to(device)
                emb = model.extract_embeddings(pixel_values=pixel_values, layer=layer)
                all_embs.append(emb.cpu().numpy())
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu().numpy())
                elif "label" in batch:
                    all_labels.append(batch["label"].cpu().numpy())
        unmasked_embs = np.concatenate(all_embs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    masked_embeddings_list = []
    for mask_idx, mask in enumerate(masks):
        model.set_attention_head_mask(mask)
        all_embs = []
        all_labels_m = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [mask {mask_idx+1}/{len(masks)}]"):
                pixel_values = batch["pixel_values"].to(device)
                emb = model.extract_embeddings(pixel_values=pixel_values, layer=layer)
                all_embs.append(emb.cpu().numpy())
                if labels_arr is None:
                    if "labels" in batch:
                        all_labels_m.append(batch["labels"].cpu().numpy())
                    elif "label" in batch:
                        all_labels_m.append(batch["label"].cpu().numpy())
        masked_embeddings_list.append(np.concatenate(all_embs, axis=0))
        if labels_arr is None and all_labels_m:
            labels_arr = np.concatenate(all_labels_m, axis=0)

    model.set_attention_head_mask(None)
    if labels_arr is None:
        labels_arr = np.array([])

    return unmasked_embs, masked_embeddings_list, labels_arr


def extract_vit_mc_dropout_embeddings(
    model: ViTClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    n_forward: int = 25,
    layer: str = "last",
    desc: str = "MC-Dropout extraction",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    MC-Dropout: run model in train mode (dropout active) for T forward passes.
    Returns list of T embedding arrays + labels.
    """
    model.to(device)
    model.set_attention_head_mask(None)
    
    all_pass_embeddings = []
    labels_arr = None
    
    for pass_idx in range(n_forward):
        model.train()  # enable dropout
        all_embs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [pass {pass_idx+1}/{n_forward}]"):
                pixel_values = batch["pixel_values"].to(device)
                outputs = model.forward(pixel_values=pixel_values)
                hidden_states = outputs["hidden_states"]
                if layer == "last":
                    emb = hidden_states[-1][:, 0, :]
                else:
                    emb = hidden_states[-1][:, 0, :]
                all_embs.append(emb.cpu().numpy())
                if labels_arr is None:
                    if "labels" in batch:
                        all_labels.append(batch["labels"].cpu().numpy())
                    elif "label" in batch:
                        all_labels.append(batch["label"].cpu().numpy())
        all_pass_embeddings.append(np.concatenate(all_embs, axis=0))
        if labels_arr is None and all_labels:
            labels_arr = np.concatenate(all_labels, axis=0)
    
    model.eval()
    if labels_arr is None:
        labels_arr = np.array([])
    
    return all_pass_embeddings, labels_arr


def extract_vit_embeddings_from_dataloader(
    model: ViTClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    layer: str = "last",
    desc: str = "Extracting embeddings",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for entire dataset."""
    model.eval()
    model.to(device)
    model.set_attention_head_mask(None)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch["pixel_values"].to(device)
            emb = model.extract_embeddings(pixel_values=pixel_values, layer=layer)
            all_embeddings.append(emb.cpu().numpy())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
            elif "label" in batch:
                all_labels.append(batch["label"].cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    return all_embeddings, all_labels
