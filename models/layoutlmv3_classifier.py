"""
LayoutLMv3 Document Classifier.
Wraps LayoutLMv3ForSequenceClassification with embedding extraction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    LayoutLMv3Config,
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class LayoutLMv3DocumentClassifier(nn.Module):
    """
    LayoutLMv3-based document classifier with embedding extraction hooks.
    
    Supports:
    - Standard classification (forward pass → logits)
    - Dense embedding extraction (CLS token from last hidden state)
    - Multi-layer embedding extraction (CLS from all layers, for AvgAvg)
    - Attention weight capture (for analysis/debugging)
    
    Note: Attention head masking is implemented via PyTorch forward hooks
    on LayoutLMv3SelfAttention modules. This is necessary because
    transformers >= 5.x removed native head_mask support from LayoutLMv3.
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        num_labels: int = 9,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load LayoutLMv3 config
        config = LayoutLMv3Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        
        if label2id is not None:
            config.label2id = label2id
        if id2label is not None:
            config.id2label = id2label
        
        # Load model
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        
        # Storage for attention head masks (used by AHM)
        self._attention_head_mask = None
        self._head_mask_hooks = []  # Forward hooks for head masking
        
        # Cache attention module references for hook installation
        self._attention_modules = self._find_self_attention_modules()
        
        # Storage for captured attention weights
        self._captured_attentions = []

    def _find_self_attention_modules(self):
        """
        Find all LayoutLMv3SelfAttention modules in the encoder, ordered by layer.
        Returns a list of (layer_idx, module) tuples.
        """
        from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3SelfAttention
        attn_modules = []
        # The encoder is at self.model.layoutlmv3.encoder.layer
        encoder = self.model.layoutlmv3.encoder
        for layer_idx, layer_module in enumerate(encoder.layer):
            # Each layer has .attention.self which is LayoutLMv3SelfAttention
            self_attn = layer_module.attention.self
            assert isinstance(self_attn, LayoutLMv3SelfAttention), \
                f"Expected LayoutLMv3SelfAttention at layer {layer_idx}, got {type(self_attn)}"
            attn_modules.append((layer_idx, self_attn))
        return attn_modules

    def _install_head_mask_hooks(self):
        """
        Install forward hooks on all self-attention modules to apply head masking.
        
        The hook intercepts the output of LayoutLMv3SelfAttention.forward(),
        which returns context_layer of shape (batch, seq, all_head_size=768).
        We reshape to (batch, seq, num_heads, head_size), apply the binary mask
        per head, and reshape back. This is mathematically equivalent to zeroing
        attention scores before the value matmul (as described in the paper's
        Figure 1) for binary masks.
        """
        self._remove_head_mask_hooks()  # Clear any existing hooks
        
        if self._attention_head_mask is None:
            return
        
        mask = self._attention_head_mask  # (num_layers, num_heads)
        
        for layer_idx, attn_module in self._attention_modules:
            layer_mask = mask[layer_idx]  # (num_heads,)
            num_heads = attn_module.num_attention_heads
            head_size = attn_module.attention_head_size
            
            def make_hook(lm, nh, hs):
                """Create a closure for this layer's mask."""
                def hook_fn(module, input, output):
                    # output is a tuple: (context_layer,) or (context_layer, attention_probs)
                    context_layer = output[0]  # (batch, seq, all_head_size)
                    batch, seq, _ = context_layer.shape
                    # Reshape to (batch, seq, num_heads, head_size)
                    context_reshaped = context_layer.view(batch, seq, nh, hs)
                    # Apply mask: (1, 1, num_heads, 1)
                    device = context_layer.device
                    mask_tensor = lm.to(device).view(1, 1, nh, 1)
                    context_masked = context_reshaped * mask_tensor
                    # Reshape back to (batch, seq, all_head_size)
                    context_layer_new = context_masked.view(batch, seq, nh * hs)
                    # Return modified output tuple
                    return (context_layer_new,) + output[1:]
                return hook_fn
            
            handle = attn_module.register_forward_hook(
                make_hook(layer_mask, num_heads, head_size)
            )
            self._head_mask_hooks.append(handle)

    def _remove_head_mask_hooks(self):
        """Remove all installed head mask hooks."""
        for handle in self._head_mask_hooks:
            handle.remove()
        self._head_mask_hooks = []
        
    def set_attention_head_mask(self, mask: Optional[torch.Tensor]):
        """
        Set the attention head mask matrix and install/remove forward hooks.
        
        Args:
            mask: Tensor of shape (num_layers, num_heads) with values in {0, 1}.
                  None to disable masking.
        """
        self._attention_head_mask = mask
        self._install_head_mask_hooks()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Returns dict with:
            - logits: Classification logits (batch_size, num_labels)
            - loss: Cross-entropy loss (if labels provided)
            - hidden_states: All layer hidden states (tuple of (batch_size, seq_len, hidden_size))
            - attentions: All layer attention weights (tuple of (batch_size, num_heads, seq_len, seq_len))
        """
        # Head masking is applied via forward hooks installed by set_attention_head_mask().
        # No need to pass head_mask parameter (not supported in transformers >= 5.x).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        layer: str = "last",
    ) -> torch.Tensor:
        """
        Extract dense embeddings (CLS token) from the model.
        
        Args:
            layer: "last" for last hidden state, "all" for all layers averaged,
                   or int for specific layer index.
                   
        Returns:
            Embeddings of shape (batch_size, hidden_size)
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
        
        hidden_states = outputs["hidden_states"]  # tuple of (batch, seq_len, hidden)
        
        if layer == "last":
            # CLS token from last hidden layer
            embeddings = hidden_states[-1][:, 0, :]  # (batch, hidden_size)
        elif layer == "all":
            # AvgAvg method (Chen et al. 2022): average across both
            # sequence positions AND layers for holistic sentence embeddings
            all_seq_avg = torch.stack(
                [hs.mean(dim=1) for hs in hidden_states], dim=0
            )  # (num_layers+1, batch, hidden)
            embeddings = all_seq_avg.mean(dim=0)  # (batch, hidden)
        elif isinstance(layer, int):
            embeddings = hidden_states[layer][:, 0, :]
        else:
            raise ValueError(f"Unknown layer specification: {layer}")
        
        return embeddings
    
    def extract_all_layer_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        skip_embedding_layer: bool = True,
    ) -> List[torch.Tensor]:
        """
        Extract sequence-averaged embeddings from ALL transformer layers.
        Returns list of tensors, each (batch_size, hidden_size).
        Used by AvgAvg (Chen et al. 2022) and Gnome methods.
        
        Per Chen et al. 2022 "Holistic Sentence Embeddings", AvgAvg averages
        across both sequence positions and layers. This method returns per-layer
        sequence-averaged embeddings; the caller then averages across layers.
        
        Args:
            skip_embedding_layer: If True (default), skip layer 0 (embedding layer)
                and return only the 12 transformer layers (layers 1-12).
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
        
        hidden_states = outputs["hidden_states"]
        # hidden_states[0] = embedding layer output, hidden_states[1:] = transformer layers
        start_idx = 1 if skip_embedding_layer else 0
        # Average across sequence positions (dim=1) for each layer
        return [hs.mean(dim=1) for hs in hidden_states[start_idx:]]
    
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get classification logits. Shape: (batch_size, num_labels)."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
        return outputs["logits"]


def extract_embeddings_with_multiple_masks(
    model: LayoutLMv3DocumentClassifier,
    dataloader: torch.utils.data.DataLoader,
    masks: List[torch.Tensor],
    device: str = "cuda",
    layer: str = "last",
    include_unmasked: bool = True,
    desc: str = "Extracting multi-mask embeddings",
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Extract embeddings under multiple attention head masks + unmasked.
    Used by AHSS to compute sensitivity scores.

    Args:
        model: LayoutLMv3DocumentClassifier
        dataloader: DataLoader for the dataset
        masks: list of T mask tensors, each (num_layers, num_heads)
        device: compute device
        layer: embedding layer ("last", "all", or int)
        include_unmasked: if True, also extract unmasked embeddings
        desc: progress bar description

    Returns:
        unmasked_embeddings: (N, D) or None if include_unmasked=False
        masked_embeddings_list: list of T arrays, each (N, D)
        labels: (N,)
    """
    model.eval()
    model.to(device)

    # First pass: unmasked embeddings
    unmasked_embs = None
    labels_arr = None
    if include_unmasked:
        model.set_attention_head_mask(None)
        all_embs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [unmasked]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                emb = model.extract_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    layer=layer,
                )
                all_embs.append(emb.cpu().numpy())
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu().numpy())
        unmasked_embs = np.concatenate(all_embs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    # T passes: one per mask
    masked_embeddings_list = []
    for mask_idx, mask in enumerate(masks):
        model.set_attention_head_mask(mask)
        all_embs = []
        all_labels_m = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [mask {mask_idx+1}/{len(masks)}]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                emb = model.extract_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    layer=layer,
                )
                all_embs.append(emb.cpu().numpy())
                if labels_arr is None and "labels" in batch:
                    all_labels_m.append(batch["labels"].cpu().numpy())
        masked_embeddings_list.append(np.concatenate(all_embs, axis=0))
        if labels_arr is None and all_labels_m:
            labels_arr = np.concatenate(all_labels_m, axis=0)

    # Remove mask
    model.set_attention_head_mask(None)

    if labels_arr is None:
        labels_arr = np.array([])

    return unmasked_embs, masked_embeddings_list, labels_arr


def extract_embeddings_from_dataloader(
    model: LayoutLMv3DocumentClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    layer: str = "last",
    desc: str = "Extracting embeddings",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for entire dataset via dataloader.
    
    Returns:
        embeddings: np.ndarray of shape (N, hidden_size)
        labels: np.ndarray of shape (N,)
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            embeddings = model.extract_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                layer=layer,
            )
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    return all_embeddings, all_labels


def extract_logits_from_dataloader(
    model: LayoutLMv3DocumentClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    desc: str = "Extracting logits",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract logits for entire dataset via dataloader.
    
    Returns:
        logits: np.ndarray of shape (N, num_labels)
        labels: np.ndarray of shape (N,)
    """
    model.eval()
    model.to(device)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            logits = model.get_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
            
            all_logits.append(logits.cpu().numpy())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    return all_logits, all_labels


def extract_all_layer_embeddings_from_dataloader(
    model: LayoutLMv3DocumentClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    desc: str = "Extracting multi-layer embeddings",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract CLS embeddings from all layers for entire dataset.
    
    Returns:
        layer_embeddings: List of np.ndarray, each (N, hidden_size), one per layer
        labels: np.ndarray of shape (N,)
    """
    model.eval()
    model.to(device)
    
    all_layer_embeddings = None
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            layer_embs = model.extract_all_layer_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
            
            if all_layer_embeddings is None:
                all_layer_embeddings = [[] for _ in range(len(layer_embs))]
            
            for i, emb in enumerate(layer_embs):
                all_layer_embeddings[i].append(emb.cpu().numpy())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())
    
    # Concatenate
    all_layer_embeddings = [
        np.concatenate(layer_embs, axis=0)
        for layer_embs in all_layer_embeddings
    ]
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    return all_layer_embeddings, all_labels


def extract_mc_dropout_embeddings(
    model: LayoutLMv3DocumentClassifier,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 25,
    device: str = "cuda",
    layer: str = "last",
    desc: str = "MC-Dropout extraction",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract embeddings using MC-Dropout (model in train mode for stochastic dropout).

    For each forward pass, dropout is active, producing different embeddings.
    The variance across passes measures epistemic uncertainty.

    Args:
        model: LayoutLMv3DocumentClassifier
        dataloader: DataLoader for the dataset
        n_samples: Number of stochastic forward passes (T)
        device: compute device
        layer: embedding layer specification

    Returns:
        embedding_samples: list of T arrays, each (N, D)
        labels: (N,)
    """
    model.to(device)
    # Enable dropout by putting model in train mode
    model.train()
    # But disable gradient computation
    model.set_attention_head_mask(None)

    embedding_samples = []
    labels_arr = None

    for t in range(n_samples):
        all_embs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} [pass {t+1}/{n_samples}]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                # Forward through model (dropout active in train mode)
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states
                if layer == "last":
                    emb = hidden_states[-1][:, 0, :]
                elif layer == "all":
                    all_seq_avg = torch.stack(
                        [hs.mean(dim=1) for hs in hidden_states], dim=0
                    )
                    emb = all_seq_avg.mean(dim=0)
                else:
                    emb = hidden_states[int(layer)][:, 0, :]

                all_embs.append(emb.cpu().numpy())
                if labels_arr is None and "labels" in batch:
                    all_labels.append(batch["labels"].cpu().numpy())

        embedding_samples.append(np.concatenate(all_embs, axis=0))
        if labels_arr is None and all_labels:
            labels_arr = np.concatenate(all_labels, axis=0)

    # Restore eval mode
    model.eval()

    if labels_arr is None:
        labels_arr = np.array([])

    return embedding_samples, labels_arr
