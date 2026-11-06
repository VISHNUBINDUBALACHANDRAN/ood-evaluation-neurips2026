"""
Differentiable Attention Head Masking (D-AHM).

Instead of random search over binary masks (AHM: 25 random trials),
D-AHM uses gradient-based optimization with continuous relaxation to
find masks that maximize ID embedding stability.

Key technical choices:
  - Continuous relaxation: sigmoid(alpha/tau) with temperature annealing
  - Objective: minimize ID-eval embedding displacement under masking
    (= find masks where ID embeddings are MOST stable)
  - This implicitly maximizes AHSS discriminability: masks that barely
    change ID embeddings will dramatically change OOD embeddings
  - Uses Straight-Through Estimator for discretization at test time

References:
  - Voita et al. (ACL 2019) — differentiable L0 gates for head pruning
  - Jang et al. (ICLR 2017) — Gumbel-Softmax for discrete relaxation
  - Our key difference: optimization objective is STABILITY, not EFFICIENCY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import os
import json
import logging

logger = logging.getLogger(__name__)


class DifferentiableMaskModule(nn.Module):
    """
    Learnable continuous mask parameters for attention heads.

    Parameters:
        alpha: (num_layers, num_heads) — logits that determine mask values
        Mask at temperature tau: mask = sigmoid(alpha / tau)
        As tau → 0, sigmoid becomes a step function (hard mask)
    """

    def __init__(self, num_layers: int = 12, num_heads: int = 12,
                 init_value: float = 2.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Initialize with positive values so initial mask ≈ all-ones
        self.alpha = nn.Parameter(
            torch.full((num_layers, num_heads), init_value)
        )

    def get_soft_mask(self, tau: float = 1.0) -> torch.Tensor:
        """Get continuous mask via sigmoid with temperature."""
        return torch.sigmoid(self.alpha / tau)

    def get_hard_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get binary mask by thresholding sigmoid(alpha)."""
        with torch.no_grad():
            soft = torch.sigmoid(self.alpha)
            return (soft > threshold).float()

    def get_straight_through_mask(self, tau: float = 1.0) -> torch.Tensor:
        """
        Straight-Through Estimator: forward uses hard mask,
        backward uses soft gradient through sigmoid.
        """
        soft = torch.sigmoid(self.alpha / tau)
        hard = (soft > 0.5).float()
        # Straight-through: use hard in forward, soft gradient in backward
        return hard - soft.detach() + soft

    def get_masking_rate(self) -> float:
        """Current fraction of heads being masked (mask value < 0.5)."""
        with torch.no_grad():
            soft = torch.sigmoid(self.alpha)
            return float((soft < 0.5).float().mean())

    def get_mask_summary(self) -> Dict:
        """Summary statistics of current mask state."""
        with torch.no_grad():
            soft = torch.sigmoid(self.alpha)
            return {
                "mean": float(soft.mean()),
                "min": float(soft.min()),
                "max": float(soft.max()),
                "masking_rate": self.get_masking_rate(),
                "per_layer_masking": [
                    float((soft[i] < 0.5).float().mean())
                    for i in range(self.num_layers)
                ],
            }


class DifferentiableAHMOptimizer:
    """
    Gradient-based optimization of attention head masks for AHSS.

    Objective: Find masks where ID embeddings are MOST STABLE.
    This means finding masks that, when applied, barely change the
    [CLS] embedding for ID data. Such masks will maximally disrupt
    OOD embeddings (since OOD representations are fragile).

    Loss = mean_x ||f(x; M_soft) - f(x; 1)||^2 + lambda * sparsity_reg

    where:
      - f(x; M_soft) = embedding with soft mask applied
      - f(x; 1) = embedding with no mask (all heads active)
      - sparsity_reg encourages a target masking rate (e.g., 10-20%)

    Temperature annealing: tau starts high (soft masks) → decreases to
    near-zero (hard masks). This follows curriculum learning practice.
    """

    def __init__(
        self,
        model,
        num_layers: int = 12,
        num_heads: int = 12,
        lr: float = 0.01,
        num_steps: int = 200,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        target_masking_rate: float = 0.15,
        sparsity_lambda: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.lr = lr
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.target_masking_rate = target_masking_rate
        self.sparsity_lambda = sparsity_lambda

        self.mask_module = DifferentiableMaskModule(
            num_layers, num_heads
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.mask_module.parameters(), lr=lr
        )

    def _get_tau(self, step: int) -> float:
        """Cosine annealing schedule for temperature."""
        progress = step / max(self.num_steps - 1, 1)
        tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1 + np.cos(np.pi * progress)
        )
        return tau

    def _compute_embedding_with_mask(self, batch, mask_tensor):
        """Forward pass with a given mask tensor (differentiable)."""
        # Install the soft mask via hooks
        self.model.set_attention_head_mask(mask_tensor)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        bbox = batch["bbox"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)

        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        # CLS token from last hidden state
        embedding = outputs.hidden_states[-1][:, 0, :]
        return embedding

    def _compute_embedding_unmasked(self, batch):
        """Forward pass with no mask (reference embedding)."""
        self.model.set_attention_head_mask(None)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        bbox = batch["bbox"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        return outputs.hidden_states[-1][:, 0, :].detach()

    def optimize(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_masks_to_produce: int = 5,
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Run gradient-based mask optimization.

        Args:
            dataloader: ID training/eval dataloader
            num_masks_to_produce: number of diverse binary masks to output

        Returns:
            masks: list of (num_layers, num_heads) binary tensors
            metadata: dict with training curves and analysis
        """
        self.model.to(self.device)
        self.model.eval()  # Freeze model weights
        # But note: mask_module is in train mode for optimization

        # Pre-compute unmasked embeddings for all batches
        logger.info("Pre-computing unmasked reference embeddings...")
        ref_embeddings = []
        batches_cache = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Reference embeddings"):
                ref_emb = self._compute_embedding_unmasked(batch)
                ref_embeddings.append(ref_emb)
                batches_cache.append(batch)

        history = {
            "loss": [], "displacement": [], "sparsity": [],
            "tau": [], "masking_rate": [],
        }

        logger.info(f"Starting D-AHM optimization: {self.num_steps} steps, "
                     f"lr={self.lr}, tau={self.tau_start}→{self.tau_end}")

        for step in range(self.num_steps):
            tau = self._get_tau(step)
            soft_mask = self.mask_module.get_soft_mask(tau)

            total_displacement = 0.0
            n_samples = 0

            # Sample a subset of batches for efficiency
            batch_indices = np.random.choice(
                len(batches_cache),
                size=min(8, len(batches_cache)),
                replace=False,
            )

            self.optimizer.zero_grad()

            for bi in batch_indices:
                batch = batches_cache[bi]
                ref_emb = ref_embeddings[bi]

                # Forward with soft mask (gradients flow through sigmoid)
                masked_emb = self._compute_embedding_with_mask(batch, soft_mask)

                # Displacement loss: ||f(x; M_soft) - f(x; 1)||^2
                displacement = ((masked_emb - ref_emb) ** 2).sum(dim=1).mean()
                total_displacement += displacement
                n_samples += 1

            avg_displacement = total_displacement / n_samples

            # Sparsity regularization: push toward target masking rate
            current_masking_rate = 1.0 - soft_mask.mean()
            sparsity_loss = (current_masking_rate - self.target_masking_rate) ** 2

            loss = avg_displacement + self.sparsity_lambda * sparsity_loss
            loss.backward()
            self.optimizer.step()

            # Logging
            history["loss"].append(float(loss.detach()))
            history["displacement"].append(float(avg_displacement.detach()))
            history["sparsity"].append(float(sparsity_loss.detach()))
            history["tau"].append(tau)
            history["masking_rate"].append(self.mask_module.get_masking_rate())

            if step % 20 == 0 or step == self.num_steps - 1:
                logger.info(
                    f"  Step {step:>3}/{self.num_steps}: loss={float(loss):.6f}, "
                    f"disp={float(avg_displacement):.6f}, "
                    f"mask_rate={self.mask_module.get_masking_rate():.3f}, "
                    f"tau={tau:.3f}"
                )

        # Remove hooks
        self.model.set_attention_head_mask(None)

        # Produce final binary masks
        logger.info("\nProducing binary masks from learned parameters...")
        masks = self._produce_diverse_masks(num_masks_to_produce)

        metadata = {
            "history": history,
            "final_mask_summary": self.mask_module.get_mask_summary(),
            "alpha_values": self.mask_module.alpha.detach().cpu().numpy().tolist(),
            "num_steps": self.num_steps,
            "lr": self.lr,
            "tau_start": self.tau_start,
            "tau_end": self.tau_end,
            "target_masking_rate": self.target_masking_rate,
        }

        return masks, metadata

    def _produce_diverse_masks(self, num_masks: int) -> List[torch.Tensor]:
        """
        Produce num_masks diverse binary masks from learned alpha.

        Strategy:
          - Mask 0: threshold at 0.5 (deterministic)
          - Masks 1-N: sample from Bernoulli(sigmoid(alpha)) with different seeds
            for diversity, then round
        """
        masks = []
        with torch.no_grad():
            soft = torch.sigmoid(self.mask_module.alpha)

            # Deterministic threshold mask
            masks.append((soft > 0.5).float().cpu())

            # Stochastic masks for diversity
            for i in range(num_masks - 1):
                torch.manual_seed(i + 42)
                sampled = torch.bernoulli(soft)
                # Ensure at least 50% of heads are active per layer
                for layer_idx in range(self.mask_module.num_layers):
                    if sampled[layer_idx].sum() < self.mask_module.num_heads * 0.5:
                        # Re-activate some heads based on highest alpha
                        inactive = (sampled[layer_idx] == 0).nonzero(as_tuple=True)[0]
                        scores = soft[layer_idx][inactive]
                        n_activate = int(self.mask_module.num_heads * 0.5 - sampled[layer_idx].sum())
                        top_k = scores.topk(min(n_activate, len(scores))).indices
                        sampled[layer_idx][inactive[top_k]] = 1.0
                masks.append(sampled.float().cpu())

        logger.info(f"Produced {len(masks)} binary masks from learned parameters")
        for i, m in enumerate(masks):
            rate = float((m == 0).float().mean())
            logger.info(f"  Mask {i}: masking_rate={rate:.3f}")

        return masks
