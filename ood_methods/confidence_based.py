"""
Confidence-based OOD Detection Methods.

Methods that use the model's softmax confidence or logit outputs
to distinguish between ID and OOD data.

Implements: MSP, MaxLogit, Energy, KL Divergence, GradNorm
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MSPScorer:
    """
    Maximum Softmax Probability (MSP).
    Hendrycks & Gimpel, 2016.
    
    Score = max_c softmax(logits)_c
    Higher score → more likely ID.
    OOD data tends to have lower max softmax probability.
    """
    
    def __init__(self):
        self.name = "msp"
    
    def score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute MSP scores.
        
        Args:
            logits: (N, num_classes) raw logit outputs
            
        Returns:
            scores: (N,) MSP scores (higher = more likely ID)
        """
        # Softmax probabilities
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probs = F.softmax(logits_tensor, dim=1).numpy()
        # Max probability per sample
        return probs.max(axis=1)


class MaxLogitScorer:
    """
    Maximum Logit Score.
    
    Score = max_c logits_c
    Higher score → more likely ID.
    """
    
    def __init__(self):
        self.name = "maxLogit"
    
    def score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute MaxLogit scores.
        
        Args:
            logits: (N, num_classes)
            
        Returns:
            scores: (N,) max logit values
        """
        return logits.max(axis=1)


class EnergyScorer:
    """
    Energy-based OOD Detection.
    Liu et al., 2020.
    
    Score = T * log(sum_c exp(logits_c / T))
    Higher energy → more likely ID.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.name = "energy"
        self.temperature = temperature
    
    def score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute energy scores.
        
        Args:
            logits: (N, num_classes)
            
        Returns:
            scores: (N,) energy scores (higher = more likely ID)
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        # E(x) = T * log(sum_c exp(f_c(x) / T))
        energy = self.temperature * torch.logsumexp(
            logits_tensor / self.temperature, dim=1
        )
        return energy.numpy()


class KLDivergenceScorer:
    """
    KL Divergence OOD Score.
    
    Score = KL(softmax(logits) || uniform)
    Higher KL divergence → prediction is more "certain" → more likely ID.
    OOD data tends to produce more uniform predictions.
    """
    
    def __init__(self):
        self.name = "kl"
    
    def score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute KL divergence from uniform distribution.
        
        Args:
            logits: (N, num_classes)
            
        Returns:
            scores: (N,) KL divergence scores (higher = more likely ID)
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probs = F.softmax(logits_tensor, dim=1)
        
        num_classes = logits.shape[1]
        uniform = torch.ones_like(probs) / num_classes
        
        # KL(P || Q) = sum(P * log(P/Q))
        kl = F.kl_div(
            uniform.log(), probs, reduction='none'
        ).sum(dim=1)
        
        return kl.numpy()


class GradNormScorer:
    """
    GradNorm OOD Detection.
    Huang et al., 2021.
    
    Score = ||∇_θ KL(ŷ || u)||
    Gradient norm of KL divergence between predicted distribution
    and uniform distribution, w.r.t. the model's last layer parameters.
    
    Higher gradient norm → more likely ID (model "cares more" about ID data).
    """
    
    def __init__(self):
        self.name = "gradNorm"
    
    def score(
        self,
        model,
        dataloader,
        device: str = "cuda",
        num_classes: int = 9,
    ) -> np.ndarray:
        """
        Compute GradNorm scores. Requires model and data (not just logits).
        
        Args:
            model: LayoutLMv3DocumentClassifier
            dataloader: DataLoader
            device: Compute device
            num_classes: Number of ID classes
            
        Returns:
            scores: (N,) gradient norm scores
        """
        model.eval()
        model.to(device)
        
        all_scores = []
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            # Enable gradients for this forward pass
            model.zero_grad()
            
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
            
            logits = outputs["logits"]
            
            # Process each sample individually for gradient computation
            batch_scores = []
            for i in range(logits.shape[0]):
                model.zero_grad()
                
                sample_logits = logits[i:i+1]  # (1, num_classes)
                probs = F.softmax(sample_logits, dim=1)
                uniform = torch.ones_like(probs) / num_classes
                
                # KL divergence
                kl_loss = F.kl_div(
                    (probs + 1e-8).log(), uniform, reduction='sum'
                )
                
                # Compute gradient w.r.t. last layer (classifier weights)
                kl_loss.backward(retain_graph=True)
                
                # Get gradient norm from classifier layer
                grad_norm = 0.0
                for name, param in model.model.classifier.named_parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                batch_scores.append(grad_norm)
            
            all_scores.extend(batch_scores)
        
        return np.array(all_scores)
