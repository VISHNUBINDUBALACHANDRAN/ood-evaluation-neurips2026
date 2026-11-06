"""
Feature/Logits-based OOD Detection Methods.

Methods that operate on intermediate feature representations
or logit outputs to distinguish between ID and OOD data.

Implements: VIM, Residual, NECO, ReAct, ASH
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from sklearn.covariance import EmpiricalCovariance
import logging

logger = logging.getLogger(__name__)


class VIMScorer:
    """
    Virtual-logit Matching (VIM).
    Wang et al., 2022.
    
    Computes the principal subspace of training features, then
    for test data: score = energy of (logits + virtual_logit)
    where virtual_logit = norm of residual in null space.
    """
    
    def __init__(self, principal_dim: int = 384):
        """
        Args:
            principal_dim: Number of principal components to keep
                          (paper uses hidden_size // 2 = 384)
        """
        self.name = "vim"
        self.principal_dim = principal_dim
        self.u = None  # Principal directions
        self.train_mean = None
        self.alpha = None  # Scaling factor
    
    def fit(
        self,
        train_embeddings: np.ndarray,
        train_logits: np.ndarray,
    ):
        """
        Fit VIM on training data.
        
        Args:
            train_embeddings: (N_train, hidden_size) CLS embeddings
            train_logits: (N_train, num_classes) logit outputs
        """
        self.train_mean = train_embeddings.mean(axis=0)
        centered = train_embeddings - self.train_mean
        
        # SVD to find principal subspace
        # We keep the NULL space (complement of principal subspace)
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Principal directions (top-D)
        self.u = Vt[:self.principal_dim].T  # (hidden_size, principal_dim)
        
        # Null space directions
        self.null_space = Vt[self.principal_dim:].T  # (hidden_size, null_dim)
        
        # Compute alpha: ratio that calibrates virtual logit scale
        # alpha = norm of logits / norm of residuals on training data
        train_residuals = self._compute_residual(train_embeddings)
        train_vim_norms = np.linalg.norm(train_residuals, axis=1)
        train_energy = self._logsumexp(train_logits)
        
        self.alpha = train_energy.mean() / (train_vim_norms.mean() + 1e-8)
        
        logger.info(f"VIM fitted: principal_dim={self.principal_dim}, alpha={self.alpha:.4f}")
    
    def _compute_residual(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings into null space."""
        centered = embeddings - self.train_mean
        # Project onto null space
        residual = centered @ self.null_space  # (N, null_dim)
        return residual
    
    def _logsumexp(self, logits: np.ndarray) -> np.ndarray:
        """Compute logsumexp of logits."""
        return np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1)) + logits.max(axis=1)
    
    def score(
        self,
        test_embeddings: np.ndarray,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        """
        Compute VIM scores.
        
        Args:
            test_embeddings: (N, hidden_size)
            test_logits: (N, num_classes)
            
        Returns:
            scores: (N,) VIM scores (higher = more likely ID)
        """
        # Residual norm (virtual logit)
        residuals = self._compute_residual(test_embeddings)
        virtual_logit = self.alpha * np.linalg.norm(residuals, axis=1, keepdims=True)
        
        # Augmented logits = original logits + virtual logit
        augmented_logits = np.concatenate([test_logits, virtual_logit], axis=1)
        
        # Score = negative energy of augmented logits
        # OOD data has larger residual → larger virtual logit → higher energy
        # Negate so that higher score = more likely ID (convention)
        scores = -self._logsumexp(augmented_logits)
        
        return scores


class ResidualScorer:
    """
    Residual Score for OOD Detection.
    
    Score = norm of the residual after projecting features onto
    the principal subspace of training features.
    Lower residual norm → more likely ID.
    We negate for consistency (higher = more likely ID).
    """
    
    def __init__(self, principal_dim: int = 384):
        self.name = "residual"
        self.principal_dim = principal_dim
        self.u = None
        self.train_mean = None
    
    def fit(self, train_embeddings: np.ndarray):
        """Fit principal subspace on training embeddings."""
        self.train_mean = train_embeddings.mean(axis=0)
        centered = train_embeddings - self.train_mean
        
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.u = Vt[:self.principal_dim].T  # (hidden_size, principal_dim)
        
        logger.info(f"Residual scorer fitted: principal_dim={self.principal_dim}")
    
    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute residual scores.
        
        Returns:
            scores: (N,) negative residual norms (higher = more likely ID)
        """
        centered = test_embeddings - self.train_mean
        
        # Project onto principal subspace
        projection = centered @ self.u @ self.u.T  # (N, hidden_size)
        
        # Residual = original - projection
        residual = centered - projection
        
        # Norm of residual (negate: lower residual = more ID-like)
        scores = -np.linalg.norm(residual, axis=1)
        
        return scores


class NECOScorer:
    """
    Neural Collapse OOD Detection (NECO).
    Ammar et al., 2024.
    
    Leverages neural collapse properties: in well-trained networks,
    class features collapse to simplex equiangular tight frame (ETF) vertices.
    OOD data deviates from this geometry.
    
    Score = alignment with nearest class prototype in ETF space.
    """
    
    def __init__(self):
        self.name = "neco"
        self.class_means = None
        self.global_mean = None
    
    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        """
        Fit NECO by computing class means (prototypes).
        
        Args:
            train_embeddings: (N_train, hidden_size)
            train_labels: (N_train,) integer labels
        """
        classes = np.unique(train_labels)
        self.global_mean = train_embeddings.mean(axis=0)
        
        self.class_means = {}
        for c in classes:
            mask = train_labels == c
            self.class_means[c] = train_embeddings[mask].mean(axis=0)
        
        logger.info(f"NECO fitted with {len(classes)} classes")
    
    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute NECO scores.
        
        Score is based on how well each test sample aligns with the
        neural collapse structure (nearest class mean, after centering).
        
        Returns:
            scores: (N,) NECO scores (higher = more likely ID)
        """
        # Center embeddings
        centered = test_embeddings - self.global_mean
        centered_norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        centered_normalized = centered / centered_norms
        
        # Center class means
        class_means_array = np.array(list(self.class_means.values()))
        centered_means = class_means_array - self.global_mean
        mean_norms = np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-8
        centered_means_normalized = centered_means / mean_norms
        
        # Cosine similarity to nearest class mean
        sim_matrix = centered_normalized @ centered_means_normalized.T  # (N, C)
        
        # Max similarity = alignment with nearest class prototype
        scores = sim_matrix.max(axis=1)
        
        return scores


class ReActScorer:
    """
    ReAct: Rectified Activations for OOD Detection.
    Sun et al., 2021.
    
    Clips/truncates activations at a threshold before the classifier head,
    then re-computes logits → energy score.  This reduces the impact of noisy
    activations that cause overconfident OOD predictions.
    
    When classifier_weight / classifier_bias are provided the clipped
    features are properly re-forwarded through the linear head.  Otherwise
    we fall back to a proportional scaling approximation.
    """
    
    def __init__(self, threshold_percentile: float = 90.0, temperature: float = 1.0):
        self.name = "react"
        self.threshold_percentile = threshold_percentile
        self.temperature = temperature
        self.threshold = None
    
    def fit(self, train_embeddings: np.ndarray):
        """Fit threshold from training data activations."""
        self.threshold = np.percentile(train_embeddings, self.threshold_percentile)
        logger.info(f"ReAct threshold: {self.threshold:.4f} (p{self.threshold_percentile})")
    
    def score(
        self,
        test_embeddings: np.ndarray,
        test_logits: np.ndarray,
        classifier_weight: Optional[np.ndarray] = None,
        classifier_bias: Optional[np.ndarray] = None,
        dense_weight: Optional[np.ndarray] = None,
        dense_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ReAct scores.
        
        If dense_weight is supplied, the full classifier head is applied
        (dense + tanh + out_proj).  If only classifier_weight is supplied,
        falls back to single-layer forwarding.
        Otherwise falls back to proportional logit scaling.
        
        Returns:
            scores: (N,) energy scores after activation rectification
        """
        clipped = np.clip(test_embeddings, a_min=None, a_max=self.threshold)
        
        if dense_weight is not None and classifier_weight is not None:
            # Full classifier head: dense + tanh + out_proj
            dense_out = clipped @ dense_weight.T
            if dense_bias is not None:
                dense_out = dense_out + dense_bias
            dense_out = np.tanh(dense_out)
            new_logits = dense_out @ classifier_weight.T
            if classifier_bias is not None:
                new_logits = new_logits + classifier_bias
        elif classifier_weight is not None:
            # Single-layer fallback (out_proj only)
            new_logits = clipped @ classifier_weight.T
            if classifier_bias is not None:
                new_logits = new_logits + classifier_bias
        else:
            # Approximation fallback
            original_norms = np.linalg.norm(test_embeddings, axis=1) + 1e-8
            clipped_norms = np.linalg.norm(clipped, axis=1) + 1e-8
            scale = clipped_norms / original_norms
            new_logits = test_logits * scale[:, np.newaxis]
        
        logits_tensor = torch.tensor(new_logits, dtype=torch.float32)
        energy = self.temperature * torch.logsumexp(
            logits_tensor / self.temperature, dim=1
        )
        return energy.numpy()


class ASHScorer:
    """
    Activation Shaping (ASH) for OOD Detection.
    Djurisic et al., 2023.
    
    "Extremely Simple Activation Shaping" — prunes (zeros out) a
    percentage of the smallest activations, then re-computes logits
    through the classifier head → energy score.
    
    When classifier_weight / classifier_bias are provided the shaped
    features are properly re-forwarded.  Otherwise falls back to
    proportional scaling.
    """
    
    def __init__(self, pruning_percentile: float = 90.0, temperature: float = 1.0):
        self.name = "ash"
        self.pruning_percentile = pruning_percentile
        self.temperature = temperature
    
    def score(
        self,
        test_embeddings: np.ndarray,
        test_logits: np.ndarray,
        classifier_weight: Optional[np.ndarray] = None,
        classifier_bias: Optional[np.ndarray] = None,
        dense_weight: Optional[np.ndarray] = None,
        dense_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ASH scores.
        
        Returns:
            scores: (N,) energy scores after activation shaping
        """
        shaped = test_embeddings.copy()
        for i in range(shaped.shape[0]):
            threshold = np.percentile(np.abs(shaped[i]), self.pruning_percentile)
            shaped[i][np.abs(shaped[i]) < threshold] = 0.0
        
        if dense_weight is not None and classifier_weight is not None:
            # Full classifier head: dense + tanh + out_proj
            dense_out = shaped @ dense_weight.T
            if dense_bias is not None:
                dense_out = dense_out + dense_bias
            dense_out = np.tanh(dense_out)
            new_logits = dense_out @ classifier_weight.T
            if classifier_bias is not None:
                new_logits = new_logits + classifier_bias
        elif classifier_weight is not None:
            # Single-layer fallback (out_proj only)
            new_logits = shaped @ classifier_weight.T
            if classifier_bias is not None:
                new_logits = new_logits + classifier_bias
        else:
            # Approximation fallback — scale logits proportionally
            original_sum = np.abs(test_embeddings).sum(axis=1, keepdims=True) + 1e-8
            shaped_sum = np.abs(shaped).sum(axis=1, keepdims=True) + 1e-8
            scale = original_sum / shaped_sum
            new_logits = test_logits * scale.flatten()[:, np.newaxis]
        
        logits_tensor = torch.tensor(new_logits, dtype=torch.float32)
        energy = self.temperature * torch.logsumexp(
            logits_tensor / self.temperature, dim=1
        )
        return energy.numpy()
