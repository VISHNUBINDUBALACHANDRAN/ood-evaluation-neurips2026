"""
Distance/Density-based OOD Detection Methods.

Methods that use distance metrics in the embedding space to
distinguish between ID and OOD data.

Implements: Mahalanobis Distance, kNN / kNN+
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class MahalanobisScorer:
    """
    Mahalanobis Distance-based OOD Detection.
    Lee et al., 2018.
    
    Fits class-conditional Gaussians N(μ_c, Σ) on training embeddings
    with a shared covariance matrix Σ.
    
    Score = -min_c (x - μ_c)^T Σ^{-1} (x - μ_c)
    Higher score (less negative) → more likely ID.
    
    This is the primary scoring method used with AHM in the paper.
    """
    
    def __init__(self):
        self.name = "Mahalanobis"
        self.class_means = {}       # {class_id: mean_vector}
        self.precision_matrix = None  # Σ^{-1}
        self.shared_cov = None       # Σ
    
    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        """
        Fit class-conditional Gaussians with shared covariance.
        
        Args:
            train_embeddings: (N_train, hidden_size) CLS embeddings
            train_labels: (N_train,) integer class labels
        """
        classes = np.unique(train_labels)
        hidden_size = train_embeddings.shape[1]
        
        # Compute class means
        self.class_means = {}
        for c in classes:
            mask = train_labels == c
            self.class_means[c] = train_embeddings[mask].mean(axis=0)
        
        # Compute shared covariance matrix
        # Σ = (1/N) Σ_c Σ_{x ∈ c} (x - μ_c)(x - μ_c)^T
        centered_data = np.zeros_like(train_embeddings)
        for c in classes:
            mask = train_labels == c
            centered_data[mask] = train_embeddings[mask] - self.class_means[c]
        
        # Use shrunk covariance for numerical stability
        cov_estimator = ShrunkCovariance(shrinkage=0.01)
        cov_estimator.fit(centered_data)
        self.shared_cov = cov_estimator.covariance_
        self.precision_matrix = cov_estimator.precision_
        
        logger.info(f"Mahalanobis fitted: {len(classes)} classes, "
                    f"feature dim={hidden_size}")
    
    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance scores.
        
        Score = -min_c (x - μ_c)^T Σ^{-1} (x - μ_c)
        
        Args:
            test_embeddings: (N, hidden_size)
            
        Returns:
            scores: (N,) negative Mahalanobis distances (higher = more likely ID)
        """
        N = test_embeddings.shape[0]
        num_classes = len(self.class_means)
        
        # Compute Mahalanobis distance to each class
        mah_distances = np.zeros((N, num_classes))
        
        for idx, (c, mean) in enumerate(self.class_means.items()):
            diff = test_embeddings - mean  # (N, hidden_size)
            # (x - μ)^T Σ^{-1} (x - μ) = diff @ precision @ diff.T diagonal
            mah_distances[:, idx] = np.sum(
                diff @ self.precision_matrix * diff, axis=1
            )
        
        # Score = negative minimum distance (higher = closer to some class = more ID)
        scores = -mah_distances.min(axis=1)
        
        return scores
    
    def get_class_distances(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Get per-class Mahalanobis distances. Shape: (N, num_classes)."""
        N = test_embeddings.shape[0]
        num_classes = len(self.class_means)
        
        distances = np.zeros((N, num_classes))
        for idx, (c, mean) in enumerate(self.class_means.items()):
            diff = test_embeddings - mean
            distances[:, idx] = np.sum(
                diff @ self.precision_matrix * diff, axis=1
            )
        
        return distances


class KNNScorer:
    """
    k-Nearest Neighbor OOD Detection (kNN / kNN+).
    Sun et al., 2022.
    
    Score = -distance to the K-th nearest neighbor in training set.
    Higher score (less negative distance) → more likely ID.
    
    Simple and effective: no distributional assumptions needed.
    """
    
    def __init__(self, k: int = 50, metric: str = "euclidean"):
        """
        Args:
            k: Number of nearest neighbors
            metric: Distance metric ("euclidean", "cosine")
        """
        self.name = "knn"
        self.k = k
        self.metric = metric
        self.train_embeddings = None
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Store training embeddings as the reference set.
        
        Args:
            train_embeddings: (N_train, hidden_size)
        """
        self.train_embeddings = train_embeddings
        
        if self.metric == "cosine":
            # Normalize for cosine distance
            norms = np.linalg.norm(self.train_embeddings, axis=1, keepdims=True) + 1e-8
            self.train_embeddings_normalized = self.train_embeddings / norms
        
        logger.info(f"kNN fitted: N_train={train_embeddings.shape[0]}, k={self.k}")
    
    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute kNN OOD scores.
        
        Score = -distance to K-th nearest neighbor.
        
        Args:
            test_embeddings: (N, hidden_size)
            
        Returns:
            scores: (N,) negative K-th NN distances (higher = more likely ID)
        """
        if self.metric == "cosine":
            norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8
            test_normalized = test_embeddings / norms
            # Cosine distance = 1 - cosine_similarity
            distances = cdist(test_normalized, self.train_embeddings_normalized, metric="cosine")
        else:
            distances = cdist(test_embeddings, self.train_embeddings, metric=self.metric)
        
        # Sort distances, take K-th nearest
        sorted_distances = np.sort(distances, axis=1)
        k = min(self.k, sorted_distances.shape[1] - 1)
        kth_distances = sorted_distances[:, k]
        
        # Negate: closer = higher score = more ID
        scores = -kth_distances
        
        return scores
    
    def score_avg_k(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Score using average distance to top-K neighbors (kNN+ variant).
        
        Returns:
            scores: (N,) negative average K-NN distances
        """
        if self.metric == "cosine":
            norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8
            test_normalized = test_embeddings / norms
            distances = cdist(test_normalized, self.train_embeddings_normalized, metric="cosine")
        else:
            distances = cdist(test_embeddings, self.train_embeddings, metric=self.metric)
        
        sorted_distances = np.sort(distances, axis=1)
        k = min(self.k, sorted_distances.shape[1])
        avg_k_distances = sorted_distances[:, :k].mean(axis=1)
        
        return -avg_k_distances
