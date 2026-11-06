"""
Sensitivity-based OOD Detection — AHSS (Attention Head Sensitivity Scoring).

Core idea (Representation Fragility Hypothesis):
  - ID data is redundantly encoded across attention heads → masking heads
    barely changes the embedding → LOW AHSS score
  - OOD data is fragilely encoded → masking heads shifts the embedding
    dramatically → HIGH AHSS score

AHSS(x) = (1/T) * Σ_t ||f(x; M_t) - f(x; 1)||²

where f(x; M_t) is the CLS embedding under mask M_t, f(x; 1) is the
unmasked embedding, and T is the number of masks.

Convention: HIGHER AHSS = more OOD. For scoring compatibility with
existing methods (where higher = more ID-like), we return -AHSS.
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class AHSSScorer:
    """
    Attention Head Sensitivity Scorer for OOD detection.

    Measures how much a sample's embedding changes under attention head masking.
    ID samples (redundantly encoded) → low sensitivity → low AHSS.
    OOD samples (fragilely encoded) → high sensitivity → high AHSS.

    API follows the same pattern as MahalanobisScorer/KNNScorer:
    - fit() to store reference (unmasked) embeddings (optional for pure AHSS)
    - score() to compute AHSS scores
    - score_combined() to combine AHSS with distance-based scores
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, L2-normalize embeddings before computing
                       displacement. This makes AHSS scale-invariant.
        """
        self.normalize = normalize
        self._fitted = False

    def fit(
        self,
        unmasked_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Store reference data (optional). Can be used later for normalization
        or combined scoring calibration.

        Args:
            unmasked_embeddings: (N, D) unmasked CLS embeddings for train/eval ID
            labels: (N,) class labels (optional, for future use with NC analysis)
        """
        self.train_unmasked = unmasked_embeddings
        self.train_labels = labels
        self._fitted = True

    @staticmethod
    def compute_ahss(
        unmasked_embeddings: np.ndarray,
        masked_embeddings_list: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute AHSS scores.

        AHSS(x) = (1/T) * Σ_t ||f(x; M_t) - f(x; 1)||²

        Args:
            unmasked_embeddings: (N, D) embeddings with all heads active
            masked_embeddings_list: list of T arrays, each (N, D),
                                    embeddings under mask M_t
            normalize: L2-normalize embeddings before computing displacement

        Returns:
            ahss_scores: (N,) per-sample AHSS scores (higher = more OOD)
        """
        T = len(masked_embeddings_list)
        if T == 0:
            raise ValueError("Need at least one masked embedding set")

        ref = unmasked_embeddings.astype(np.float64)
        if normalize:
            norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
            ref = ref / norms

        N = ref.shape[0]
        ahss = np.zeros(N, dtype=np.float64)

        for masked_embs in masked_embeddings_list:
            m = masked_embs.astype(np.float64)
            if normalize:
                m_norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
                m = m / m_norms
            # ||f(x; M_t) - f(x; 1)||²
            diff = m - ref
            ahss += np.sum(diff ** 2, axis=1)

        ahss /= T
        return ahss

    def score(
        self,
        unmasked_embeddings: np.ndarray,
        masked_embeddings_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute OOD scores based on AHSS.

        Convention: returns -AHSS so that HIGHER score = more ID-like
        (consistent with Mahalanobis/kNN scoring convention).

        Args:
            unmasked_embeddings: (N, D) unmasked embeddings
            masked_embeddings_list: list of T arrays, each (N, D)

        Returns:
            scores: (N,) scores where higher = more ID-like
        """
        ahss = self.compute_ahss(
            unmasked_embeddings, masked_embeddings_list, normalize=self.normalize
        )
        return -ahss

    def score_raw(
        self,
        unmasked_embeddings: np.ndarray,
        masked_embeddings_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute raw AHSS scores (higher = more OOD).
        Useful for analysis and visualization.
        """
        return self.compute_ahss(
            unmasked_embeddings, masked_embeddings_list, normalize=self.normalize
        )

    @staticmethod
    def score_combined(
        ahss_scores: np.ndarray,
        distance_scores: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Combine AHSS with distance-based scores (e.g., Mahalanobis).

        Score(x) = α * distance_score(x) + (1-α) * (-AHSS(x))

        Both inputs should follow the convention: higher = more ID-like.

        Args:
            ahss_scores: (N,) AHSS scores (already negated, higher = more ID)
            distance_scores: (N,) distance-based scores (higher = more ID)
            alpha: weight for distance score (0=pure AHSS, 1=pure distance)

        Returns:
            combined: (N,) combined scores
        """
        # Normalize both to [0,1] range before combining
        def _normalize(s):
            s_min, s_max = s.min(), s.max()
            if s_max - s_min < 1e-10:
                return np.zeros_like(s)
            return (s - s_min) / (s_max - s_min)

        ahss_norm = _normalize(ahss_scores)
        dist_norm = _normalize(distance_scores)
        return alpha * dist_norm + (1.0 - alpha) * ahss_norm


class MCDropoutScorer:
    """
    MC-Dropout baseline for comparison with AHSS.

    Uses stochastic dropout during inference (Monte Carlo Dropout)
    to measure prediction/embedding uncertainty. This serves as
    the primary baseline to differentiate AHSS from.

    Key differences from AHSS:
    - MC-Dropout: random NEURON zeroing (unstructured), measures OUTPUT variance
    - AHSS: structured ATTENTION HEAD masking, measures EMBEDDING displacement
    """

    def __init__(self, n_samples: int = 25, normalize: bool = True):
        """
        Args:
            n_samples: Number of MC forward passes (T)
            normalize: L2-normalize embeddings before computing variance
        """
        self.n_samples = n_samples
        self.normalize = normalize

    @staticmethod
    def compute_mc_uncertainty(
        embedding_samples: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute MC-Dropout uncertainty as mean embedding variance.

        Args:
            embedding_samples: list of T arrays, each (N, D),
                               embeddings from T stochastic forward passes
            normalize: L2-normalize before computing variance

        Returns:
            uncertainty: (N,) per-sample uncertainty (higher = more OOD)
        """
        T = len(embedding_samples)
        if T == 0:
            raise ValueError("Need at least one embedding sample")

        samples = []
        for emb in embedding_samples:
            e = emb.astype(np.float64)
            if normalize:
                norms = np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
                e = e / norms
            samples.append(e)

        # Stack: (T, N, D)
        stacked = np.stack(samples, axis=0)
        # Variance across T passes, then sum across D dimensions
        var_per_dim = np.var(stacked, axis=0)  # (N, D)
        uncertainty = np.sum(var_per_dim, axis=1)  # (N,)

        return uncertainty

    def score(self, embedding_samples: List[np.ndarray]) -> np.ndarray:
        """
        Compute OOD scores. Returns -uncertainty (higher = more ID-like).
        """
        uncertainty = self.compute_mc_uncertainty(
            embedding_samples, normalize=self.normalize
        )
        return -uncertainty
