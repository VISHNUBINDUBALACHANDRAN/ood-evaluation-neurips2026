"""
Neural Collapse Metrics for AHSS Analysis (Contribution C3).

Measures the degree of Neural Collapse (NC) in the model's representations
with and without attention head masking. The theoretical connection:
  - Under perfect NC, ID features converge to class means on a simplex (ETF)
  - Masking heads perturbs the feature extractor
  - For ID: collapsed structure makes features STABLE (stay near class means)
  - For OOD: features don't lie on collapsed manifold → perturbation scatters them

NC Metrics (based on Papyan et al. 2020):
  NC1: Within-class variability collapse (traces of within-class covariance)
  NC2: Convergence to simplex ETF (equiangular tight frame)
  NC3: Self-duality (alignment between class means and classifier weights)
  NC4: Simplification to nearest class mean (NCC accuracy)

References:
  - Papyan et al. (2020) "Prevalence of Neural Collapse"
  - Zhu et al. (ICML 2021) "A Geometric Analysis of Neural Collapse"
  - Galanti et al. (ICML 2025) "Neural Collapse measurements"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy.linalg import sqrtm

logger = logging.getLogger(__name__)


def compute_class_means(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-class mean embeddings.

    Returns:
        class_means: (K, D) array of class mean embeddings
        global_mean: (D,) global mean
        K: number of classes
    """
    unique_labels = np.sort(np.unique(labels))
    K = len(unique_labels)
    D = embeddings.shape[1]

    class_means = np.zeros((K, D))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_means[i] = embeddings[mask].mean(axis=0)

    global_mean = embeddings.mean(axis=0)
    return class_means, global_mean, K


def compute_nc1(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    NC1: Within-class variability collapse.

    Measures Tr(Σ_W @ Σ_B^{-1}) / K, where:
      Σ_W = average within-class covariance
      Σ_B = between-class covariance
    
    Lower NC1 → more collapsed (features cluster tightly around class means).

    Returns:
        nc1: float (lower = more collapsed)
    """
    class_means, global_mean, K = compute_class_means(embeddings, labels)
    unique_labels = np.sort(np.unique(labels))
    D = embeddings.shape[1]

    # Within-class covariance
    Sigma_W = np.zeros((D, D))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_embs = embeddings[mask]
        centered = class_embs - class_means[i]
        Sigma_W += centered.T @ centered
    Sigma_W /= len(embeddings)

    # Between-class covariance
    Sigma_B = np.zeros((D, D))
    for i, label in enumerate(unique_labels):
        n_k = (labels == label).sum()
        diff = (class_means[i] - global_mean).reshape(-1, 1)
        Sigma_B += n_k * (diff @ diff.T)
    Sigma_B /= len(embeddings)

    # Regularize Sigma_B for numerical stability
    reg = 1e-6 * np.eye(D)
    Sigma_B_inv = np.linalg.pinv(Sigma_B + reg)

    nc1 = np.trace(Sigma_W @ Sigma_B_inv) / K
    return float(nc1)


def compute_nc2(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    NC2: Convergence to Simplex ETF.

    Measures how close the centered & normalized class means are to forming
    an equiangular tight frame (regular simplex).

    The ideal ETF has normalized means with pairwise cosine = -1/(K-1).

    Returns:
        nc2: float (lower = closer to perfect ETF)
    """
    class_means, global_mean, K = compute_class_means(embeddings, labels)

    # Center class means
    centered_means = class_means - global_mean

    # Normalize
    norms = np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-8
    normalized_means = centered_means / norms

    # Compute cosine similarity matrix
    cos_sim = normalized_means @ normalized_means.T

    # Ideal ETF: diagonal = 1, off-diagonal = -1/(K-1)
    ideal = -np.ones((K, K)) / (K - 1)
    np.fill_diagonal(ideal, 1.0)

    # NC2 = Frobenius norm of difference
    nc2 = np.linalg.norm(cos_sim - ideal, 'fro') / K
    return float(nc2)


def compute_nc3(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: Optional[np.ndarray] = None,
) -> float:
    """
    NC3: Self-duality (alignment between class means and classifier weights).

    In perfect NC, classifier weight W_k ∝ (class_mean_k - global_mean).
    NC3 measures the cosine similarity between these.

    Returns:
        nc3: float (higher = better alignment, closer to 1.0)
    """
    class_means, global_mean, K = compute_class_means(embeddings, labels)
    centered_means = class_means - global_mean

    # Normalize both
    mean_norms = np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-8
    norm_means = centered_means / mean_norms

    weight_norms = np.linalg.norm(classifier_weight, axis=1, keepdims=True) + 1e-8
    norm_weights = classifier_weight / weight_norms

    # Per-class cosine similarity
    cos_sims = np.sum(norm_means * norm_weights, axis=1)
    nc3 = float(np.mean(cos_sims))
    return nc3


def compute_nc4(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    NC4: Nearest Class Mean (NCM) classification accuracy.

    In perfect NC, the NCM classifier achieves the same accuracy as the
    learned linear classifier. High NC4 = features well-collapsed.

    Returns:
        nc4: float (accuracy, higher = more collapsed, max 1.0)
    """
    class_means, _, K = compute_class_means(embeddings, labels)
    unique_labels = np.sort(np.unique(labels))

    # Predict by nearest class mean
    # distances: (N, K)
    dists = np.zeros((len(embeddings), K))
    for i in range(K):
        diff = embeddings - class_means[i]
        dists[:, i] = np.sum(diff ** 2, axis=1)

    predictions = unique_labels[np.argmin(dists, axis=1)]
    accuracy = float(np.mean(predictions == labels))
    return accuracy


def compute_all_nc_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_weight: Optional[np.ndarray] = None,
    classifier_bias: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all four Neural Collapse metrics.

    Args:
        embeddings: (N, D) feature embeddings
        labels: (N,) class labels
        classifier_weight: (K, D) classifier weight matrix (for NC3)
        classifier_bias: (K,) classifier bias (for NC3)

    Returns:
        dict with nc1, nc2, nc3, nc4 values
    """
    metrics = {}

    metrics["nc1"] = compute_nc1(embeddings, labels)
    metrics["nc2"] = compute_nc2(embeddings, labels)
    metrics["nc4"] = compute_nc4(embeddings, labels)

    if classifier_weight is not None:
        metrics["nc3"] = compute_nc3(
            embeddings, labels, classifier_weight, classifier_bias
        )
    else:
        metrics["nc3"] = None

    return metrics


def compute_nc_under_masks(
    unmasked_embeddings: np.ndarray,
    masked_embeddings_list: List[np.ndarray],
    labels: np.ndarray,
    classifier_weight: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute NC metrics for unmasked and each masked configuration.
    This measures how masking affects Neural Collapse degree.

    Hypothesis: masks that INCREASE NC degree (tighter clusters) are the
    same masks that reduce AHSS for ID data.

    Returns:
        dict with unmasked NC metrics and per-mask NC metrics
    """
    results = {
        "unmasked": compute_all_nc_metrics(
            unmasked_embeddings, labels, classifier_weight
        ),
        "masked": [],
    }

    for i, masked_emb in enumerate(masked_embeddings_list):
        nc = compute_all_nc_metrics(masked_emb, labels, classifier_weight)
        nc["mask_idx"] = i
        results["masked"].append(nc)

    # Compute correlation between NC1 change and AHSS displacement
    if masked_embeddings_list:
        nc1_changes = []
        ahss_per_mask = []
        ref_nc1 = results["unmasked"]["nc1"]

        for i, masked_emb in enumerate(masked_embeddings_list):
            nc1 = results["masked"][i]["nc1"]
            nc1_changes.append(nc1 - ref_nc1)

            disp = np.mean(np.sum((masked_emb - unmasked_embeddings) ** 2, axis=1))
            ahss_per_mask.append(disp)

        if len(nc1_changes) > 2:
            corr = np.corrcoef(nc1_changes, ahss_per_mask)[0, 1]
            results["nc1_ahss_correlation"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            results["nc1_ahss_correlation"] = None

    return results
