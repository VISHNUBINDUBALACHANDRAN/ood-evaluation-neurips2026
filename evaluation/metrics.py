"""
Evaluation Metrics for OOD Detection.

Implements: AUROC, FPR@95TPR, and aggregation utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logger = logging.getLogger(__name__)


def compute_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve (AUROC).
    
    Convention: ID samples labeled as 1 (positive), OOD as 0 (negative).
    Higher scores should correspond to ID data.
    
    Args:
        id_scores: (N_id,) OOD detection scores for ID data
        ood_scores: (N_ood,) OOD detection scores for OOD data
        
    Returns:
        auroc: float in [0, 1], higher is better
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        logger.warning("AUROC computation failed (possibly constant predictions)")
        auroc = 0.5
    
    return float(auroc)


def compute_fpr_at_tpr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """
    Compute False Positive Rate at a given True Positive Rate threshold.
    
    FPR@95TPR: The false positive rate when the true positive rate is 95%.
    Lower is better — means fewer OOD samples are incorrectly classified as ID
    when 95% of actual ID samples are correctly identified.
    
    Args:
        id_scores: (N_id,) scores for ID data (higher = more ID-like)
        ood_scores: (N_ood,) scores for OOD data  
        tpr_target: Target TPR (default 0.95 as in the paper)
        
    Returns:
        fpr: float in [0, 1], lower is better
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    
    try:
        fpr_values, tpr_values, thresholds = roc_curve(labels, scores)
        
        # Find the FPR at the threshold closest to the target TPR
        # TPR values are in increasing order
        idx = np.searchsorted(tpr_values, tpr_target, side='left')
        idx = min(idx, len(fpr_values) - 1)
        
        # Interpolate if needed
        if idx > 0 and tpr_values[idx] != tpr_target:
            # Linear interpolation
            tpr_low = tpr_values[idx - 1]
            tpr_high = tpr_values[idx]
            fpr_low = fpr_values[idx - 1]
            fpr_high = fpr_values[idx]
            
            if tpr_high - tpr_low > 0:
                alpha = (tpr_target - tpr_low) / (tpr_high - tpr_low)
                fpr = fpr_low + alpha * (fpr_high - fpr_low)
            else:
                fpr = fpr_values[idx]
        else:
            fpr = fpr_values[idx]
            
    except ValueError:
        logger.warning("FPR computation failed")
        fpr = 1.0
    
    return float(fpr)


def evaluate_ood_method(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "",
) -> Dict[str, float]:
    """
    Evaluate a single OOD method, computing both AUROC and FPR@95.
    
    Args:
        id_scores: (N_id,) scores for ID data
        ood_scores: (N_ood,) scores for OOD data
        method_name: Name of the method (for logging)
        
    Returns:
        dict with 'auroc' and 'fpr' keys
    """
    auroc = compute_auroc(id_scores, ood_scores)
    fpr = compute_fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95)
    
    logger.info(f"{method_name}: AUROC={auroc:.4f}, FPR@95={fpr:.4f}")
    
    return {
        "method": method_name,
        "auroc": auroc,
        "fpr": fpr,
    }


def aggregate_results(
    all_run_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple runs (seeds).
    Compute mean and standard deviation for each method/metric.
    
    Args:
        all_run_results: List of dicts, each from evaluate_ood_method()
                        across multiple seeds. Format:
                        [
                            {"method": "msp", "auroc": 0.95, "fpr": 0.12},
                            {"method": "msp", "auroc": 0.93, "fpr": 0.14},
                            ...
                        ]
                        
    Returns:
        aggregated: {
            "msp": {"auroc_mean": 0.94, "auroc_std": 0.01, "fpr_mean": 0.13, "fpr_std": 0.01},
            ...
        }
    """
    from collections import defaultdict
    
    # Group by method
    method_results = defaultdict(lambda: {"auroc": [], "fpr": []})
    
    for result in all_run_results:
        method = result["method"]
        method_results[method]["auroc"].append(result["auroc"])
        method_results[method]["fpr"].append(result["fpr"])
    
    # Compute statistics
    aggregated = {}
    for method, metrics in method_results.items():
        aggregated[method] = {
            "auroc_mean": float(np.mean(metrics["auroc"])),
            "auroc_std": float(np.std(metrics["auroc"])),
            "fpr_mean": float(np.mean(metrics["fpr"])),
            "fpr_std": float(np.std(metrics["fpr"])),
        }
    
    return aggregated


def format_results_table(
    aggregated_results: Dict[str, Dict[str, float]],
    dataset_name: str = "",
    experiment_type: str = "",
) -> str:
    """
    Format aggregated results as a table string matching Table 1 from the paper.
    
    Args:
        aggregated_results: Output from aggregate_results()
        dataset_name: Dataset name for header
        experiment_type: "intra" or "cross" for header
        
    Returns:
        Formatted table string
    """
    header = f"\n{'='*60}"
    header += f"\n{dataset_name} ({experiment_type}-dataset OOD)"
    header += f"\n{'='*60}"
    header += f"\n{'Method':<20} {'AUROC':>12} {'FPR@95':>12}"
    header += f"\n{'-'*44}"
    
    rows = []
    for method, metrics in sorted(aggregated_results.items()):
        auroc_str = f"{metrics['auroc_mean']:.3f}±{metrics['auroc_std']:.3f}"
        fpr_str = f"{metrics['fpr_mean']:.3f}±{metrics['fpr_std']:.3f}"
        rows.append(f"{method:<20} {auroc_str:>12} {fpr_str:>12}")
    
    return header + "\n" + "\n".join(rows) + "\n"


def compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for checkpoint selection.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
        a(i) = mean intra-cluster distance for sample i
        b(i) = mean nearest-cluster distance for sample i
    
    Higher score → better intra-class compactness and inter-class separation.
    Used to filter out bad checkpoints during training.
    
    Args:
        embeddings: (N, hidden_size) feature embeddings
        labels: (N,) class labels
        
    Returns:
        score: float, silhouette score (range [-1, 1])
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        logger.warning("Less than 2 classes for silhouette score, returning 0")
        return 0.0
    
    if len(embeddings) < len(unique_labels):
        logger.warning("Too few samples for silhouette score, returning 0")
        return 0.0
    
    try:
        score = sklearn_silhouette(embeddings, labels, metric="euclidean")
    except ValueError as e:
        logger.warning(f"Silhouette score computation failed: {e}")
        score = 0.0
    
    return float(score)
