"""
Embedding Aggregation Methods for OOD Detection.

Methods that modify how embeddings are extracted/aggregated
to improve OOD detection performance.

Implements: AvgAvg (Chen et al., 2022), Gnome (Chen et al., 2023)
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AvgAvgEmbedding:
    """
    AvgAvg Embedding Aggregation.
    Chen et al., 2022 - "Holistic Sentence Embeddings for Better OOD Detection"
    
    Instead of using only the last layer's CLS token:
    1. Extract CLS token from ALL layers of the fine-tuned model
    2. Average across sequence length (already done since we use CLS)
    3. Average across all layers
    
    This produces embeddings that capture information from multiple
    levels of abstraction, improving OOD detection.
    """
    
    def __init__(self):
        self.name = "avgavg"
    
    def aggregate(
        self,
        all_layer_embeddings: List[np.ndarray],
    ) -> np.ndarray:
        """
        Aggregate multi-layer CLS embeddings via averaging.
        
        Args:
            all_layer_embeddings: List of (N, hidden_size) arrays,
                                  one per transformer layer (including embedding layer)
                                  
        Returns:
            aggregated: (N, hidden_size) averaged embeddings
        """
        # Stack: (num_layers, N, hidden_size)
        stacked = np.stack(all_layer_embeddings, axis=0)
        
        # Average across layers
        aggregated = stacked.mean(axis=0)  # (N, hidden_size)
        
        logger.info(f"AvgAvg: aggregated {len(all_layer_embeddings)} layers → "
                    f"shape {aggregated.shape}")
        
        return aggregated
    
    def aggregate_selective(
        self,
        all_layer_embeddings: List[np.ndarray],
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Aggregate selected layers only.
        
        Args:
            all_layer_embeddings: List of (N, hidden_size) arrays
            layer_indices: Which layers to include (None = all)
            
        Returns:
            aggregated: (N, hidden_size)
        """
        if layer_indices is not None:
            selected = [all_layer_embeddings[i] for i in layer_indices]
        else:
            selected = all_layer_embeddings
        
        return self.aggregate(selected)


class GnomeEmbedding:
    """
    Gnome Embedding Method.
    Chen et al., 2023 - "Fine-tuning Deteriorates General Textual 
    Out-of-Distribution Detection by Distorting Task-Agnostic Features"
    
    Key insight: Fine-tuning distorts task-agnostic features that are
    useful for OOD detection. Solution: combine embeddings from BOTH
    the pretrained and fine-tuned models.
    
    Gnome = pretrained_embedding ⊕ finetuned_embedding
    (concatenation or weighted combination)
    """
    
    def __init__(self, combination: str = "concat"):
        """
        Args:
            combination: How to combine pretrained and finetuned embeddings.
                        "concat" - concatenate (doubles dimension)
                        "average" - element-wise average
                        "weighted" - weighted average
        """
        self.name = "gnome"
        self.combination = combination
        self.weight = 0.5  # For weighted combination
    
    def aggregate(
        self,
        pretrained_embeddings: np.ndarray,
        finetuned_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Combine pretrained and finetuned embeddings.
        
        Args:
            pretrained_embeddings: (N, hidden_size) from pretrained model
            finetuned_embeddings: (N, hidden_size) from fine-tuned model
            
        Returns:
            combined: (N, combined_size)
        """
        if self.combination == "concat":
            combined = np.concatenate(
                [pretrained_embeddings, finetuned_embeddings], axis=1
            )
        elif self.combination == "average":
            combined = (pretrained_embeddings + finetuned_embeddings) / 2.0
        elif self.combination == "weighted":
            combined = (
                self.weight * pretrained_embeddings +
                (1 - self.weight) * finetuned_embeddings
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")
        
        logger.info(f"Gnome ({self.combination}): pretrained {pretrained_embeddings.shape} + "
                    f"finetuned {finetuned_embeddings.shape} → {combined.shape}")
        
        return combined
