"""
Attention Head Masking (AHM) - Core Implementation.
Implements Algorithm 1 from the paper:
"Out-of-Distribution Detection with Attention Head Masking 
 for Multimodal Document Classification"

The AHM technique selectively masks attention heads within different
layers of the encoder to generate embeddings that enhance the separation
between ID and OOD data.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import os
import json
import logging

logger = logging.getLogger(__name__)


class AttentionHeadMasker:
    """
    Manages attention head mask matrices for transformer models.
    
    An AHM matrix is an N×H binary matrix where:
      - N = number of transformer layers
      - H = number of attention heads per layer
      - A[i,j] = 1 means head j in layer i is active
      - A[i,j] = 0 means head j in layer i is masked (zeroed out)
    
    When applied, masked heads produce zero attention scores, effectively
    inhibiting the propagation of their information through the value matrix.
    """
    
    def __init__(self, num_layers: int = 12, num_heads: int = 12):
        self.num_layers = num_layers
        self.num_heads = num_heads
    
    def create_mask(self, masking_percentage: float, seed: Optional[int] = None) -> torch.Tensor:
        """
        Create a random AHM binary mask matrix.
        
        Algorithm 1, Steps 5-8:
        1. Initialize N×H matrix A with all ones
        2. Randomly set p% of elements per layer to zero
        
        Args:
            masking_percentage: Fraction of heads to mask per layer (0.0 to 1.0)
            seed: Optional random seed for reproducibility
            
        Returns:
            Binary mask tensor of shape (num_layers, num_heads)
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Initialize all ones
        mask = np.ones((self.num_layers, self.num_heads), dtype=np.float32)
        
        # Number of heads to mask per layer
        num_to_mask = max(1, int(self.num_heads * masking_percentage))
        
        # Randomly mask heads in each layer
        for layer_idx in range(self.num_layers):
            heads_to_mask = rng.choice(
                self.num_heads, size=num_to_mask, replace=False
            )
            mask[layer_idx, heads_to_mask] = 0.0
        
        return torch.tensor(mask, dtype=torch.float32)
    
    def create_full_mask(self) -> torch.Tensor:
        """Create an all-ones mask (no masking)."""
        return torch.ones(self.num_layers, self.num_heads, dtype=torch.float32)
    
    def save_mask(self, mask: torch.Tensor, path: str):
        """Save mask to file."""
        np.save(path, mask.cpu().numpy())
    
    def load_mask(self, path: str) -> torch.Tensor:
        """Load mask from file."""
        return torch.tensor(np.load(path), dtype=torch.float32)
    
    def get_masking_ratio(self, mask: torch.Tensor) -> float:
        """Get the actual masking ratio of a mask."""
        return 1.0 - mask.mean().item()


class AHMOptimizer:
    """
    Implements Algorithm 1: Optimization of Transformer-based Model 
    using Attention Head Masking for OOD Detection.
    
    The optimizer:
    1. Generates T random AHM masks with masking percentage p
    2. For each mask, extracts embeddings from train and eval (ID) data
    3. Computes similarity scores (top-K nearest neighbor similarity)
    4. Ranks masks by similarity score
    5. Selects top-F masks
    6. Ensembles (averages) embeddings from the top-F masks
    
    The resulting embeddings are used with distance-based OOD detection
    methods (Mahalanobis, kNN) for improved ID/OOD separation.
    """
    
    def __init__(
        self,
        model,
        num_layers: int = 12,
        num_heads: int = 12,
        exploration_budget: int = 25,
        masking_percentages: List[float] = None,
        neighbors_K: int = 10,
        top_F: int = 5,
        device: str = "cuda",
    ):
        """
        Args:
            model: LayoutLMv3DocumentClassifier instance
            num_layers: Number of transformer layers (12 for LayoutLMv3-base)
            num_heads: Number of attention heads per layer (12 for LayoutLMv3-base)
            exploration_budget: T - number of random masks to try
            masking_percentages: p - list of masking percentages to try
            neighbors_K: K - number of nearest neighbors for similarity
            top_F: F - number of top masks to select for ensemble
            device: Compute device
        """
        self.model = model
        self.masker = AttentionHeadMasker(num_layers, num_heads)
        self.exploration_budget = exploration_budget
        self.masking_percentages = masking_percentages or [0.1, 0.2]
        self.neighbors_K = neighbors_K
        self.top_F = top_F
        self.device = device
        
        # Results storage
        self.mask_results = []  # List of (mask, score, masking_pct)
        self.best_masks = []   # Top-F masks after optimization
    
    def _extract_embeddings_with_mask(
        self,
        dataloader: torch.utils.data.DataLoader,
        mask: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract CLS embeddings from model with a specific AHM mask applied.
        
        Algorithm 1, Steps 9-10:
        Apply mask → forward pass → extract dense hidden representations.
        
        Args:
            dataloader: DataLoader for the dataset
            mask: AHM mask tensor (num_layers, num_heads)
            
        Returns:
            Embeddings array of shape (N, hidden_size)
        """
        self.model.eval()
        self.model.to(self.device)
        self.model.set_attention_head_mask(mask)
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                embeddings = self.model.extract_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    layer="last",
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Remove mask after extraction
        self.model.set_attention_head_mask(None)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _compute_similarity_score(
        self,
        train_embeddings: np.ndarray,
        eval_embeddings: np.ndarray,
    ) -> float:
        """
        Compute average similarity score between eval and train embeddings.
        
        Algorithm 1, Steps 11-12:
        For each eval embedding, find top-K nearest neighbors in train set,
        compute cosine similarity, and average across all eval points.
        
        Higher score = better mask (eval data is more similar to train data,
        indicating the mask preserves ID-relevant features).
        
        Args:
            train_embeddings: (N_train, hidden_size)
            eval_embeddings: (N_eval, hidden_size)
            
        Returns:
            float: Average top-K cosine similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Normalize embeddings
        train_norm = train_embeddings / (
            np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8
        )
        eval_norm = eval_embeddings / (
            np.linalg.norm(eval_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute cosine similarity matrix (N_eval, N_train)
        sim_matrix = cosine_similarity(eval_norm, train_norm)
        
        # For each eval point, get top-K similarities
        K = min(self.neighbors_K, sim_matrix.shape[1])
        top_k_sims = np.sort(sim_matrix, axis=1)[:, -K:]  # (N_eval, K)
        
        # Average similarity across all eval points and their top-K neighbors
        avg_similarity = top_k_sims.mean()
        
        return float(avg_similarity)
    
    def optimize(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
        """
        Run Algorithm 1: AHM mask optimization.
        
        Args:
            train_dataloader: DataLoader for ID training data
            eval_dataloader: DataLoader for ID evaluation data
            verbose: Print progress
            
        Returns:
            best_masks: List of top-F mask tensors
            train_ensemble_embeddings: Ensembled train embeddings (N_train, hidden_size)
            eval_ensemble_embeddings: Ensembled eval embeddings (N_eval, hidden_size)
        """
        logger.info(f"Starting AHM optimization: T={self.exploration_budget}, "
                    f"p={self.masking_percentages}, K={self.neighbors_K}, F={self.top_F}")
        
        self.mask_results = []
        trial_idx = 0
        
        # Step 4: For each trial
        # Distribute budget evenly, with remainder going to first percentages
        num_pcts = len(self.masking_percentages)
        base_trials = self.exploration_budget // num_pcts
        remainder = self.exploration_budget % num_pcts
        
        for p_idx, p in enumerate(self.masking_percentages):
            trials_per_p = base_trials + (1 if p_idx < remainder else 0)
            
            iterator = range(trials_per_p)
            if verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"AHM Optimization (p={p})",
                    total=trials_per_p,
                )
            
            for trial in iterator:
                # Steps 5-8: Create random mask
                mask = self.masker.create_mask(masking_percentage=p)
                
                # Steps 9-10: Extract embeddings with this mask
                train_embs = self._extract_embeddings_with_mask(
                    train_dataloader, mask
                )
                eval_embs = self._extract_embeddings_with_mask(
                    eval_dataloader, mask
                )
                
                # Steps 11-12: Compute similarity score
                score = self._compute_similarity_score(train_embs, eval_embs)
                
                self.mask_results.append({
                    "mask": mask,
                    "score": score,
                    "masking_pct": p,
                    "trial_idx": trial_idx,
                    "train_embeddings": train_embs,
                    "eval_embeddings": eval_embs,
                })
                
                if verbose:
                    iterator.set_postfix({"score": f"{score:.4f}"})
                
                trial_idx += 1
        
        # Step 13: Rank masks by similarity score (descending)
        self.mask_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 14: Select top-F masks
        self.best_masks = [r["mask"] for r in self.mask_results[:self.top_F]]
        
        logger.info(f"Top {self.top_F} mask scores: "
                    f"{[r['score'] for r in self.mask_results[:self.top_F]]}")
        
        # Step 15: Ensemble (average) embeddings from top-F masks
        train_ensemble = np.mean(
            [r["train_embeddings"] for r in self.mask_results[:self.top_F]],
            axis=0,
        )
        eval_ensemble = np.mean(
            [r["eval_embeddings"] for r in self.mask_results[:self.top_F]],
            axis=0,
        )
        
        return self.best_masks, train_ensemble, eval_ensemble
    
    def get_embeddings_for_ood(
        self,
        ood_dataloader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """
        Extract and ensemble embeddings for OOD data using the optimized masks.
        Must be called after optimize().
        
        Args:
            ood_dataloader: DataLoader for OOD test data
            
        Returns:
            ood_ensemble_embeddings: (N_ood, hidden_size)
        """
        if not self.best_masks:
            raise RuntimeError("Must call optimize() before get_embeddings_for_ood()")
        
        ood_embeddings_list = []
        
        for i, mask in enumerate(self.best_masks):
            logger.info(f"Extracting OOD embeddings with mask {i+1}/{len(self.best_masks)}")
            ood_embs = self._extract_embeddings_with_mask(ood_dataloader, mask)
            ood_embeddings_list.append(ood_embs)
        
        # Ensemble (average)
        return np.mean(ood_embeddings_list, axis=0)
    
    def save_results(self, save_dir: str):
        """Save AHM optimization results (masks and scores)."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best masks
        for i, mask in enumerate(self.best_masks):
            mask_path = os.path.join(save_dir, f"best_mask_{i}.npy")
            np.save(mask_path, mask.cpu().numpy())
        
        # Save scores and metadata
        meta = {
            "exploration_budget": self.exploration_budget,
            "masking_percentages": self.masking_percentages,
            "neighbors_K": self.neighbors_K,
            "top_F": self.top_F,
            "all_scores": [
                {
                    "trial_idx": r["trial_idx"],
                    "score": r["score"],
                    "masking_pct": r["masking_pct"],
                }
                for r in self.mask_results
            ],
        }
        
        with open(os.path.join(save_dir, "ahm_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"AHM results saved to {save_dir}")
    
    def load_masks(self, save_dir: str) -> List[torch.Tensor]:
        """Load previously saved best masks."""
        masks = []
        i = 0
        while True:
            mask_path = os.path.join(save_dir, f"best_mask_{i}.npy")
            if not os.path.exists(mask_path):
                break
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
            masks.append(mask)
            i += 1
        
        self.best_masks = masks
        logger.info(f"Loaded {len(masks)} AHM masks from {save_dir}")
        return masks
