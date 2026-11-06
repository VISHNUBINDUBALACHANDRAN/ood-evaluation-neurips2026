"""
run_ahm_optimization.py — Attention Head Masking Optimization (Algorithm 1).

Implements the full AHM pipeline:
1. Load fine-tuned model
2. Generate T random AHM masks with masking percentage p
3. For each mask: extract embeddings, compute similarity score
4. Rank masks, select top-F
5. Ensemble embeddings from top-F masks
6. Save AHM-enhanced embeddings for OOD evaluation

Usage:
    python run_ahm_optimization.py \
        --model_path ./outputs/checkpoints/FinanceDocs_intra_seed42/best_model \
        --data_path ./colated_tensors \
        --output_dir ./outputs/ahm_masks \
        --dataset financedocs \
        --experiment_type intra \
        --seed 42
"""

import argparse
import os
import sys
import logging
import json
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (
    get_default_config,
    get_label2id,
    get_id2label,
)
from models.layoutlmv3_classifier import LayoutLMv3DocumentClassifier
from models.attention_masking import AttentionHeadMasker, AHMOptimizer
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AHM Optimization (Algorithm 1)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed tensor dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/ahm_masks",
                        help="Directory to save AHM results")
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD dataset (for cross-dataset)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override exploration budget T")
    parser.add_argument("--top_f", type=int, default=None,
                        help="Override top-F masks to select")
    
    args = parser.parse_args()
    config = get_default_config()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get dataset config
    if args.dataset == "financedocs":
        dataset_config = config.financedocs
    else:
        dataset_config = config.tobacco
    
    # Output directory — use args.output_dir directly.
    # The orchestrator already builds the per-seed path.
    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)
    
    # ========================================
    # Load and prepare data
    # ========================================
    logger.info(f"Loading data from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    
    if args.experiment_type == "intra":
        id_classes = dataset_config.id_classes
        ood_class = dataset_config.ood_class
        label2id = get_label2id(id_classes)
        num_labels = len(id_classes)
        
        id_dataset = filter_dataset_by_classes(dataset, id_classes)
        ood_dataset = filter_dataset_by_classes(dataset, [ood_class])
        
        id_dataset = encode_labels(id_dataset, label2id)
        # Encode OOD labels with dummy class ID so collate_fn can convert to tensor
        ood_dataset = encode_labels(ood_dataset, {ood_class: num_labels})
    else:
        all_classes = dataset_config.all_classes
        label2id = get_label2id(all_classes)
        num_labels = len(all_classes)
        
        id_dataset = encode_labels(dataset, label2id)
        ood_dataset = None
        if args.ood_data_path:
            ood_dataset = load_from_disk(args.ood_data_path)
            # Encode OOD labels with dummy class ID so collate_fn can convert to tensor
            ood_label_set = set(ood_dataset["labels"])
            ood_dummy_map = {lbl: num_labels for lbl in ood_label_set}
            ood_dataset = encode_labels(ood_dataset, ood_dummy_map)
            logger.info(f"Cross-dataset OOD loaded: {len(ood_dataset)} samples, labels mapped to dummy id {num_labels}")
    
    # Train/eval split (same as training)
    labels = np.array(id_dataset["labels"])
    indices = np.arange(len(id_dataset))
    train_idx, eval_idx = train_test_split(
        indices, train_size=config.train_eval_split,
        random_state=args.seed, stratify=labels,
    )
    
    train_dataset = id_dataset.select(train_idx)
    eval_dataset = id_dataset.select(eval_idx)
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    eval_loader = create_dataloader(eval_dataset, batch_size=args.batch_size)
    ood_loader = create_dataloader(ood_dataset, batch_size=args.batch_size) if ood_dataset else None
    
    # ========================================
    # Load fine-tuned model
    # ========================================
    logger.info(f"Loading model from {args.model_path}")
    id2label = get_id2label(
        dataset_config.id_classes if args.experiment_type == "intra"
        else dataset_config.all_classes
    )
    
    model = LayoutLMv3DocumentClassifier(
        model_name=args.model_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.eval()
    model.to(device)
    
    # ========================================
    # Run AHM Optimization (Algorithm 1)
    # ========================================
    budget = args.budget or config.ahm.exploration_budget_T
    top_f = args.top_f or config.ahm.top_ahm_matrices_F
    
    logger.info(f"Starting AHM optimization: T={budget}, "
                f"p={config.ahm.percentage_masking_p}, "
                f"K={config.ahm.neighbors_K}, F={top_f}")
    
    ahm_optimizer = AHMOptimizer(
        model=model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        exploration_budget=budget,
        masking_percentages=config.ahm.percentage_masking_p,
        neighbors_K=config.ahm.neighbors_K,
        top_F=top_f,
        device=device,
    )
    
    # Run optimization
    best_masks, train_ahm_embs, eval_ahm_embs = ahm_optimizer.optimize(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        verbose=True,
    )
    
    # Save AHM results
    ahm_optimizer.save_results(run_output_dir)
    
    # Save AHM-enhanced embeddings
    np.save(os.path.join(run_output_dir, "train_ahm_embeddings.npy"), train_ahm_embs)
    np.save(os.path.join(run_output_dir, "eval_ahm_embeddings.npy"), eval_ahm_embs)
    
    logger.info(f"AHM train embeddings: {train_ahm_embs.shape}")
    logger.info(f"AHM eval embeddings: {eval_ahm_embs.shape}")
    
    # ========================================
    # Extract AHM embeddings for OOD data
    # ========================================
    if ood_loader:
        logger.info("Extracting AHM-enhanced OOD embeddings...")
        ood_ahm_embs = ahm_optimizer.get_embeddings_for_ood(ood_loader)
        np.save(os.path.join(run_output_dir, "ood_ahm_embeddings.npy"), ood_ahm_embs)
        logger.info(f"AHM OOD embeddings: {ood_ahm_embs.shape}")
    
    # ========================================
    # Also extract AvgAvg + AHM embeddings (mahAvgAvg_AHM)
    # ========================================
    logger.info("Extracting AvgAvg + AHM embeddings...")
    
    # For each of the best masks, extract all-layer embeddings and compute AvgAvg
    from ood_methods.embedding_methods import AvgAvgEmbedding
    avgavg = AvgAvgEmbedding()
    
    train_avgavg_ahm_list = []
    eval_avgavg_ahm_list = []
    ood_avgavg_ahm_list = []
    
    for mask_idx, mask in enumerate(best_masks):
        logger.info(f"AvgAvg+AHM: processing mask {mask_idx+1}/{len(best_masks)}")
        
        model.set_attention_head_mask(mask)
        
        # Extract all-layer embeddings
        from models.layoutlmv3_classifier import extract_all_layer_embeddings_from_dataloader
        
        train_layers, _ = extract_all_layer_embeddings_from_dataloader(
            model, train_loader, device=device, desc=f"Train AvgAvg mask {mask_idx}"
        )
        eval_layers, _ = extract_all_layer_embeddings_from_dataloader(
            model, eval_loader, device=device, desc=f"Eval AvgAvg mask {mask_idx}"
        )
        
        train_avgavg_ahm_list.append(avgavg.aggregate(train_layers))
        eval_avgavg_ahm_list.append(avgavg.aggregate(eval_layers))
        
        if ood_loader:
            ood_layers, _ = extract_all_layer_embeddings_from_dataloader(
                model, ood_loader, device=device, desc=f"OOD AvgAvg mask {mask_idx}"
            )
            ood_avgavg_ahm_list.append(avgavg.aggregate(ood_layers))
    
    model.set_attention_head_mask(None)  # Reset
    
    # Ensemble the AvgAvg+AHM embeddings
    train_avgavg_ahm = np.mean(train_avgavg_ahm_list, axis=0)
    eval_avgavg_ahm = np.mean(eval_avgavg_ahm_list, axis=0)
    
    np.save(os.path.join(run_output_dir, "train_avgavg_ahm_embeddings.npy"), train_avgavg_ahm)
    np.save(os.path.join(run_output_dir, "eval_avgavg_ahm_embeddings.npy"), eval_avgavg_ahm)
    
    if ood_loader and ood_avgavg_ahm_list:
        ood_avgavg_ahm = np.mean(ood_avgavg_ahm_list, axis=0)
        np.save(os.path.join(run_output_dir, "ood_avgavg_ahm_embeddings.npy"), ood_avgavg_ahm)
    
    # Save labels
    train_labels = np.array(train_dataset["labels"])
    eval_labels = np.array(eval_dataset["labels"])
    np.save(os.path.join(run_output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(run_output_dir, "eval_labels.npy"), eval_labels)
    
    logger.info(f"All AHM results saved to {run_output_dir}")
    logger.info("AHM optimization complete!")


if __name__ == "__main__":
    main()
