"""
run_extract_embeddings.py — Extract embeddings from fine-tuned LayoutLMv3.

Extracts:
1. Last-layer CLS embeddings (standard, for Mahalanobis/kNN)
2. All-layer CLS embeddings (for AvgAvg method)
3. Logits (for confidence-based methods: MSP, Energy, MaxLogit, KL)
4. Pretrained model embeddings (for Gnome method)

Saves all embeddings as .npy files for reuse by OOD evaluation.

Usage:
    python run_extract_embeddings.py \
        --model_path ./outputs/checkpoints/FinanceDocs_intra_seed42/best_model \
        --data_path ./colated_tensors \
        --output_dir ./outputs/embeddings \
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (
    get_default_config,
    get_label2id,
    get_id2label,
)
from models.layoutlmv3_classifier import (
    LayoutLMv3DocumentClassifier,
    extract_embeddings_from_dataloader,
    extract_logits_from_dataloader,
    extract_all_layer_embeddings_from_dataloader,
)
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed tensor dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD dataset (for cross-dataset experiments)")
    
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
    
    # Output directory for this run — use args.output_dir directly.
    # The orchestrator already builds the per-seed path.
    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)
    
    # ========================================
    # Load dataset and prepare splits
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
        
        # Same train/eval split as during training
        from sklearn.model_selection import train_test_split
        labels = np.array(id_dataset["labels"])
        indices = np.arange(len(id_dataset))
        train_idx, eval_idx = train_test_split(
            indices, train_size=config.train_eval_split,
            random_state=args.seed, stratify=labels,
        )
        
        train_dataset = id_dataset.select(train_idx)
        eval_dataset = id_dataset.select(eval_idx)
        
    else:
        all_classes = dataset_config.all_classes
        label2id = get_label2id(all_classes)
        num_labels = len(all_classes)
        
        id_dataset = encode_labels(dataset, label2id)
        
        from sklearn.model_selection import train_test_split
        labels = np.array(id_dataset["labels"])
        indices = np.arange(len(id_dataset))
        train_idx, eval_idx = train_test_split(
            indices, train_size=config.train_eval_split,
            random_state=args.seed, stratify=labels,
        )
        
        train_dataset = id_dataset.select(train_idx)
        eval_dataset = id_dataset.select(eval_idx)
        
        # Load OOD dataset (other dataset)
        ood_dataset = None
        if args.ood_data_path:
            ood_dataset = load_from_disk(args.ood_data_path)
            # Encode OOD labels with dummy class ID so collate_fn can convert to tensor
            ood_label_set = set(ood_dataset["labels"])
            ood_dummy_map = {lbl: num_labels for lbl in ood_label_set}
            ood_dataset = encode_labels(ood_dataset, ood_dummy_map)
            logger.info(f"Cross-dataset OOD loaded: {len(ood_dataset)} samples, labels mapped to dummy id {num_labels}")
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    if args.experiment_type == "intra" or ood_dataset is not None:
        logger.info(f"OOD: {len(ood_dataset) if ood_dataset else 0}")
    
    # ========================================
    # Create DataLoaders
    # ========================================
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    eval_loader = create_dataloader(eval_dataset, batch_size=args.batch_size)
    ood_loader = create_dataloader(ood_dataset, batch_size=args.batch_size) if ood_dataset else None
    
    # ========================================
    # Load fine-tuned model
    # ========================================
    logger.info(f"Loading fine-tuned model from {args.model_path}")
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
    # 1. Extract last-layer CLS embeddings
    # ========================================
    logger.info("Extracting last-layer CLS embeddings...")
    
    train_embs, train_labels = extract_embeddings_from_dataloader(
        model, train_loader, device=device, layer="last", desc="Train embeddings"
    )
    eval_embs, eval_labels = extract_embeddings_from_dataloader(
        model, eval_loader, device=device, layer="last", desc="Eval embeddings"
    )
    
    np.save(os.path.join(run_output_dir, "train_embeddings.npy"), train_embs)
    np.save(os.path.join(run_output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(run_output_dir, "eval_embeddings.npy"), eval_embs)
    np.save(os.path.join(run_output_dir, "eval_labels.npy"), eval_labels)
    
    logger.info(f"Train embeddings: {train_embs.shape}, Eval embeddings: {eval_embs.shape}")
    
    if ood_loader:
        ood_embs, ood_labels = extract_embeddings_from_dataloader(
            model, ood_loader, device=device, layer="last", desc="OOD embeddings"
        )
        np.save(os.path.join(run_output_dir, "ood_embeddings.npy"), ood_embs)
        np.save(os.path.join(run_output_dir, "ood_labels.npy"), ood_labels)
        logger.info(f"OOD embeddings: {ood_embs.shape}")
    
    # ========================================
    # 2. Extract logits
    # ========================================
    logger.info("Extracting logits...")
    
    eval_logits, _ = extract_logits_from_dataloader(
        model, eval_loader, device=device, desc="Eval logits"
    )
    np.save(os.path.join(run_output_dir, "eval_logits.npy"), eval_logits)
    
    train_logits, _ = extract_logits_from_dataloader(
        model, train_loader, device=device, desc="Train logits"
    )
    np.save(os.path.join(run_output_dir, "train_logits.npy"), train_logits)
    
    if ood_loader:
        ood_logits, _ = extract_logits_from_dataloader(
            model, ood_loader, device=device, desc="OOD logits"
        )
        np.save(os.path.join(run_output_dir, "ood_logits.npy"), ood_logits)
    
    # ========================================
    # 3. Extract all-layer CLS embeddings (for AvgAvg)
    # ========================================
    logger.info("Extracting multi-layer embeddings (for AvgAvg)...")
    
    train_layer_embs, _ = extract_all_layer_embeddings_from_dataloader(
        model, train_loader, device=device, desc="Train multi-layer"
    )
    eval_layer_embs, _ = extract_all_layer_embeddings_from_dataloader(
        model, eval_loader, device=device, desc="Eval multi-layer"
    )
    
    # Save as single array: (num_layers, N, hidden_size)
    np.save(
        os.path.join(run_output_dir, "train_layer_embeddings.npy"),
        np.stack(train_layer_embs, axis=0),
    )
    np.save(
        os.path.join(run_output_dir, "eval_layer_embeddings.npy"),
        np.stack(eval_layer_embs, axis=0),
    )
    
    if ood_loader:
        ood_layer_embs, _ = extract_all_layer_embeddings_from_dataloader(
            model, ood_loader, device=device, desc="OOD multi-layer"
        )
        np.save(
            os.path.join(run_output_dir, "ood_layer_embeddings.npy"),
            np.stack(ood_layer_embs, axis=0),
        )
    
    # ========================================
    # 4. Extract pretrained model embeddings (for Gnome)
    # ========================================
    logger.info("Extracting pretrained model embeddings (for Gnome)...")
    
    pretrained_model = LayoutLMv3DocumentClassifier(
        model_name=config.model.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    pretrained_model.eval()
    pretrained_model.to(device)
    
    train_pretrained_embs, _ = extract_embeddings_from_dataloader(
        pretrained_model, train_loader, device=device, layer="last",
        desc="Train pretrained embeddings"
    )
    eval_pretrained_embs, _ = extract_embeddings_from_dataloader(
        pretrained_model, eval_loader, device=device, layer="last",
        desc="Eval pretrained embeddings"
    )
    
    np.save(os.path.join(run_output_dir, "train_pretrained_embeddings.npy"), train_pretrained_embs)
    np.save(os.path.join(run_output_dir, "eval_pretrained_embeddings.npy"), eval_pretrained_embs)
    
    if ood_loader:
        ood_pretrained_embs, _ = extract_embeddings_from_dataloader(
            pretrained_model, ood_loader, device=device, layer="last",
            desc="OOD pretrained embeddings"
        )
        np.save(os.path.join(run_output_dir, "ood_pretrained_embeddings.npy"), ood_pretrained_embs)
    
    # Cleanup pretrained model
    del pretrained_model
    torch.cuda.empty_cache()
    
    # ========================================
    # 5. Save classifier head weights (for ReAct / ASH proper re-forward)
    # ========================================
    logger.info("Saving classifier head weights for ReAct/ASH...")
    classifier_weight = None
    classifier_bias = None
    dense_weight = None
    dense_bias = None
    for name, param in model.model.named_parameters():
        if "classifier.out_proj.weight" in name:
            classifier_weight = param.detach().cpu().numpy()
        if "classifier.out_proj.bias" in name:
            classifier_bias = param.detach().cpu().numpy()
        if "classifier.dense.weight" in name:
            dense_weight = param.detach().cpu().numpy()
        if "classifier.dense.bias" in name:
            dense_bias = param.detach().cpu().numpy()
    
    if classifier_weight is not None:
        np.save(os.path.join(run_output_dir, "classifier_weight.npy"), classifier_weight)
    if classifier_bias is not None:
        np.save(os.path.join(run_output_dir, "classifier_bias.npy"), classifier_bias)
    if dense_weight is not None:
        np.save(os.path.join(run_output_dir, "dense_weight.npy"), dense_weight)
    if dense_bias is not None:
        np.save(os.path.join(run_output_dir, "dense_bias.npy"), dense_bias)
    
    # ========================================
    # Save metadata
    # ========================================
    meta = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "dataset": dataset_config.name,
        "experiment_type": args.experiment_type,
        "seed": args.seed,
        "num_labels": num_labels,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "ood_size": len(ood_dataset) if ood_dataset else 0,
        "embedding_dim": int(train_embs.shape[1]),
        "num_layers_extracted": len(train_layer_embs),
        "files_saved": [
            "train_embeddings.npy", "train_labels.npy",
            "eval_embeddings.npy", "eval_labels.npy",
            "train_logits.npy", "eval_logits.npy",
            "train_layer_embeddings.npy", "eval_layer_embeddings.npy",
            "train_pretrained_embeddings.npy", "eval_pretrained_embeddings.npy",
        ],
    }
    
    with open(os.path.join(run_output_dir, "embedding_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"All embeddings saved to {run_output_dir}")


if __name__ == "__main__":
    main()
