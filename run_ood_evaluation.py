"""
run_ood_evaluation.py — Evaluate all OOD detection methods.

Loads precomputed embeddings and logits, runs all 14+ OOD scoring methods,
computes AUROC and FPR@95, and produces the results table (Table 1).

Methods evaluated:
    Confidence-based: MSP, MaxLogit, Energy, KL
    Feature-based: GradNorm, VIM, Residual, NECO, ReAct, ASH
    Distance-based: kNN, Mahalanobis
    Embedding-based: mahAvgAvg, mahGnome
    AHM-enhanced: knnAHM, mahAHM, mahAvgAvg_AHM

Usage:
    python run_ood_evaluation.py \
        --embedding_dir ./outputs/embeddings/FinanceDocs_intra_seed42 \
        --ahm_dir ./outputs/ahm_masks/FinanceDocs_intra_seed42 \
        --output_dir ./outputs/results \
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import get_default_config, get_label2id, get_id2label
from ood_methods.confidence_based import (
    MSPScorer, MaxLogitScorer, EnergyScorer, KLDivergenceScorer, GradNormScorer,
)
from models.layoutlmv3_classifier import LayoutLMv3DocumentClassifier
from utils.data import create_dataloader, encode_labels, filter_dataset_by_classes
from ood_methods.feature_based import (
    VIMScorer, ResidualScorer, NECOScorer, ReActScorer, ASHScorer,
)
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from ood_methods.embedding_methods import AvgAvgEmbedding, GnomeEmbedding
from evaluation.metrics import (
    evaluate_ood_method,
    format_results_table,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(embedding_dir, ahm_dir=None):
    """Load all precomputed embeddings and logits."""
    data = {}
    
    files_to_load = {
        # Standard embeddings
        "train_embs": "train_embeddings.npy",
        "train_labels": "train_labels.npy",
        "eval_embs": "eval_embeddings.npy",
        "eval_labels": "eval_labels.npy",
        "ood_embs": "ood_embeddings.npy",
        "ood_labels": "ood_labels.npy",
        # Logits
        "train_logits": "train_logits.npy",
        "eval_logits": "eval_logits.npy",
        "ood_logits": "ood_logits.npy",
        # Multi-layer (for AvgAvg)
        "train_layer_embs": "train_layer_embeddings.npy",
        "eval_layer_embs": "eval_layer_embeddings.npy",
        "ood_layer_embs": "ood_layer_embeddings.npy",
        # Pretrained (for Gnome)
        "train_pretrained_embs": "train_pretrained_embeddings.npy",
        "eval_pretrained_embs": "eval_pretrained_embeddings.npy",
        "ood_pretrained_embs": "ood_pretrained_embeddings.npy",
    }
    
    for key, filename in files_to_load.items():
        path = os.path.join(embedding_dir, filename)
        if os.path.exists(path):
            data[key] = np.load(path)
            logger.info(f"Loaded {key}: shape={data[key].shape}")
        else:
            data[key] = None
            logger.warning(f"Not found: {path}")
    
    # Load AHM embeddings if available
    if ahm_dir:
        ahm_files = {
            "train_ahm_embs": "train_ahm_embeddings.npy",
            "eval_ahm_embs": "eval_ahm_embeddings.npy",
            "ood_ahm_embs": "ood_ahm_embeddings.npy",
            "train_avgavg_ahm_embs": "train_avgavg_ahm_embeddings.npy",
            "eval_avgavg_ahm_embs": "eval_avgavg_ahm_embeddings.npy",
            "ood_avgavg_ahm_embs": "ood_avgavg_ahm_embeddings.npy",
            "ahm_train_labels": "train_labels.npy",
            "ahm_eval_labels": "eval_labels.npy",
        }
        
        for key, filename in ahm_files.items():
            path = os.path.join(ahm_dir, filename)
            if os.path.exists(path):
                data[key] = np.load(path)
                logger.info(f"Loaded AHM {key}: shape={data[key].shape}")
            else:
                data[key] = None
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection methods")
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Directory with precomputed embeddings")
    parser.add_argument("--ahm_dir", type=str, default=None,
                        help="Directory with AHM-enhanced embeddings")
    parser.add_argument("--output_dir", type=str, default="./outputs/results",
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"])
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model (required for GradNorm)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to processed dataset (required for GradNorm)")
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD dataset (for cross-dataset GradNorm)")
    
    args = parser.parse_args()
    config = get_default_config()
    
    # Output directory — use args.output_dir directly.
    # The orchestrator already builds the per-seed path.
    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)
    
    # ========================================
    # Load all embeddings
    # ========================================
    logger.info("Loading embeddings...")
    data = load_embeddings(args.embedding_dir, args.ahm_dir)
    
    # Verify required data is available
    required = ["eval_embs", "eval_labels", "ood_embs", "train_embs", "train_labels"]
    for key in required:
        if data[key] is None:
            logger.error(f"Required data missing: {key}")
            return
    
    train_embs = data["train_embs"]
    train_labels = data["train_labels"]
    eval_embs = data["eval_embs"]
    eval_labels = data["eval_labels"]
    ood_embs = data["ood_embs"]
    
    all_results = []
    
    # ========================================
    # 1. Confidence-based methods
    # ========================================
    if data["eval_logits"] is not None and data["ood_logits"] is not None:
        eval_logits = data["eval_logits"]
        ood_logits = data["ood_logits"]
        
        # MSP
        logger.info("Evaluating MSP...")
        msp = MSPScorer()
        id_scores = msp.score(eval_logits)
        ood_scores = msp.score(ood_logits)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "msp"))
        
        # MaxLogit
        logger.info("Evaluating MaxLogit...")
        maxlogit = MaxLogitScorer()
        id_scores = maxlogit.score(eval_logits)
        ood_scores = maxlogit.score(ood_logits)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "maxLogit"))
        
        # Energy
        logger.info("Evaluating Energy...")
        energy = EnergyScorer(temperature=config.ood.energy_temperature)
        id_scores = energy.score(eval_logits)
        ood_scores = energy.score(ood_logits)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "energy"))
        
        # KL Divergence
        logger.info("Evaluating KL Divergence...")
        kl = KLDivergenceScorer()
        id_scores = kl.score(eval_logits)
        ood_scores = kl.score(ood_logits)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "kl"))
    
    # GradNorm (requires model + data, not just logits)
    if args.model_path and args.data_path and os.path.exists(args.model_path):
        logger.info("Evaluating GradNorm...")
        try:
            dataset_config = config.financedocs if args.dataset == "financedocs" else config.tobacco
            # Pass the class list, not the dataset name string
            if args.experiment_type == "intra":
                id_classes = dataset_config.id_classes
                ood_class_list = [dataset_config.ood_class]
            else:
                id_classes = dataset_config.all_classes
                ood_class_list = []  # Cross-dataset OOD handled differently
            label2id = get_label2id(id_classes)
            id2label = {v: k for k, v in label2id.items()}
            num_labels = len(id_classes)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Fix: use correct constructor params and load via from_pretrained pattern
            model = LayoutLMv3DocumentClassifier(
                model_name=args.model_path,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
            )
            model.to(device)

            full_ds = load_from_disk(args.data_path)

            # Fix: full_ds is a flat Dataset (not DatasetDict), so we filter directly
            from sklearn.model_selection import train_test_split
            eval_ds = filter_dataset_by_classes(full_ds, id_classes)
            eval_ds = encode_labels(eval_ds, label2id)
            # Reproduce same train/eval split as training
            labels_arr = np.array(eval_ds["labels"])
            indices = np.arange(len(eval_ds))
            _, eval_idx = train_test_split(
                indices, train_size=config.train_eval_split,
                random_state=args.seed, stratify=labels_arr,
            )
            eval_ds = eval_ds.select(eval_idx)
            eval_dl = create_dataloader(eval_ds, batch_size=8, shuffle=False)

            # Fix: use [ood_class] (singular attribute) for intra
            if args.experiment_type == "intra":
                ood_ds = filter_dataset_by_classes(full_ds, ood_class_list)
                # Encode OOD labels with dummy class ID so collate_fn can convert to tensor
                ood_ds = encode_labels(ood_ds, {ood_class_list[0]: num_labels})
            else:
                if args.ood_data_path and os.path.exists(args.ood_data_path):
                    ood_ds = load_from_disk(args.ood_data_path)
                    # Encode OOD labels with dummy class ID so collate_fn can convert to tensor
                    ood_label_set = set(ood_ds["labels"])
                    ood_dummy_map = {lbl: num_labels for lbl in ood_label_set}
                    ood_ds = encode_labels(ood_ds, ood_dummy_map)
                    logger.info(f"GradNorm cross-dataset OOD loaded: {len(ood_ds)} samples")
                elif args.data_path:
                    ood_ds = full_ds
                    logger.warning("GradNorm cross-dataset: no --ood_data_path, using full dataset as proxy")
            ood_dl = create_dataloader(ood_ds, batch_size=8, shuffle=False)

            gradnorm = GradNormScorer()
            id_scores = gradnorm.score(model, eval_dl, device=device, num_classes=num_labels)
            ood_scores = gradnorm.score(model, ood_dl, device=device, num_classes=num_labels)
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "gradNorm"))

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"GradNorm evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("Skipping GradNorm (--model_path or --data_path not provided)")

    # ========================================
    # 2. Feature-based methods
    # ========================================
    
    # Load classifier weights for ReAct/ASH proper re-forwarding
    classifier_weight = None
    classifier_bias = None
    dense_weight = None
    dense_bias = None
    clf_w_path = os.path.join(args.embedding_dir, "classifier_weight.npy")
    clf_b_path = os.path.join(args.embedding_dir, "classifier_bias.npy")
    dense_w_path = os.path.join(args.embedding_dir, "dense_weight.npy")
    dense_b_path = os.path.join(args.embedding_dir, "dense_bias.npy")
    if os.path.exists(clf_w_path):
        classifier_weight = np.load(clf_w_path)
        logger.info(f"Loaded classifier_weight: shape={classifier_weight.shape}")
    if os.path.exists(clf_b_path):
        classifier_bias = np.load(clf_b_path)
        logger.info(f"Loaded classifier_bias: shape={classifier_bias.shape}")
    if os.path.exists(dense_w_path):
        dense_weight = np.load(dense_w_path)
        logger.info(f"Loaded dense_weight: shape={dense_weight.shape}")
    if os.path.exists(dense_b_path):
        dense_bias = np.load(dense_b_path)
        logger.info(f"Loaded dense_bias: shape={dense_bias.shape}")

    # VIM
    if data["eval_logits"] is not None and data["ood_logits"] is not None:
        logger.info("Evaluating VIM...")
        vim = VIMScorer(principal_dim=config.ood.vim_principal_dim)
        vim.fit(train_embs, data["train_logits"])
        id_scores = vim.score(eval_embs, data["eval_logits"])
        ood_scores = vim.score(ood_embs, data["ood_logits"])
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "vim"))
    
    # Residual
    logger.info("Evaluating Residual...")
    residual = ResidualScorer(principal_dim=config.ood.vim_principal_dim)
    residual.fit(train_embs)
    id_scores = residual.score(eval_embs)
    ood_scores = residual.score(ood_embs)
    all_results.append(evaluate_ood_method(id_scores, ood_scores, "residual"))
    
    # NECO
    logger.info("Evaluating NECO...")
    neco = NECOScorer()
    neco.fit(train_embs, train_labels)
    id_scores = neco.score(eval_embs)
    ood_scores = neco.score(ood_embs)
    all_results.append(evaluate_ood_method(id_scores, ood_scores, "neco"))
    
    # ReAct
    if data["eval_logits"] is not None and data["ood_logits"] is not None:
        logger.info("Evaluating ReAct...")
        react = ReActScorer(threshold_percentile=config.ood.react_threshold_percentile)
        react.fit(train_embs)
        id_scores = react.score(eval_embs, data["eval_logits"],
                                classifier_weight=classifier_weight,
                                classifier_bias=classifier_bias,
                                dense_weight=dense_weight,
                                dense_bias=dense_bias)
        ood_scores = react.score(ood_embs, data["ood_logits"],
                                 classifier_weight=classifier_weight,
                                 classifier_bias=classifier_bias,
                                 dense_weight=dense_weight,
                                 dense_bias=dense_bias)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "react"))
    
    # ASH  
    if data["eval_logits"] is not None and data["ood_logits"] is not None:
        logger.info("Evaluating ASH...")
        ash = ASHScorer(pruning_percentile=config.ood.ash_pruning_percentile)
        id_scores = ash.score(eval_embs, data["eval_logits"],
                              classifier_weight=classifier_weight,
                              classifier_bias=classifier_bias,
                              dense_weight=dense_weight,
                              dense_bias=dense_bias)
        ood_scores = ash.score(ood_embs, data["ood_logits"],
                               classifier_weight=classifier_weight,
                               classifier_bias=classifier_bias,
                               dense_weight=dense_weight,
                               dense_bias=dense_bias)
        all_results.append(evaluate_ood_method(id_scores, ood_scores, "ash"))
    
    # ========================================
    # 3. Distance-based methods (standard)
    # ========================================
    
    # kNN
    logger.info("Evaluating kNN...")
    knn = KNNScorer(k=config.ood.knn_k, metric="euclidean")
    knn.fit(train_embs)
    id_scores = knn.score(eval_embs)
    ood_scores = knn.score(ood_embs)
    all_results.append(evaluate_ood_method(id_scores, ood_scores, "knn"))
    
    # Mahalanobis
    logger.info("Evaluating Mahalanobis...")
    mah = MahalanobisScorer()
    mah.fit(train_embs, train_labels)
    id_scores = mah.score(eval_embs)
    ood_scores = mah.score(ood_embs)
    all_results.append(evaluate_ood_method(id_scores, ood_scores, "Mahalanobis"))
    
    # ========================================
    # 4. Embedding-based methods
    # ========================================
    
    # AvgAvg + Mahalanobis (mahAvgAvg)
    if data["train_layer_embs"] is not None and data["eval_layer_embs"] is not None:
        logger.info("Evaluating mahAvgAvg...")
        avgavg = AvgAvgEmbedding()
        
        # Convert from (num_layers, N, hidden) to list of (N, hidden)
        train_layers = [data["train_layer_embs"][i] for i in range(data["train_layer_embs"].shape[0])]
        eval_layers = [data["eval_layer_embs"][i] for i in range(data["eval_layer_embs"].shape[0])]
        
        train_avgavg_embs = avgavg.aggregate(train_layers)
        eval_avgavg_embs = avgavg.aggregate(eval_layers)
        
        mah_avgavg = MahalanobisScorer()
        mah_avgavg.fit(train_avgavg_embs, train_labels)
        id_scores = mah_avgavg.score(eval_avgavg_embs)
        
        if data["ood_layer_embs"] is not None:
            ood_layers = [data["ood_layer_embs"][i] for i in range(data["ood_layer_embs"].shape[0])]
            ood_avgavg_embs = avgavg.aggregate(ood_layers)
            ood_scores = mah_avgavg.score(ood_avgavg_embs)
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "mahAvgAvg"))
    
    # Gnome + Mahalanobis (mahGnome)
    if (data["train_pretrained_embs"] is not None and 
        data["eval_pretrained_embs"] is not None):
        logger.info("Evaluating mahGnome...")
        gnome = GnomeEmbedding(combination="concat")
        
        train_gnome_embs = gnome.aggregate(data["train_pretrained_embs"], train_embs)
        eval_gnome_embs = gnome.aggregate(data["eval_pretrained_embs"], eval_embs)
        
        mah_gnome = MahalanobisScorer()
        mah_gnome.fit(train_gnome_embs, train_labels)
        id_scores = mah_gnome.score(eval_gnome_embs)
        
        if data["ood_pretrained_embs"] is not None:
            ood_gnome_embs = gnome.aggregate(data["ood_pretrained_embs"], ood_embs)
            ood_scores = mah_gnome.score(ood_gnome_embs)
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "mahGnome"))
    
    # ========================================
    # 5. AHM-enhanced methods (paper's contribution)
    # ========================================
    
    if data.get("train_ahm_embs") is not None:
        ahm_train_labels = data["ahm_train_labels"] if data.get("ahm_train_labels") is not None else train_labels
        
        # knnAHM
        if data.get("eval_ahm_embs") is not None and data.get("ood_ahm_embs") is not None:
            logger.info("Evaluating knnAHM...")
            knn_ahm = KNNScorer(k=config.ood.knn_k, metric="euclidean")
            knn_ahm.fit(data["train_ahm_embs"])
            id_scores = knn_ahm.score(data["eval_ahm_embs"])
            ood_scores = knn_ahm.score(data["ood_ahm_embs"])
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "knnAHM"))
        
        # mahAHM
        if data.get("eval_ahm_embs") is not None and data.get("ood_ahm_embs") is not None:
            logger.info("Evaluating mahAHM...")
            mah_ahm = MahalanobisScorer()
            mah_ahm.fit(data["train_ahm_embs"], ahm_train_labels)
            id_scores = mah_ahm.score(data["eval_ahm_embs"])
            ood_scores = mah_ahm.score(data["ood_ahm_embs"])
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "mahAHM"))
    
    # mahAvgAvg_AHM
    if data.get("train_avgavg_ahm_embs") is not None:
        logger.info("Evaluating mahAvgAvg_AHM...")
        ahm_train_labels = data["ahm_train_labels"] if data.get("ahm_train_labels") is not None else train_labels
        
        mah_avgavg_ahm = MahalanobisScorer()
        mah_avgavg_ahm.fit(data["train_avgavg_ahm_embs"], ahm_train_labels)
        id_scores = mah_avgavg_ahm.score(data["eval_avgavg_ahm_embs"])
        
        if data.get("ood_avgavg_ahm_embs") is not None:
            ood_scores = mah_avgavg_ahm.score(data["ood_avgavg_ahm_embs"])
            all_results.append(evaluate_ood_method(id_scores, ood_scores, "mahAvgAvg_AHM"))
    
    # ========================================
    # Format and save results
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Print results
    dataset_name = args.dataset.capitalize()
    table = format_results_table(
        {r["method"]: {"auroc_mean": r["auroc"], "auroc_std": 0, 
                        "fpr_mean": r["fpr"], "fpr_std": 0}
         for r in all_results},
        dataset_name=dataset_name,
        experiment_type=args.experiment_type,
    )
    logger.info(table)
    
    # Save results as JSON
    results_path = os.path.join(run_output_dir, "ood_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save formatted table
    table_path = os.path.join(run_output_dir, "results_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    
    logger.info(f"\nResults saved to {run_output_dir}")
    logger.info("OOD evaluation complete!")


if __name__ == "__main__":
    main()
