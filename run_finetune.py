"""
run_finetune.py — Fine-tune LayoutLMv3 for Document Classification.

Implements the training pipeline from Section 4.2 of the paper:
- Loads preprocessed tensor dataset
- Encodes labels to integers
- Splits data into ID train/eval (excluding OOD class for intra-dataset experiments)
- Fine-tunes LayoutLMv3ForSequenceClassification using HuggingFace Trainer
- Filters checkpoints using silhouette scores
- Saves the best checkpoint per seed

Usage:
    python run_finetune.py \
        --data_path ./colated_tensors \
        --output_dir ./outputs/checkpoints \
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
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
import evaluate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (
    get_default_config,
    get_label2id,
    get_id2label,
    ExperimentConfig,
    FinanceDocsConfig,
    Tobacco3482Config,
)
from evaluation.metrics import compute_silhouette_score
from utils.data import (
    encode_labels,
    filter_dataset_by_classes,
    data_collator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_train_eval(dataset, train_ratio=0.8, seed=42):
    """Split dataset into train and eval sets with stratification."""
    labels = np.array(dataset["labels"])
    indices = np.arange(len(dataset))
    
    train_idx, eval_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    
    train_dataset = dataset.select(train_idx)
    eval_dataset = dataset.select(eval_idx)
    
    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation during training."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class SilhouetteCheckpointCallback(TrainerCallback):
    """
    Custom callback to compute silhouette scores during training
    and log them alongside standard metrics.
    """
    
    def __init__(self, eval_dataset, silhouette_threshold=0.0):
        self.eval_dataset = eval_dataset
        self.silhouette_threshold = silhouette_threshold
        self.scores_history = []
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Compute silhouette score after each evaluation."""
        if model is None:
            return
        
        device = next(model.parameters()).device
        model.eval()
        
        # Quick silhouette on a subset for efficiency
        max_samples = min(500, len(self.eval_dataset))
        subset = self.eval_dataset.select(range(max_samples))
        
        all_embs = []
        all_labels_list = []
        
        batch_size = 16
        from utils.data import create_dataloader
        loader = create_dataloader(subset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
                
                cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                all_embs.append(cls_emb)
                all_labels_list.append(batch["labels"].numpy())
        
        all_embs = np.concatenate(all_embs, axis=0)
        all_labels_arr = np.concatenate(all_labels_list, axis=0)
        
        score = compute_silhouette_score(all_embs, all_labels_arr)
        self.scores_history.append({"epoch": state.epoch, "silhouette": score})
        logger.info(f"[Epoch {state.epoch:.1f}] Silhouette score: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed tensor dataset (from run_prepare_data.py)")
    parser.add_argument("--output_dir", type=str, default="./outputs/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--dataset", type=str, default="financedocs",
                        choices=["financedocs", "tobacco"],
                        help="Which dataset to use")
    parser.add_argument("--experiment_type", type=str, default="intra",
                        choices=["intra", "cross"],
                        help="intra-dataset or cross-dataset OOD experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override num_train_epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override per_device_train_batch_size from config")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning_rate from config")
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    # Override config if specified
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get dataset config
    if args.dataset == "financedocs":
        dataset_config = config.financedocs
    else:
        dataset_config = config.tobacco
    
    logger.info(f"Dataset: {dataset_config.name}")
    logger.info(f"Experiment type: {args.experiment_type}")
    logger.info(f"Seed: {args.seed}")
    
    # ========================================
    # Load preprocessed data
    # ========================================
    logger.info(f"Loading data from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # ========================================
    # Prepare ID / OOD splits
    # ========================================
    if args.experiment_type == "intra":
        # Intra-dataset: hold out one class as OOD, train on rest
        id_classes = dataset_config.id_classes
        ood_class = dataset_config.ood_class
        
        logger.info(f"ID classes ({len(id_classes)}): {id_classes}")
        logger.info(f"OOD class: {ood_class}")
        
        # Filter to ID classes only for training
        id_dataset = filter_dataset_by_classes(dataset, id_classes)
        ood_dataset = filter_dataset_by_classes(dataset, [ood_class])
        
        logger.info(f"ID samples: {len(id_dataset)}, OOD samples: {len(ood_dataset)}")
        
        # Create label mapping for ID classes only
        label2id = get_label2id(id_classes)
        id2label = get_id2label(id_classes)
        num_labels = len(id_classes)
        
    else:
        # Cross-dataset: train on ALL classes of this dataset
        all_classes = dataset_config.all_classes
        id_dataset = dataset
        ood_dataset = None  # OOD is the OTHER dataset
        
        label2id = get_label2id(all_classes)
        id2label = get_id2label(all_classes)
        num_labels = len(all_classes)
    
    # Encode string labels → integer IDs
    id_dataset = encode_labels(id_dataset, label2id)
    
    # Split into train/eval
    train_dataset, eval_dataset = split_train_eval(
        id_dataset,
        train_ratio=config.train_eval_split,
        seed=args.seed,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    logger.info(f"Num labels: {num_labels}")
    logger.info(f"Label mapping: {label2id}")
    
    # ========================================
    # Set up model
    # ========================================
    logger.info(f"Loading model: {config.model.model_name}")
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    
    # ========================================
    # Training arguments (Table 2)
    # ========================================
    # Use args.output_dir directly — the orchestrator already builds the
    # per-seed path; sub-scripts must NOT append another subdirectory.
    run_output_dir = args.output_dir
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_train_epochs,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.evaluation_strategy,
        eval_delay=config.training.eval_delay,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        fp16=config.training.fp16,
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # ========================================
    # Initialize Trainer (data_collator imported from utils.data)
    # ========================================
    # Silhouette callback for monitoring during training
    silhouette_cb = SilhouetteCheckpointCallback(
        eval_dataset=eval_dataset,
        silhouette_threshold=config.silhouette_threshold,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[silhouette_cb],
    )
    
    # ========================================
    # Train!
    # ========================================
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # ========================================
    # Checkpoint filtering via silhouette score
    # ========================================
    logger.info("Computing silhouette score for best checkpoint...")
    
    # Extract embeddings for silhouette score
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    all_embeddings = []
    all_labels_list = []
    
    eval_dataloader = trainer.get_eval_dataloader()
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            
            # CLS token from last hidden state
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding)
            all_labels_list.append(batch["labels"].cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels_arr = np.concatenate(all_labels_list, axis=0)
    
    silhouette = compute_silhouette_score(all_embeddings, all_labels_arr)
    logger.info(f"Silhouette score: {silhouette:.4f}")
    
    # Enforce silhouette threshold — reject seed if score is too low
    if silhouette < config.silhouette_threshold:
        logger.warning(
            f"Silhouette score {silhouette:.4f} < threshold "
            f"{config.silhouette_threshold:.4f}. This checkpoint may be unreliable."
        )
    
    # ========================================
    # Save classifier head weights (for ReAct / ASH proper re-forwarding)
    # ========================================
    classifier_weight = None
    classifier_bias = None
    for name, param in model.named_parameters():
        if "classifier" in name and "weight" in name:
            classifier_weight = param.detach().cpu().numpy()
        if "classifier" in name and "bias" in name:
            classifier_bias = param.detach().cpu().numpy()
    
    if classifier_weight is not None:
        np.save(os.path.join(run_output_dir, "classifier_weight.npy"), classifier_weight)
        logger.info(f"Saved classifier weight: {classifier_weight.shape}")
    if classifier_bias is not None:
        np.save(os.path.join(run_output_dir, "classifier_bias.npy"), classifier_bias)
        logger.info(f"Saved classifier bias: {classifier_bias.shape}")
    
    # Save metadata
    meta = {
        "seed": args.seed,
        "dataset": dataset_config.name,
        "experiment_type": args.experiment_type,
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "silhouette_score": silhouette,
        "eval_accuracy": eval_metrics.get("eval_accuracy", None),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    
    meta_path = os.path.join(run_output_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save best model
    best_model_path = os.path.join(run_output_dir, "best_model")
    trainer.save_model(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")
    
    logger.info(f"Training complete! Silhouette: {silhouette:.4f}, "
                f"Eval accuracy: {eval_metrics.get('eval_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()
