"""
Configuration for OOD Detection with Attention Head Masking.
Hyperparameters from Tables 2 and 3 of the paper.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ============================================================
# Dataset Configuration
# ============================================================

@dataclass
class FinanceDocsConfig:
    """FinanceDocs dataset configuration (10 classes)."""
    name: str = "FinanceDocs"
    num_classes: int = 10
    all_classes: List[str] = field(default_factory=lambda: [
        "articles_of_association",
        "financial_info",
        "images__10k_page",
        "list_directors",
        "resumes",
        "scientific_ai_paper",
        "sec_13",
        "sec_form",
        "sec_letter",
        "shareholder_letter",
    ])
    ood_class: str = "resumes"  # Held out as OOD for intra-dataset experiments
    id_classes: List[str] = field(default_factory=lambda: [
        "articles_of_association",
        "financial_info",
        "images__10k_page",
        "list_directors",
        "scientific_ai_paper",
        "sec_13",
        "sec_form",
        "sec_letter",
        "shareholder_letter",
    ])


@dataclass
class Tobacco3482Config:
    """Tobacco3482 dataset configuration (10 classes)."""
    name: str = "Tobacco3482"
    num_classes: int = 10
    all_classes: List[str] = field(default_factory=lambda: [
        "Memo", "Email", "Letter", "Form", "Report",
        "Scientific", "Note", "News", "Advertisement", "Resume",
    ])
    # OOD class: "Advertisement" in standard labelling, "ADVE" in some datasets.
    # The parse_data_paths / dataset labels must match this value exactly.
    ood_class: str = "Advertisement"
    ood_class_aliases: List[str] = field(default_factory=lambda: ["ADVE", "Advertisement"])
    id_classes: List[str] = field(default_factory=lambda: [
        "Memo", "Email", "Letter", "Form", "Report",
        "Scientific", "Note", "News", "Resume",
    ])


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class ModelConfig:
    """LayoutLMv3 model configuration."""
    model_name: str = "microsoft/layoutlmv3-base"
    processor_name: str = "microsoft/layoutlmv3-base"
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    max_seq_length: int = 512
    image_size: int = 224
    total_params: str = "125.92M"


# ============================================================
# Training Hyperparameters (Table 2)
# ============================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters from Table 2 of the paper."""
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 65
    eval_delay: int = 5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4


# ============================================================
# AHM Hyperparameters (Table 3)
# ============================================================

@dataclass
class AHMConfig:
    """Attention Head Masking hyperparameters from Table 3."""
    exploration_budget_T: int = 25
    percentage_masking_p: List[float] = field(default_factory=lambda: [0.1, 0.2])
    neighbors_K: int = 10
    top_ahm_matrices_F: int = 5


# ============================================================
# AHSS Configuration (Attention Head Sensitivity Scoring)
# ============================================================

@dataclass
class AHSSConfig:
    """AHSS hyperparameters."""
    # Number of masks to use for sensitivity scoring
    # Phase 0: reuse the 5 AHM masks + generate additional random masks
    num_random_masks: int = 25          # T random masks for AHSS
    masking_percentages: List[float] = field(default_factory=lambda: [0.1, 0.2])
    # Whether to include the 5 optimized AHM masks in AHSS computation
    use_ahm_masks: bool = True
    # L2-normalize embeddings before computing displacement
    normalize_embeddings: bool = True
    # Combined scoring alpha (0 = pure AHSS, 1 = pure Mahalanobis)
    combined_alpha: float = 0.5
    # Alpha grid for searching optimal combination weight
    alpha_grid: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])


# ============================================================
# OOD Evaluation Configuration
# ============================================================

@dataclass
class OODConfig:
    """OOD detection evaluation configuration."""
    # kNN parameters for OOD scoring
    knn_k: int = 50
    # Mahalanobis parameters
    mahalanobis_layer: str = "last"  # Use last hidden layer
    # ReAct parameters
    react_threshold_percentile: float = 90.0
    # ASH parameters
    ash_pruning_percentile: float = 90.0
    # Energy temperature
    energy_temperature: float = 1.0
    # VIM parameters
    vim_principal_dim: int = 384  # half of hidden_size


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    # Random seeds for 5 runs
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_runs: int = 5
    
    # Device
    device: str = "cuda"
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    embedding_dir: str = "./outputs/embeddings"
    results_dir: str = "./outputs/results"
    ahm_dir: str = "./outputs/ahm_masks"
    
    # Train/eval split ratio for ID data
    train_eval_split: float = 0.8
    
    # Silhouette score threshold for checkpoint filtering
    silhouette_threshold: float = 0.0  # Keep checkpoints with score > threshold
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ahm: AHMConfig = field(default_factory=AHMConfig)
    ahss: AHSSConfig = field(default_factory=AHSSConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    financedocs: FinanceDocsConfig = field(default_factory=FinanceDocsConfig)
    tobacco: Tobacco3482Config = field(default_factory=Tobacco3482Config)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_label2id(class_list: List[str]) -> Dict[str, int]:
    """Create label-to-id mapping from class list."""
    return {label: idx for idx, label in enumerate(sorted(class_list))}


def get_id2label(class_list: List[str]) -> Dict[int, str]:
    """Create id-to-label mapping from class list."""
    return {idx: label for idx, label in enumerate(sorted(class_list))}
