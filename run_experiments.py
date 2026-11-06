"""
run_experiments.py — Master experiment orchestration script.

Runs the complete AHM paper reproduction pipeline for all 4 experimental
configurations, each with 5 random seeds:

    1. FinanceDocs intra-dataset (OOD = Resumes)
    2. FinanceDocs cross-dataset (OOD = Tobacco3482)
    3. Tobacco3482 intra-dataset (OOD = ADVE)
    4. Tobacco3482 cross-dataset (OOD = FinanceDocs)

Pipeline per seed:
    run_finetune → run_extract_embeddings → run_ahm_optimization → run_ood_evaluation

After all seeds finish, aggregates results (mean ± std) and produces
Table 1 from the paper.

Usage:
    # Run all experiments
    python run_experiments.py --data_dir ./colated_tensors --output_dir ./outputs

    # Run a single experiment
    python run_experiments.py \
        --data_dir ./colated_tensors \
        --output_dir ./outputs \
        --dataset financedocs \
        --experiment_type intra \
        --seeds 42 1337 2024

    # Resume from a specific stage
    python run_experiments.py \
        --data_dir ./colated_tensors \
        --output_dir ./outputs \
        --start_stage extract
"""

import argparse
import os
import sys
import subprocess
import json
import logging
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import get_default_config
from evaluation.metrics import aggregate_results, format_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = ["finetune", "extract", "ahm", "evaluate"]


def run_command(cmd, description=""):
    """Run a subprocess command and handle errors."""
    logger.info(f"{'=' * 60}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"CMD: {' '.join(cmd)}")
    logger.info(f"{'=' * 60}")
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    elapsed = time.time() - start_time
    
    if result.stdout:
        logger.info(result.stdout[-2000:])  # Last 2000 chars
    
    if result.returncode != 0:
        logger.error(f"FAILED after {elapsed:.1f}s: {description}")
        logger.error(result.stderr[-2000:])
        return False
    
    logger.info(f"COMPLETED in {elapsed:.1f}s: {description}")
    return True


def run_finetune_stage(data_path, output_dir, dataset, experiment_type, seed):
    """Run model fine-tuning for a single seed."""
    model_dir = os.path.join(
        output_dir, "checkpoints",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    
    # Skip if best model already exists
    if os.path.exists(os.path.join(model_dir, "best_model", "config.json")):
        logger.info(f"Skipping finetune (already done): {model_dir}")
        return True
    
    cmd = [
        PYTHON, os.path.join(BASE_DIR, "run_finetune.py"),
        "--data_path", data_path,
        "--output_dir", model_dir,
        "--dataset", dataset,
        "--experiment_type", experiment_type,
        "--seed", str(seed),
    ]
    
    return run_command(cmd, f"Finetuning {dataset}/{experiment_type} seed={seed}")


def run_extract_stage(data_path, model_path, output_dir, dataset, experiment_type, seed,
                      ood_data_path=None):
    """Run embedding extraction for a single seed."""
    emb_dir = os.path.join(
        output_dir, "embeddings",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    
    # Skip if embeddings already exist
    if os.path.exists(os.path.join(emb_dir, "eval_embeddings.npy")):
        logger.info(f"Skipping extraction (already done): {emb_dir}")
        return True
    
    cmd = [
        PYTHON, os.path.join(BASE_DIR, "run_extract_embeddings.py"),
        "--model_path", model_path,
        "--data_path", data_path,
        "--output_dir", emb_dir,
        "--dataset", dataset,
        "--experiment_type", experiment_type,
        "--seed", str(seed),
    ]
    
    if ood_data_path:
        cmd.extend(["--ood_data_path", ood_data_path])
    
    return run_command(cmd, f"Extracting embeddings {dataset}/{experiment_type} seed={seed}")


def run_ahm_stage(data_path, model_path, output_dir, dataset, experiment_type, seed,
                  ood_data_path=None):
    """Run AHM optimization for a single seed."""
    ahm_dir = os.path.join(
        output_dir, "ahm_masks",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    
    # Skip if AHM embeddings already exist
    if os.path.exists(os.path.join(ahm_dir, "train_ahm_embeddings.npy")):
        logger.info(f"Skipping AHM (already done): {ahm_dir}")
        return True
    
    cmd = [
        PYTHON, os.path.join(BASE_DIR, "run_ahm_optimization.py"),
        "--model_path", model_path,
        "--data_path", data_path,
        "--output_dir", ahm_dir,
        "--dataset", dataset,
        "--experiment_type", experiment_type,
        "--seed", str(seed),
    ]
    
    if ood_data_path:
        cmd.extend(["--ood_data_path", ood_data_path])
    
    return run_command(cmd, f"AHM optimization {dataset}/{experiment_type} seed={seed}")


def run_evaluate_stage(output_dir, dataset, experiment_type, seed,
                       model_path=None, data_path=None, ood_data_path=None):
    """Run OOD evaluation for a single seed."""
    emb_dir = os.path.join(
        output_dir, "embeddings",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    ahm_dir = os.path.join(
        output_dir, "ahm_masks",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    results_dir = os.path.join(
        output_dir, "results",
        f"{dataset}_{experiment_type}_seed{seed}"
    )
    
    cmd = [
        PYTHON, os.path.join(BASE_DIR, "run_ood_evaluation.py"),
        "--embedding_dir", emb_dir,
        "--output_dir", results_dir,
        "--dataset", dataset,
        "--experiment_type", experiment_type,
        "--seed", str(seed),
    ]
    
    if os.path.isdir(ahm_dir):
        cmd.extend(["--ahm_dir", ahm_dir])
    
    if model_path:
        cmd.extend(["--model_path", model_path])
    if data_path:
        cmd.extend(["--data_path", data_path])
    if ood_data_path:
        cmd.extend(["--ood_data_path", ood_data_path])
    
    return run_command(cmd, f"OOD evaluation {dataset}/{experiment_type} seed={seed}")


def aggregate_all_results(output_dir, dataset, experiment_type, seeds):
    """Aggregate results across seeds and produce final table."""
    all_seed_results = []
    
    for seed in seeds:
        results_path = os.path.join(
            output_dir, "results",
            f"{dataset}_{experiment_type}_seed{seed}",
            "ood_results.json"
        )
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                seed_results = json.load(f)
                all_seed_results.append(seed_results)
        else:
            logger.warning(f"Results not found for seed {seed}: {results_path}")
    
    if not all_seed_results:
        logger.error(f"No results to aggregate for {dataset}/{experiment_type}")
        return
    
    # Flatten: all_seed_results is List[List[Dict]], aggregate_results wants List[Dict]
    flat_results = [r for seed_results in all_seed_results for r in seed_results]
    
    # Aggregate by method
    aggregated = aggregate_results(flat_results)
    
    # Format table
    table = format_results_table(
        aggregated,
        dataset_name=dataset.capitalize(),
        experiment_type=experiment_type,
    )
    
    # Save aggregated results
    agg_dir = os.path.join(output_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    
    agg_path = os.path.join(agg_dir, f"{dataset}_{experiment_type}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    table_path = os.path.join(agg_dir, f"{dataset}_{experiment_type}_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"AGGREGATED RESULTS: {dataset}/{experiment_type} ({len(all_seed_results)} seeds)")
    logger.info(f"{'=' * 70}")
    logger.info(f"\n{table}")
    
    return aggregated


def run_experiment(data_path, output_dir, dataset, experiment_type, seeds,
                   start_stage="finetune", ood_data_path=None):
    """Run full experiment pipeline for one dataset/experiment configuration."""
    
    config = get_default_config()
    start_idx = STAGES.index(start_stage) if start_stage in STAGES else 0
    
    for seed in seeds:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"EXPERIMENT: {dataset}/{experiment_type} | SEED: {seed}")
        logger.info(f"{'#' * 70}")
        
        # Paths
        model_dir = os.path.join(
            output_dir, "checkpoints",
            f"{dataset}_{experiment_type}_seed{seed}"
        )
        model_path = os.path.join(model_dir, "best_model")
        
        # Stage 1: Finetune
        if start_idx <= 0:
            success = run_finetune_stage(
                data_path, output_dir, dataset, experiment_type, seed
            )
            if not success:
                logger.error(f"Finetune failed for seed {seed}, skipping...")
                continue
        
        # Stage 2: Extract embeddings
        if start_idx <= 1:
            success = run_extract_stage(
                data_path, model_path, output_dir,
                dataset, experiment_type, seed,
                ood_data_path=ood_data_path,
            )
            if not success:
                logger.error(f"Extraction failed for seed {seed}, skipping...")
                continue
        
        # Stage 3: AHM optimization
        if start_idx <= 2:
            success = run_ahm_stage(
                data_path, model_path, output_dir,
                dataset, experiment_type, seed,
                ood_data_path=ood_data_path,
            )
            if not success:
                logger.error(f"AHM failed for seed {seed}, skipping...")
                continue
        
        # Stage 4: OOD evaluation
        if start_idx <= 3:
            success = run_evaluate_stage(
                output_dir, dataset, experiment_type, seed,
                model_path=model_path,
                data_path=data_path,
                ood_data_path=ood_data_path,
            )
            if not success:
                logger.error(f"Evaluation failed for seed {seed}")
                continue
    
    # Aggregate across seeds
    aggregate_all_results(output_dir, dataset, experiment_type, seeds)


def main():
    parser = argparse.ArgumentParser(
        description="Master experiment orchestration — reproduce AHM paper results"
    )
    parser.add_argument("--data_dir", type=str, default="./colated_tensors",
                        help="Base directory for preprocessed data")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["financedocs", "tobacco"],
                        help="Run only this dataset (default: both)")
    parser.add_argument("--experiment_type", type=str, default=None,
                        choices=["intra", "cross"],
                        help="Run only this experiment type (default: both)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: [42, 123, 456, 789, 1024])")
    parser.add_argument("--start_stage", type=str, default="finetune",
                        choices=STAGES,
                        help="Resume from this stage (skip earlier stages)")
    parser.add_argument("--ood_data_dir", type=str, default=None,
                        help="Directory with OOD dataset for cross-dataset experiments")
    
    args = parser.parse_args()
    config = get_default_config()
    
    seeds = args.seeds or config.seeds
    
    # Determine which experiments to run
    datasets = [args.dataset] if args.dataset else ["financedocs", "tobacco"]
    exp_types = [args.experiment_type] if args.experiment_type else ["intra", "cross"]
    
    total_start = time.time()
    
    logger.info(f"{'*' * 70}")
    logger.info("AHM Paper Reproduction — Full Experiment Pipeline")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Experiment types: {exp_types}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Start stage: {args.start_stage}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"{'*' * 70}")
    
    for dataset in datasets:
        for exp_type in exp_types:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"STARTING: {dataset} / {exp_type}")
            logger.info(f"{'=' * 70}")
            
            # Determine data path
            data_path = args.data_dir
            ood_data_path = args.ood_data_dir if exp_type == "cross" else None
            
            run_experiment(
                data_path=data_path,
                output_dir=args.output_dir,
                dataset=dataset,
                experiment_type=exp_type,
                seeds=seeds,
                start_stage=args.start_stage,
                ood_data_path=ood_data_path,
            )
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'*' * 70}")
    logger.info(f"ALL EXPERIMENTS COMPLETED in {total_elapsed / 3600:.1f} hours")
    logger.info(f"{'*' * 70}")


if __name__ == "__main__":
    main()
