# Reproduction Guide

This document maps the results in the paper to the commands that produce them. Per-seed result files are included under `outputs/` and `verification_2026-10/results/` (see [results_README.md](results_README.md)), so the reported numbers can be checked without re-running training. All commands are run from the repository root.

## 1. Environment
```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The reported runs used one 80 GB GPU (CUDA 12.9 driver, PyTorch 2.5.1 CUDA 12.1 build). The LayoutLMv3 leave-class-out checkpoints were trained with `transformers==4.28.0` (see `requirements.txt`).

## 2. Data and checkpoints
- **Datasets:** see [DATA_ACCESS.md](DATA_ACCESS.md). The scripts read them from `data/`. The LayoutLMv3 inputs are prepared with `run_prepare_data.py`.
- **Fine-tuned checkpoints (optional):**
  1. Download them from Zenodo (DOI [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711)) and verify them as described in [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md).
  2. Unzip every archive into `outputs/`, for example `unzip -q <archive>.zip -d outputs/`.

  The archives contain `checkpoints/`, `vit_checkpoints/`, `cifar100_checkpoints/`, `roberta_checkpoints/` and `nearood_polarity/`, which are the paths the scripts use. Fine-tuning is then skipped for every run whose checkpoint is present.

## 3. Statistics from the released results (no GPU, seconds)
```bash
python statistics/compute_paper_statistics.py   # 23 paired tests with Bonferroni / Benjamini-Hochberg, n = 15 comparisons,
                                                 # bootstrap CIs, headroom check, uniform X-Mahalanobis values
python statistics/export_per_seed_auroc.py      # statistics/per_seed_auroc.csv (2,000 per-seed values with their source files)
```

## 4. Main results (seeds 42, 123, 456, 789, 1024)

### LayoutLMv3-base: FinanceDocs and Tobacco3482 (Table 1)
These stages run per configuration: fine-tuning, feature extraction, AHM mask selection, and OOD scoring of all methods.

Set the paths for the configuration you want:
- **FinanceDocs:** `DATA=./data/colated_tensors`.
- **Tobacco3482:** `DATA=./data/tobacco/colated_tensors`.
- **Cross-dataset configurations:** `OOD` is the other dataset's folder.

```bash
python run_experiments.py --dataset tobacco --experiment_type cross \
    --data_dir ./data/tobacco/colated_tensors --ood_data_dir ./data/colated_tensors \
    --seeds 42 123 456 789 1024 --output_dir ./outputs
python run_phase1_all_seeds.py                         # AHSS and MC-Dropout for the four document configurations
python run_xmaha_baseline.py --architecture layoutlmv3 --run_all
```
`run_experiments.py` chains `run_finetune.py`, `run_extract_embeddings.py`, `run_ahm_optimization.py` and `run_ood_evaluation.py`. `scripts_rebuttal/run_rebuttal_pipeline.sh` shows the individual commands with all arguments.

### ViT-B/16: CIFAR-10 and CIFAR-100 (Table 2)
```bash
python run_vit_experiments.py --run_all                # CIFAR-10 intra and CIFAR-10 vs SVHN (incl. MC-Dropout)
python run_vit_extended_eval.py --run_all              # additional baselines on the CIFAR-10 configurations
python run_cifar100_benchmark.py --run_all             # CIFAR-100 vs SVHN and vs DTD
python run_xmaha_baseline.py --architecture vit --run_all
python run_mahavgavg_ahm_crossarch.py --run_all        # mahAvgAvg_AHM for ViT-B/16 and RoBERTa-base
```

### RoBERTa-base: SST-2 against MultiNLI, AG News and 20 Newsgroups (Table 3)
```bash
python run_roberta_experiments.py --run_all
```

## 5. Additional experiments
| Result | Command |
|---|---|
| n = 15 headline comparisons (Tobacco-Cross; SST-2 vs 20 Newsgroups): seeds 7, 13, 31, 77, 101, 202, 303, 404, 505, 2026 | `bash scripts_rebuttal/run_rebuttal_pipeline.sh` |
| RoBERTa-large | `python run_roberta_experiments.py --ood_source {mnli,agnews,20news} --seed S --model_name roberta-large` |
| CIFAR-100 class-count subsets | `python run_cifar100_benchmark.py --ood_source svhn --seed S --num_classes {10,20,50}` and `python run_mahavgavg_ahm_crossarch.py --architecture vit_cifar100 --ood_source svhn --seed S --num_classes N` |
| Near-OOD, vision | `python run_nearood_polarity.py --arch vision --seeds S --epochs_vit 5 --n_eval 800 --n_masks 25 --bs 128 --out outputs/nearood_polarity/vision_v2_seedS.json` |
| Near-OOD, text, and the AHSS input-length analysis | `python run_nearood_polarity.py --arch text --seeds S --epochs_txt 4 --n_eval 800 --n_masks 25 --bs_txt 32 --max_len 256 --out outputs/nearood_polarity/text_seedS.json` |
| Near-OOD, documents (leave-class-out) | `python train_leave_class_out.py --holdout_class {Memo,Form,Resume,Scientific} --seed S --data ./data/tobacco/colated_tensors`, then `python verification_2026-10/code/e1_doc_nearood.py lco Memo Form Resume Scientific` and `… ahss` |
| MC-Dropout dropout-rate intervention | `python run_mcdropout_mechanism.py --arch vit`; LayoutLMv3 with the fine-tuned weights: `python verification_2026-10/code/e4_mcdropout_layoutlmv3.py 42 123 456 789 1024` |
| Reference statistics fitted on masked features | `python verification_2026-10/code/e2_roberta.py {base,large} 42 123 456 789 1024` and `python verification_2026-10/code/e5_vit.py <configs> <seeds>` (see `verification_2026-10/code/README.md`) |
| Hyperparameter ablation (appendix) | `python run_ablation_studies.py --ablation all --dataset {financedocs,tobacco} --seed 42` |

The shell scripts in `scripts_rebuttal/` run these experiments in batches. The split files in `split_configs/` and the masks in `random_head_masks/` are regenerated by the scripts in those folders. `croissant.json` is regenerated with `python croissant/make_croissant.py`.

## 6. Aggregation
```bash
python run_aggregate_results.py          # summary tables from the per-seed results -> outputs/aggregated/ (add --latex for LaTeX tables)
python run_final_aggregate.py            # combined summary across architectures
python aggregate_rebuttal_results.py     # summary of the additional experiments
```

## 7. Notes on reproducibility
- Fine-tuning and scoring are seeded with Python, NumPy and PyTorch, using the seeds above. GPU kernels are not fully deterministic, so re-runs can differ slightly.
- A re-run with the same software versions reproduced the stored embeddings and per-seed AUROC values within the tolerances reported in `verification_2026-10/results/SUMMARY.md`.
- Pre-trained backbones are loaded from the Hugging Face Hub: `microsoft/layoutlmv3-base`, `google/vit-base-patch16-224`, `roberta-base`, `roberta-large`.
