# Result Files

This repository includes the result files of the reported runs. The numbers in the paper can be checked without re-running training.

| Folder | Contents |
|---|---|
| `outputs/results/<config>_seed<seed>/` | LayoutLMv3-base: per-seed OOD results of all methods (`ood_results.json`) |
| `outputs/checkpoints/<config>_seed<seed>/` | LayoutLMv3-base: fine-tuning metrics and metadata (no weights; see [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md)) |
| `outputs/xmaha_results/` | X-Mahalanobis per seed. The paper uses the uniform variant (`xmaha_uniform`). |
| `outputs/vit_results/`, `outputs/cifar100_results/` | ViT-B/16 per seed: CIFAR-10, CIFAR-100 and the CIFAR-100 class-count subsets |
| `outputs/roberta_results/` | RoBERTa-base and RoBERTa-large per seed: SST-2 against MultiNLI, AG News and 20 Newsgroups |
| `outputs/nearood_polarity/` | Near-OOD vision (`vision_v2_seed*.json`) and text (`text_seed*.json`) per seed |
| `outputs/aggregated/` and the `aggregated/` sub-folders | Means and standard deviations over seeds |
| `verification_2026-10/results/` | Leave-class-out document near-OOD, the analyses with masked reference statistics, and the LayoutLMv3 MC-Dropout variance |

The seeds are 42, 123, 456, 789 and 1024. Tobacco-Cross and SST-2 vs 20 Newsgroups also have seeds 7, 13, 31, 77, 101, 202, 303, 404, 505 and 2026.

**All values in one table.** `statistics/per_seed_auroc.csv` lists the per-seed AUROC and FPR95 values of the reported studies (2,000 rows: the 11 standard configurations, the additional seeds, class count, RoBERTa-large and near-OOD) with the file each comes from. The MC-Dropout dropout-rate intervention and the hyperparameter ablation are in `outputs/mcdropout_mechanism/` and `outputs/ablation_results/`. `statistics/paper_statistics.json` / `.md` hold the statistics reported in the paper: paired tests with Bonferroni and Benjamini–Hochberg correction, the n = 15 comparisons, bootstrap intervals and uniform X-Mahalanobis values. The scripts in `statistics/` write both.

**Files not used for the reported numbers.**
- `outputs/statistical_tests/`: an earlier test table, replaced by `statistics/`.
- `outputs/FINAL_PAPER_TABLES.json` and the `xmaha_optimized` entries: optimised X-Mahalanobis, whose layer weights were tuned on the evaluation and OOD data.
- `outputs/near_ood_results/`: the original document near-OOD construction, where the OOD class was seen in fine-tuning. The reported values use the leave-class-out models.
- `outputs/nearood_polarity/vision_seed*.json`: see below.

**RoBERTa-base and RoBERTa-large.**
- RoBERTa-base results are in `outputs/roberta_results/sst2_{20news,agnews,mnli}_seed*` (25 runs).
- RoBERTa-large results are in `outputs/roberta_results/sst2_robertalarge_{20news,agnews,mnli}_seed*` (15 runs).
- The `architecture` field inside these files is a fixed string in `run_roberta_experiments.py` and reads "RoBERTa-base" for both. The folder name identifies the model.

**Near-OOD vision.** The paper reports `vision_v2_seed*.json`, which uses 600 OOD images per class. The files `vision_seed*.json` use the 100 test-split images per class only.

For the commands that produce these files, see [REPRODUCTION.md](REPRODUCTION.md).
