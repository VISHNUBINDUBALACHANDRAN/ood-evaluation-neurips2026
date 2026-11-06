# Changelog

All notable changes to this release are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-11-06

### Added
- Code for fine-tuning and post-hoc OOD scoring on:
  - LayoutLMv3-base (FinanceDocs, Tobacco3482);
  - ViT-B/16 (CIFAR-10, CIFAR-100);
  - RoBERTa-base and RoBERTa-large (SST-2 against MultiNLI, AG News and 20 Newsgroups).

  It covers the AHM-augmented distance methods (mahAHM, knnAHM, mahAvgAvg_AHM) of Constantinou et al. (arXiv:2408.11237; Scientific Reports 16:2449, 2026), the AHSS score defined in the paper, and the baseline methods.
- Per-seed result files of the reported runs. The seeds are 42, 123, 456, 789 and 1024, plus ten further seeds for the two headline comparisons.
- Near-OOD split configurations (documents, vision, text) and the attention-head masks.
- The statistics script, a table of all per-seed AUROC values, and Croissant metadata (`croissant.json`).
- Fine-tuned checkpoints on Zenodo (DOI [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711)).
- Datasheet, reproduction guide, and hosting and maintenance plan.
- Dual licence: Apache-2.0 for code and CC-BY-4.0 for documentation, configurations and outputs.

### Notes
- Source datasets are not redistributed. See [DATA_ACCESS.md](DATA_ACCESS.md).
