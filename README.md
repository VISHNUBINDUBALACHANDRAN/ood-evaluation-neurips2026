<div align="center">

# Auditing Attention Head Masking for Out-of-Distribution Detection: Cross-Architecture Failures and Polarity Inversions

**[Vishnu Bindu Balachandran](https://www.vishnubindubalachandran.com)**

*NeurIPS 2026 · Evaluations and Datasets Track*

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13)](https://openreview.net/forum?id=bqTe5C6VVH)
[![Checkpoints](https://zenodo.org/badge/DOI/10.5281/zenodo.23338711.svg)](https://doi.org/10.5281/zenodo.23338711)
[![Croissant](https://img.shields.io/badge/metadata-Croissant%201.1-orange)](croissant.json)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Results license](https://img.shields.io/badge/results-CC%20BY%204.0-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-3776AB)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C)](requirements.txt)

</div>

---

**TL;DR:** Attention-head-masking (AHM) methods for out-of-distribution (OOD) detection were evaluated on LayoutLMv3 for documents and, in a later journal version, on RoBERTa for text. Audited under one protocol on LayoutLMv3, ViT-B/16, RoBERTa-base and RoBERTa-large, they do not transfer reliably. The audit finds four architecture-dependent phenomena that single-modality benchmarks can miss.

This repository contains everything behind the paper:
- the code;
- the per-seed results of every reported run;
- the statistics;
- the near-OOD splits and attention-head masks;
- Croissant metadata.

The 90 fine-tuned checkpoints are on [Zenodo](https://doi.org/10.5281/zenodo.23338711).

## Key findings

| | Finding | Evidence |
|---|---|---|
| 1 | **MC-Dropout collapses on ViT-B/16.** ViT-B/16 has dropout 0.0, so its MC-Dropout score carries no signal. | AUROC 0.487–0.504 on all four ViT settings. Injecting dropout at inference restores 0.920–0.927. LayoutLMv3 (dropout 0.1) reaches 0.932. |
| 2 | **AHSS inverts polarity on RoBERTa-base.** OOD inputs move *less* under head masking than ID inputs. | Mean AUROC 0.296 on SST-2, against 0.874 on LayoutLMv3. With long ID inputs the inversion disappears (0.553), and at RoBERTa-large scale AHSS is at chance (0.540). |
| 3 | **Near-OOD degradation depends on modality, class and method.** | Mahalanobis drops 13.0 AUROC points on vision and 14.4 on text. On documents the drop depends on the held-out class: under 1 point for Letter vs Memo, up to about 10 points for the others. |
| 4 | **AHM distance scores need reference statistics fitted under the test-time masks, and the need grows with scale.** | On RoBERTa-large, mahAHM falls to 0.345 when its statistics are fitted on unmasked features. With the matched fit it reaches 0.897, on par with Mahalanobis (0.903). On RoBERTa-base the difference is small. |

**Head-to-head (15 seeds).** mahAvgAvg_AHM exceeds the uniform X-Mahalanobis baseline on two settings. Both results survive Bonferroni correction over 23 paired tests.
- **Tobacco-Cross: +0.0136.** The baseline is the same pipeline without masks, so this isolates the effect of the masks.
- **SST-2 → 20 Newsgroups: +0.0577.** This pair is separable by input length alone (AUROC 1.000), so the margin reflects pooling more than masking.

On standard CIFAR benchmarks X-Mahalanobis remains stronger. On CIFAR-100, the deficit of head-mask averaging grows monotonically with the number of classes, 8.1× from 10 to 100 classes.

## Quick start

```bash
git clone https://github.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026.git
cd ood-evaluation-neurips2026
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Recompute every statistic in the paper from the released per-seed results (no GPU, seconds)
python statistics/compute_paper_statistics.py
```

All per-seed values are in one table:

```python
import pandas as pd

df = pd.read_csv("statistics/per_seed_auroc.csv")            # 2,000 rows, each with its source file
main = df[df.study == "main"]
print(main.groupby(["configuration", "method"]).auroc.agg(["mean", "std"]))
```

## Reproducing the experiments

[REPRODUCTION.md](REPRODUCTION.md) maps every table and analysis to its command. In outline:

| Step | Where |
|---|---|
| Install the pinned environment | [`requirements.txt`](requirements.txt) |
| Download the datasets (not redistributed) | [DATA_ACCESS.md](DATA_ACCESS.md) |
| Optionally download the fine-tuned checkpoints and unzip them into `outputs/` | [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) |
| Run the experiments | [REPRODUCTION.md](REPRODUCTION.md) |

The seeds are 42, 123, 456, 789 and 1024. The two headline comparisons also use 7, 13, 31, 77, 101, 202, 303, 404, 505 and 2026.

## Setting

| Modality | Backbone | In-distribution | OOD sets |
|---|---|---|---|
| Documents | LayoutLMv3-base | FinanceDocs, Tobacco3482 | held-out class, cross-dataset; near-OOD: one Tobacco3482 class held out of fine-tuning |
| Images | ViT-B/16 | CIFAR-10, CIFAR-100 | held-out class, SVHN, DTD; near-OOD: CIFAR-100 small mammals |
| Text | RoBERTa-base, RoBERTa-large | SST-2 | MultiNLI, AG News, 20 Newsgroups; near-OOD: 20 Newsgroups comp.* |

The methods are the AHM-augmented distance methods (mahAHM, knnAHM, mahAvgAvg_AHM) of Constantinou et al. ([2024](https://arxiv.org/abs/2408.11237); [journal version, 2026](https://doi.org/10.1038/s41598-025-32328-9)), the AHSS score, which reads out the embedding shift under the same head masking (defined in the paper), and baselines including MSP, MaxLogit, Energy, KL, Mahalanobis, k-NN, X-Mahalanobis, ReAct, ASH, GradNorm, ViM, Residual, NeCo and MC-Dropout. Coverage varies by architecture; see the paper.

## Repository structure

```
├── run_*.py                    entry points: fine-tuning, feature extraction, OOD scoring, baselines, near-OOD
├── train_leave_class_out.py    LayoutLMv3 fine-tuning with one Tobacco3482 class held out
├── configs/ models/ ood_methods/ evaluation/ utils/    library code
├── scripts_rebuttal/           shell pipelines for the additional experiments
├── outputs/                    per-seed result files of the reported runs (see results_README.md)
├── statistics/                 paper statistics + table of all per-seed AUROC values, with their scripts
├── split_configs/              near-OOD splits (documents, vision, text) with exact members
├── random_head_masks/          random head masks of the ViT-B/16 and RoBERTa runs
├── manifests/                  checkpoint hashes, LayoutLMv3 configs, AHM-selected masks
├── verification_2026-10/       re-runs behind several reported values, with a pre-registered protocol
├── croissant.json              Croissant 1.1 metadata (with Responsible-AI fields)
└── *.md                        documentation (see below)
```

## Checkpoints

Zenodo holds 90 fine-tuned checkpoints (44.26 GB) of LayoutLMv3-base, ViT-B/16, RoBERTa-base and RoBERTa-large: **[10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711)**. [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) explains how to verify and load them.

## Citation

```bibtex
@inproceedings{balachandran2026auditing,
  title     = {Auditing Attention Head Masking for Out-of-Distribution Detection: Cross-Architecture Failures and Polarity Inversions},
  author    = {Balachandran, Vishnu Bindu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

For the checkpoints:

```bibtex
@dataset{balachandran2026checkpoints,
  title     = {Fine-tuned transformer checkpoints for a cross-architecture audit of attention-head-masking OOD detection (NeurIPS 2026)},
  author    = {Balachandran, Vishnu Bindu},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23338711}
}
```

## License

| Material | License |
|---|---|
| Code | Apache-2.0 |
| Documentation, configurations, split files, masks, results | CC BY 4.0 |
| Fine-tuned checkpoints: ViT-B/16, RoBERTa | CC BY 4.0 |
| Fine-tuned checkpoints: LayoutLMv3 | CC BY-NC-SA 4.0 (inherited from `microsoft/layoutlmv3-base`) |

See [LICENSE](LICENSE). The source datasets are not redistributed and keep their own licences ([DATA_ACCESS.md](DATA_ACCESS.md)).

## Acknowledgements

The AHM masks and the AHM-augmented distance methods evaluated here are due to Constantinou et al. (2024; journal version 2026). The evaluation builds on the public datasets and pre-trained models listed in [DATA_ACCESS.md](DATA_ACCESS.md). This work received no external funding.

## Documentation

[REPRODUCTION](REPRODUCTION.md) · [DATA_ACCESS](DATA_ACCESS.md) · [MODEL_CHECKPOINTS](MODEL_CHECKPOINTS.md) · [Result files](results_README.md) · [DATASHEET](DATASHEET.md) · [HOSTING_AND_MAINTENANCE](HOSTING_AND_MAINTENANCE.md) · [AUTHOR_STATEMENT](AUTHOR_STATEMENT.md) · [ETHICS](ETHICS.md) · [CHANGELOG](CHANGELOG.md) · [ERRATA](ERRATA.md)

## Contact

Questions and issues: please open a [GitHub issue](https://github.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026/issues) or e-mail vishnubindubalachandran@outlook.com.
