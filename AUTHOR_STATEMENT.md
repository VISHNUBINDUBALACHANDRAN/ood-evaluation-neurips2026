# Author Statement of Responsibility

The author of *Auditing Attention Head Masking for Out-of-Distribution Detection: Cross-Architecture Failures and Polarity Inversions*, Vishnu Bindu Balachandran, makes the following declarations about the released artifacts.

## 1. Original contribution
The audit is the author's own work. It comprises:
- the (ID, OOD) configurations and near-OOD splits;
- the unified evaluation protocol with fixed seeds;
- the paired significance analysis;
- the aggregation code.

The author does **not** claim ownership of:
- The **methods** evaluated. The AHM masks and the AHM-augmented distance methods (mahAHM, knnAHM, mahAvgAvg_AHM) are due to **Constantinou et al. (arXiv:2408.11237; Scientific Reports 16:2449, 2026)**. AHSS, defined in the paper, is a direct readout of the hypothesis behind them rather than a new method. All baseline methods are attributed to their original publications and re-implemented under the unified protocol.
- The **source datasets**. Each is cited and used under its original licence (see [DATA_ACCESS.md](DATA_ACCESS.md)).
- The **pre-trained backbone weights** (LayoutLMv3-base, ViT-B/16, RoBERTa-base, RoBERTa-large). These are obtained from the Hugging Face Hub under their respective licences.

## 2. Acceptance of responsibility
The author accepts full responsibility for any rights violations or misuse arising from the released artifacts. Reported issues will be answered through the channels listed in [HOSTING_AND_MAINTENANCE.md](HOSTING_AND_MAINTENANCE.md).

## 3. Public source datasets
All source datasets are publicly released and used under their original licences: CIFAR-10, CIFAR-100, SVHN, DTD, SST-2, MultiNLI, AG News, 20 Newsgroups, FinanceDocs and Tobacco3482. They are **not** redistributed here. [DATA_ACCESS.md](DATA_ACCESS.md) gives their sources.

## 4. Licences
- **Code:** Apache License 2.0.
- **Documentation, split configurations, evaluation outputs, attention-head masks:** CC-BY-4.0.
- **ViT-B/16, RoBERTa-base and RoBERTa-large fine-tuned checkpoints** (Zenodo): CC-BY-4.0.
- **LayoutLMv3-base fine-tuned checkpoints** (Zenodo): **CC-BY-NC-SA-4.0** (non-commercial, share-alike). This licence is inherited from the upstream `microsoft/layoutlmv3-base` licence and propagated under share-alike.
- See [LICENSE](LICENSE) and [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) for details.

## 5. Fine-tuned checkpoints
The fine-tuned checkpoints of the runs reported in the paper with seeds 42, 123, 456, 789 and 1024 are archived on Zenodo, DOI [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711). [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) lists them and explains how to download and verify them.

## 6. Author and contact
Vishnu Bindu Balachandran, Independent AI Researcher.
- Website: https://www.vishnubindubalachandran.com
- E-mail: vishnubindubalachandran@outlook.com

## 7. Maintenance commitment
The author commits, on a best-effort basis, to:
- monitoring the public issue tracker for at least 24 months after publication;
- working towards resolving reproducibility issues within 30 days where feasible;
- versioning all releases (semantic versioning) and archiving each release on Zenodo with its own DOI;
- publishing corrections to reported numbers or methodology in a public `ERRATA.md` file in the code repository.

No institutional backing, multi-year funding or third-party transfer agreement is claimed. See [HOSTING_AND_MAINTENANCE.md](HOSTING_AND_MAINTENANCE.md) for the scope of these commitments.

## 8. Competing interests
The author declares no competing interests. This includes any relationship with the original AHM authors (Constantinou et al.) or with the maintainers of any source dataset or pre-trained backbone used in this audit.
