# Ethics Statement

This document expands on the broader-impacts answer in the paper checklist.

## 1. Nature of the Work
This is an **audit** of existing OOD-detection methods on existing public datasets. We do not collect new data, label new instances, or release new methods. We provide a unified evaluation protocol and report empirical findings (including MC-Dropout collapse on ViT-B/16, AHSS polarity inversion on RoBERTa-base, and near-OOD degradation that depends on modality, class and method).

## 2. Source-Dataset Ethics
- **Image** (CIFAR-10, CIFAR-100, SVHN, DTD): Public, non-personal, no privacy concerns. SVHN images are house-number crops from Google Street View; the original dataset publishers obtained appropriate permissions.
- **Text** (SST-2, MultiNLI, AG News, 20 Newsgroups): Public text corpora. 20 Newsgroups contains 1990s Usenet posts that may include offensive language reflective of that period — see [DATASHEET.md](DATASHEET.md) Q15.
- **Document** (FinanceDocs, Tobacco-3482): Public document corpora. Tobacco-3482 is sourced from publicly released tobacco-industry litigation documents and may incidentally contain employee names — see [DATASHEET.md](DATASHEET.md) Q15–Q16.

We do not redistribute any source dataset; users download from canonical sources under each source's original license.

## 3. PII / Sensitive Data
No personally identifiable information (PII), no biometric data, no medical records, no financial accounts. The Tobacco litigation documents may incidentally contain names of employees of tobacco companies, but these documents are public-record litigation exhibits.

## 4. Potential for Misuse
**Limited and indirect.** OOD detection is dual-use technology — it can be used to flag inputs outside a model's training distribution (positive use: safety in deployment; negative use: filtering input populations in ways that may compound bias). Our audit does not produce a new deployable detector; it characterizes existing methods. We document failure modes (Section 6) precisely so that practitioners do not over-trust OOD scores from architectures or methods where they collapse.

## 5. Bias and Fairness Considerations
- The text datasets (SST-2, MultiNLI, AG News, 20 Newsgroups) reflect the demographic and topical biases of their original sources (movie reviews, multi-genre crowdsourced text, news, Usenet).
- The image datasets (CIFAR, SVHN, DTD) are object/digit/texture classification and do not include faces or demographic attributes.
- The document datasets (FinanceDocs, Tobacco) reflect the linguistic and structural conventions of US-English business documents.
- We do not evaluate fairness of OOD scores across demographic subgroups, because none of the source datasets carry the requisite demographic labels. This is a limitation of the audit.

## 6. Reproducibility
All experiments are reproducible with the hyperparameters and random seeds (42, 123, 456, 789, 1024) documented in the code and in the per-seed result files under `outputs/`. The multi-seed protocol (5 seeds) is the minimum needed for paired Wilcoxon significance tests.

## 7. User Impact
Researchers using this audit should:
- Cite the **original method papers** (especially Constantinou et al., arXiv:2408.11237 and Scientific Reports 16:2449, 2026, for the AHM masks, mahAHM, knnAHM and mahAvgAvg_AHM) when comparing against these methods.
- Report seed-level numbers (not just aggregates) when extending the audit.
- Note the architecture-specific failure modes documented in Section 6 — particularly that AHSS polarity inverts on RoBERTa-base and that MC-Dropout collapses across all four ViT-B/16 configurations (the RoBERTa SST-2 → 20-Newsgroups result is reported under the conventional MC-Dropout orientation but flagged as consistent with a polarity issue rather than complete collapse).

## 8. Children / Vulnerable Populations
None of the source datasets target children or vulnerable populations.

## 9. Compliance with NeurIPS Code of Ethics
We confirm compliance with the NeurIPS Code of Ethics. The release includes a datasheet, licence declarations, a hosting and maintenance plan, and Croissant metadata (`croissant.json`).
