# Datasheet for the AHM Cross-Architecture OOD Audit Suite

This datasheet accompanies the audit paper *Auditing Attention Head Masking for Out-of-Distribution Detection: Cross-Architecture Failures and Polarity Inversions*. It follows the *Datasheets for Datasets* template (Gebru et al., 2018, arXiv:1803.09010), adapted for an **audit/benchmark suite** that re-uses pre-existing public datasets.

> **Important:** This work does not introduce a new dataset or a new method. It evaluates post-hoc OOD-detection methods on four fine-tuned transformers: LayoutLMv3-base, ViT-B/16, RoBERTa-base and RoBERTa-large.
> - The methods are 16 method groups, including the AHM-augmented distance methods of Constantinou et al. (arXiv:2408.11237; Scientific Reports 16:2449, 2026).
> - The configurations are eleven standard (ID, OOD) configurations plus near-OOD configurations in three modalities.
> - Every configuration uses a unified protocol with five fixed seeds.
> - The paired comparisons are tested with Wilcoxon signed-rank tests, with Bonferroni and Benjamini–Hochberg correction over the 23 tests.

---

## Motivation

### Q1. For what purpose was the audit suite created?
To provide a **reproducible cross-architecture, cross-modality evaluation protocol** for post-hoc OOD-detection methods on transformer backbones. Existing OOD benchmarks (OpenOOD, OOD-Bench) cover a single modality (vision). This suite unifies eleven standard (ID, OOD) configurations under a single protocol. It adds near-OOD configurations in three modalities:
- Tobacco3482 with one class held out of fine-tuning;
- CIFAR-100 small mammals;
- 20 Newsgroups comp.*.

### Q2. Who created this audit suite?
Vishnu Bindu Balachandran (Independent AI Researcher), the author of the audit paper.

### Q3. Who funded its creation?
No external funding. The work was carried out and paid for personally by the author; there was no sponsorship of any kind.

---

## Composition

### Q4. What do the instances represent?
Instances are **(input, label)** pairs drawn from ten public source datasets. The eleven standard configurations are:

| # | Config | ID dataset | OOD dataset | Modality | Architecture |
|---|--------|-----------|-------------|----------|--------------|
| 1 | `financedocs_intra` | FinanceDocs (9 classes) | FinanceDocs (held-out class) | Document | LayoutLMv3-base |
| 2 | `financedocs_cross` | FinanceDocs | Tobacco3482 | Document | LayoutLMv3-base |
| 3 | `tobacco_intra` | Tobacco3482 (9 classes) | Tobacco3482 (held-out class Advertisement) | Document | LayoutLMv3-base |
| 4 | `tobacco_cross` | Tobacco3482 | FinanceDocs | Document | LayoutLMv3-base |
| 5 | `cifar10_intra` | CIFAR-10 (9 classes) | CIFAR-10 (held-out class) | Image | ViT-B/16 |
| 6 | `cifar10_svhn` | CIFAR-10 | SVHN | Image | ViT-B/16 |
| 7 | `cifar100_svhn` | CIFAR-100 | SVHN | Image | ViT-B/16 |
| 8 | `cifar100_dtd` | CIFAR-100 | DTD | Image | ViT-B/16 |
| 9 | `sst2_mnli` | SST-2 | MultiNLI | Text | RoBERTa-base |
| 10 | `sst2_agnews` | SST-2 | AG News | Text | RoBERTa-base |
| 11 | `sst2_20news` | SST-2 | 20 Newsgroups | Text | RoBERTa-base |

Further settings:
- **RoBERTa-large:** configurations 9–11.
- **CIFAR-100 class-count subsets:** 10, 20 and 50 classes, against SVHN.
- **Near-OOD:**
  - Tobacco3482 with Memo, Form, Resume or Scientific held out of fine-tuning and used as OOD;
  - ViT-B/16 on four CIFAR-100 small mammals, with near-OOD squirrel and far-OOD rocket;
  - RoBERTa-base on 20 Newsgroups comp.*, with near-OOD sci.electronics and far-OOD talk.politics.mideast.

### Q5. How many instances are used?
- **FinanceDocs, train / eval / OOD:** intra 2,008 / 503 / 287; cross 2,238 / 560 / 3,482 (all Tobacco3482).
- **Tobacco3482, train / eval / OOD:** intra 2,601 / 651 / 230; cross 2,785 / 697 / 2,798 (all FinanceDocs).
- **CIFAR-10:** intra 40,500 / 4,500 / 1,000 (held-out class, test split). With all classes: 45,000 / 5,000.
- **CIFAR-100:** 45,000 / 5,000.
- **SVHN** (OOD): the first 10,000 test images.
- **DTD** (OOD): the test split (1,880 images).
- **SST-2:** the 67,349 GLUE training sentences, split 90/10 into 60,614 / 6,735.
- **MultiNLI, AG News, 20 Newsgroups** (OOD): 5,000 texts each (see Q19).
- **Near-OOD:** sizes per split are listed in `split_configs/`.

### Q6. Does the suite contain all possible instances, or is it a sample?
The vision and document datasets are used in full, except SVHN (first 10,000 test images). The text OOD sets are 5,000-text samples (see Q19). `split_configs/` lists the exact members of every near-OOD split.

### Q7. What data does each instance consist of?
- **Image:** 224×224 RGB tensor and an integer class label.
- **Text:** tokenized sequence (RoBERTa BPE, 128 tokens; 256 in the near-OOD text setting) and an integer label.
- **Document:** 224×224 RGB page image, OCR tokens with 2-D bounding boxes, and a document-level class label.

### Q8. Is there a label or target associated with each instance?
Yes. ID instances carry their original class label; OOD instances are scored without a class label.

### Q9. Is any information missing from individual instances?
For the document datasets (FinanceDocs, Tobacco3482), some pages have low OCR quality. They are kept to reflect realistic conditions.

### Q10. Are relationships between individual instances made explicit?
No. Instances are treated as independent for OOD scoring.

### Q11. Are there recommended data splits?
**Yes.** The train/eval splits depend on the seed (42, 123, 456, 789, 1024):
- documents: stratified 80/20 split with `random_state = seed`;
- CIFAR and SST-2: seeded 90/10 split.

The same seed also controls fine-tuning and the random head masks. `split_configs/` lists the near-OOD splits.

### Q12. Are there errors, sources of noise, or redundancies?
Source datasets carry their known issues: CIFAR-10 label noise, SVHN cropping artifacts, MNLI annotation disagreement and 20 Newsgroups Usenet-era language. The source datasets are not re-cleaned, to preserve comparability with prior OOD literature.

### Q13. Is the suite self-contained?
**No. It links to external resources.**
- All source datasets are downloaded from their canonical hosts: the Hugging Face Hub, torchvision, scikit-learn and the official dataset URLs. See [DATA_ACCESS.md](DATA_ACCESS.md).
- The pre-trained weights (LayoutLMv3-base, ViT-B/16, RoBERTa-base, RoBERTa-large) come from the Hugging Face Hub.

### Q14. Does the suite contain confidential data?
No. All sources are publicly released datasets.

### Q15. Does the suite contain offensive, insulting, or threatening data?
- **20 Newsgroups** contains historical Usenet posts that may include offensive language reflective of 1990s internet culture.
- **MultiNLI** is sourced from diverse genres, including 9/11 reports and government documents.
- **Tobacco3482** is sourced from publicly released tobacco-industry litigation documents.
- These source datasets are not modified; users should be aware of the original content.

### Q16. Does the suite relate to people?
- **MultiNLI, SST-2 and 20 Newsgroups** contain text written by people; all sources are publicly distributed.
- **No PII** is intentionally included. Tobacco3482 may incidentally contain employee names from publicly released litigation documents.

---

## Collection Process

### Q17. How was the data acquired?
No new data was collected. Each source dataset was downloaded from its canonical public host. [DATA_ACCESS.md](DATA_ACCESS.md) lists the URLs and licences.

### Q18. What mechanisms were used to collect the data?
N/A. There was no new collection.

### Q19. If sampling, what was the strategy?
- **MultiNLI:** the first 5,000 premises of the GLUE `validation_matched` split.
- **AG News:** the first 5,000 test texts.
- **20 Newsgroups:** the first 5,000 documents of `sklearn.datasets.fetch_20newsgroups(subset='test')`, with its default shuffle, headers kept, and each text truncated to 512 characters.
- **Near-OOD splits:** seeded selections, described with their exact members in `split_configs/`.

### Q20. Who was involved in data collection?
N/A.

### Q21. Over what timeframe was the data collected?
The source datasets were originally collected between 1993 (20 Newsgroups) and 2024 (FinanceDocs). The audit was carried out in 2026.

### Q22. Were ethical-review processes conducted?
The original source-dataset publications include their own ethics review, and this audit relies on those. The composition was additionally checked for offensive content (Q15).

---

## Preprocessing / Cleaning / Labeling

### Q23. Was any preprocessing done?
- **Image:** resize to 224×224 and normalisation with ImageNet mean and standard deviation.
- **Text:** RoBERTa BPE tokenization, with truncation and padding to 128 tokens (256 in the near-OOD text setting).
- **Document:** Tesseract OCR for tokens and bounding boxes through the LayoutLMv3 processor (`run_prepare_data.py`).
- **No instance-level relabeling.**

### Q24. Was the "raw" data saved?
The audit suite does not redistribute raw data; users download it from the canonical sources.

### Q25. Is the preprocessing software available?
Yes: `run_prepare_data.py` and `utils/data.py` (Apache-2.0).

---

## Uses

### Q26. Has the suite been used for any tasks already?
- The cross-architecture OOD audit reported in the paper.
- Reproduction of AHM (Constantinou et al., 2024) under multi-seed evaluation.
- Evaluation of X-Mahalanobis (NeurIPS 2025) on document and text modalities.

### Q27. Is there a repository linking to all uses?
Downstream uses can be reported in the issue tracker of the public code repository.

### Q28. What other tasks could the suite be used for?
- Calibration evaluation across modalities.
- Architecture-conditional ablation of OOD-detection methods.
- Robustness-vs-OOD trade-off studies.
- Mechanistic interpretability of attention-head specialization across architectures.

### Q29. Is there anything about the composition that might affect future uses?
- The intra-dataset configurations use a single held-out class. Researchers may wish to extend them to multi-class held-out settings.
- The RoBERTa maximum length is 128 tokens in the standard text configurations.
- Near-OOD is tested with one configuration per modality, and only at base model size.

### Q30. Are there tasks for which the suite should NOT be used?
- High-stakes deployment decisions without further domain validation.
- Adversarial-robustness studies. This suite targets natural distribution shift only.

---

## Distribution

### Q31. Will the suite be distributed?
**Yes.** The audit composition is distributed under CC-BY-4.0: split configurations, head masks, evaluation outputs, statistics and evaluation code. Source datasets are NOT redistributed; users must download them from canonical sources.

### Q32. How will it be distributed?
- **Code:** GitHub repository https://github.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026
- **Fine-tuned checkpoints:** Zenodo, DOI [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711). See [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md).
- **Croissant metadata:** `croissant.json` in the repository (Croissant 1.1 with Responsible-AI fields).

### Q33. When will it be distributed?
Released in November 2026: GitHub release v1.0.0, with checkpoints on Zenodo (DOI 10.5281/zenodo.23338711).

### Q34. License?
- **Code:** Apache-2.0.
- **Audit documentation, splits, configurations, evaluation outputs:** CC-BY-4.0.
- **ViT-B/16, RoBERTa-base and RoBERTa-large fine-tuned checkpoints** (Zenodo): CC-BY-4.0.
- **LayoutLMv3-base fine-tuned checkpoints** (Zenodo): **CC-BY-NC-SA-4.0**, inherited from the upstream `microsoft/layoutlmv3-base` licence (non-commercial, share-alike, research-only).

See [LICENSE](LICENSE) and [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md).

### Q35. Have any third parties imposed IP-based restrictions?
Each source dataset carries its own licence (see [DATA_ACCESS.md](DATA_ACCESS.md)). Users must comply with each source licence independently.

### Q36. Do any export controls apply?
No.

---

## Maintenance

### Q37. Who will support/maintain the suite?
The author. See [HOSTING_AND_MAINTENANCE.md](HOSTING_AND_MAINTENANCE.md).

### Q38. How can the maintainer be contacted?
Through the GitHub repository's issue tracker, or by e-mail to vishnubindubalachandran@outlook.com.

### Q39. Is there an erratum?
Corrections are kept in the code repository as `ERRATA.md`. There are none so far.

### Q40. Will the suite be updated?
Yes. Versioned releases follow semantic versioning, and backward-incompatible changes increment the major version. See [CHANGELOG.md](CHANGELOG.md).

### Q41. Limits on data retention?
N/A. No PII is collected.

### Q42. Will older versions continue to be supported?
Yes. Past versions remain accessible through Zenodo (immutable per-version DOI).

### Q43. Mechanism for community contributions?
Yes: pull requests on the public GitHub repository. Contributions are covered by the same Apache-2.0 / CC-BY-4.0 dual licence.
