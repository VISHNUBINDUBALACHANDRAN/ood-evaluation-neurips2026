# Data Access — Source Datasets

This benchmark **does not redistribute raw data**. All ten source datasets are publicly available; this document gives canonical URLs, licenses, and download commands.

> Every dataset in this list can be downloaded without registration or personal request.

---

## Vision Datasets

### 1. CIFAR-10
- **URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **License**: MIT-equivalent (Krizhevsky's release terms)
- **Download**: auto-downloads via `torchvision.datasets.CIFAR10(root=..., download=True)`
- **Size**: 163 MB

### 2. CIFAR-100
- **URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **License**: MIT-equivalent
- **Download**: `torchvision.datasets.CIFAR100(root=..., download=True)`
- **Size**: 161 MB

### 3. SVHN
- **URL**: http://ufldl.stanford.edu/housenumbers/
- **License**: For non-commercial use only ("for academic, research and educational purposes")
- **Download**: `torchvision.datasets.SVHN(root=..., split='test', download=True)`
- **Size**: 1.5 GB

### 4. DTD (Describable Textures Dataset)
- **URL**: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- **License**: Custom — research only, freely redistributable for academic use
- **Download**: via HuggingFace `load_dataset("dtd")` or `torchvision.datasets.DTD(root=..., download=True)`
- **Size**: 600 MB

---

## Text Datasets

### 5. SST-2 (Stanford Sentiment Treebank, binary)
- **URL**: https://huggingface.co/datasets/stanfordnlp/sst2 (also part of GLUE)
- **License**: Original SST: research-only; HuggingFace re-host: same terms
- **Download**: `datasets.load_dataset("stanfordnlp/sst2")`
- **Size**: 7 MB

### 6. MultiNLI (MNLI)
- **URL**: https://huggingface.co/datasets/nyu-mll/multi_nli
- **License**: Mixed — see https://cims.nyu.edu/~sbowman/multinli/ ; OANC portion is permissive, fiction portion is research-only
- **Download**: `datasets.load_dataset("nyu-mll/multi_nli")`
- **Size**: 220 MB

### 7. AG News
- **URL**: https://huggingface.co/datasets/fancyzhx/ag_news
- **License**: Original: "for any non-commercial use" (Zhang et al. 2015 paper)
- **Download**: `datasets.load_dataset("fancyzhx/ag_news")`
- **Size**: 32 MB

### 8. 20 Newsgroups
- **URL**: http://qwone.com/~jason/20Newsgroups/  (also `sklearn.datasets.fetch_20newsgroups`)
- **License**: Public domain (USENET public posts)
- **Download**: `from sklearn.datasets import fetch_20newsgroups`
- **Size**: 14 MB
- **Caveat**: contains historical Usenet content; see `ETHICS.md` and `DATASHEET.md` Q15.

---

## Document AI Datasets

### 9. Tobacco-3482
- **URL**: https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72  (mirror: https://huggingface.co/datasets/jordyvl/Tobacco3482)
- **License**: Publicly released litigation documents (UCSF Industry Documents Library)
- **Download**: `datasets.load_dataset("jordyvl/Tobacco3482")`
- **Size**: 290 MB

### 10. FinanceDocs
- **Source**: introduced and publicly released by Constantinou et al. (arXiv:2408.11237) — https://github.com/constantinouchristos/OOD-AHM (folder `data/`, split archive with reassembly instructions) and the dataset link given in that paper (https://drive.google.com/drive/folders/1dV9obe_3hTsDoWJyYuNLBAXEiwOPwCw7).
- **Content**: digital PDF business documents in 10 classes (SEC forms and letters, 10-K pages, articles of association, shareholder letters, lists of directors, financial information, résumés, scientific AI papers), collected by the original authors largely from public SEC filings.
- **License**: as set by the original authors (we found no separate licence statement for the dataset); we do not redistribute it.
- **Size**: ~3 GB

---

## Pre-trained Model Weights (HuggingFace Hub)

All loaded automatically by HuggingFace `transformers`. No manual downloads needed.

| Model | HF Hub ID | License |
|-------|-----------|---------|
| ViT-B/16 | `google/vit-base-patch16-224` | Apache-2.0 |
| RoBERTa-base | `roberta-base` | MIT |
| RoBERTa-large | `roberta-large` | MIT |
| LayoutLMv3-base | `microsoft/layoutlmv3-base` | CC-BY-NC-SA-4.0 (research-only) |

> **License note**: LayoutLMv3 is research-only. Users deploying this benchmark for commercial purposes must replace LayoutLMv3 with a commercially licensed alternative.

---

## Data Locations

The experiment scripts read the datasets from `data/` in the repository root (for example `data/cifar100/`, `data/hf_cache/`, and scikit-learn's 20 Newsgroups cache in `data/`). The document datasets (Tobacco3482, FinanceDocs) are converted to LayoutLMv3 inputs with

```bash
python run_prepare_data.py --data_folder <folder with one sub-folder per class> \
    --output_folder <output folder> --processor_path microsoft/layoutlmv3-base
```

which runs OCR through the LayoutLMv3 processor (`apply_ocr=True`, requires Tesseract) and writes `<output folder>/colated_tensors` (Tobacco3482: `data/tobacco/colated_tensors`; FinanceDocs: `data/colated_tensors`).
