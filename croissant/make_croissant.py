"""make_croissant.py — write croissant.json (Croissant 1.1 core + Responsible-AI fields) for the released measurements.

The record describes the measurement artifacts of this repository:
  * statistics/per_seed_auroc.csv: the per-seed AUROC / FPR95 values of the reported studies (record set `per_seed_auroc`);
  * the per-seed result files, the attention-head masks and the near-OOD split configurations (file sets).
No raw data from the source datasets is redistributed.

  python croissant/make_croissant.py [--root .] [--out croissant.json] [--local_csv]
--local_csv points the table at the local file instead of the release URL (for offline validation only).
"""
import argparse, hashlib, json, os

REPO = "https://github.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026"
RELEASE_TAG = "v1.0.0"
RAW = f"https://raw.githubusercontent.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026/{RELEASE_TAG}"
TITLE = ("Auditing Attention Head Masking for Out-of-Distribution Detection: "
         "Cross-Architecture Failures and Polarity Inversions")

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/", "prov": "http://www.w3.org/ns/prov#",
    "arrayShape": "cr:arrayShape", "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn", "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract", "field": "cr:field",
    "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
    "includes": "cr:includes", "isArray": "cr:isArray", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
    "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet",
    "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated", "replace": "cr:replace",
    "samplingRate": "cr:samplingRate", "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
    "transform": "cr:transform",
}

SOURCES = [  # public sources of every dataset and pretrained model used (the source data keep their own licences)
    ("Tobacco3482", "https://huggingface.co/datasets/jordyvl/Tobacco3482"),
    ("FinanceDocs (Constantinou et al., 2024)", "https://github.com/constantinouchristos/OOD-AHM"),
    ("CIFAR-10 and CIFAR-100", "https://www.cs.toronto.edu/~kriz/cifar.html"),
    ("SVHN", "http://ufldl.stanford.edu/housenumbers/"),
    ("Describable Textures Dataset (DTD)", "https://www.robots.ox.ac.uk/~vgg/data/dtd/"),
    ("SST-2", "https://huggingface.co/datasets/stanfordnlp/sst2"),
    ("MultiNLI", "https://huggingface.co/datasets/nyu-mll/multi_nli"),
    ("AG News", "https://huggingface.co/datasets/fancyzhx/ag_news"),
    ("20 Newsgroups", "http://qwone.com/~jason/20Newsgroups/"),
    ("LayoutLMv3-base", "https://huggingface.co/microsoft/layoutlmv3-base"),
    ("ViT-B/16", "https://huggingface.co/google/vit-base-patch16-224"),
    ("RoBERTa-base", "https://huggingface.co/FacebookAI/roberta-base"),
    ("RoBERTa-large", "https://huggingface.co/FacebookAI/roberta-large"),
]

COLUMNS = [("study", "sc:Text", "Study the value belongs to: main, headline_additional_seeds, class_count, roberta_large, "
                                "near_ood_vision, near_ood_text or near_ood_documents."),
           ("backbone", "sc:Text", "Fine-tuned transformer: LayoutLMv3-base, ViT-B/16, RoBERTa-base or RoBERTa-large."),
           ("configuration", "sc:Text", "In-distribution / OOD configuration, e.g. tobacco_cross or sst2_20news."),
           ("ood_regime", "sc:Text", "standard, near or far."),
           ("method", "sc:Text", "OOD scoring method, named as in the result file."),
           ("seed", "sc:Integer", "Fine-tuning seed."),
           ("auroc", "sc:Float", "AUROC of separating in-distribution from OOD inputs."),
           ("fpr95", "sc:Float", "False-positive rate at 95% true-positive rate."),
           ("source_file", "sc:Text", "Result file in this repository that holds the value.")]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(root, local_csv):
    csv_rel = "statistics/per_seed_auroc.csv"
    csv_path = os.path.join(root, csv_rel)
    repo_obj = {"@type": "cr:FileObject", "@id": "github-repository", "name": "github-repository",
                "description": "Source repository of the paper (code, results, masks and split configurations).",
                "contentUrl": REPO, "encodingFormat": "git+https", "sha256": "main"}
    table = {"@type": "cr:FileObject", "@id": "per-seed-auroc-csv", "name": "per_seed_auroc.csv",
             "description": "Per-seed AUROC and FPR95 values of the reported studies, one row per study, configuration, method and seed.",
             "contentUrl": os.path.abspath(csv_path) if local_csv else f"{RAW}/{csv_rel}",
             "encodingFormat": "text/csv", "sha256": sha256(csv_path)}
    file_sets = [
        ("per-seed-result-files", "Per-seed result files (JSON) of all runs, as written by the experiment scripts.",
         "application/json", ["outputs/**/*.json", "verification_2026-10/results/**/*.json"]),
        ("attention-head-masks", "Attention-head masks: AHM-selected masks of the LayoutLMv3 runs and the random masks of the "
                                 "ViT-B/16 and RoBERTa runs (NumPy arrays, 1 = head kept, 0 = head masked).",
         "application/x-npy", ["outputs/ahm_masks/**/*.npy", "random_head_masks/*.npy"]),
        ("near-ood-split-configurations", "Near-OOD split configurations (documents, vision, text) with exact members.",
         "application/json", ["split_configs/*.json"]),
    ]
    distribution = [repo_obj, table] + [
        {"@type": "cr:FileSet", "@id": i, "name": i, "description": d, "containedIn": {"@id": "github-repository"},
         "encodingFormat": fmt, "includes": inc} for i, d, fmt, inc in file_sets]
    record_set = {"@type": "cr:RecordSet", "@id": "per_seed_auroc", "name": "per_seed_auroc",
                  "description": "Per-seed OOD detection results behind every table and statistic of the paper.",
                  "field": [{"@type": "cr:Field", "@id": f"per_seed_auroc/{c}", "name": c, "description": d, "dataType": t,
                             "source": {"fileObject": {"@id": "per-seed-auroc-csv"}, "extract": {"column": c}}}
                            for c, t, d in COLUMNS]}
    cite = ("@inproceedings{balachandran2026auditing,\n  title={" + TITLE + "},\n  author={Balachandran, Vishnu Bindu},\n"
            "  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},\n"
            "  year={2026}\n}")
    return {
        "@context": CONTEXT, "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "ahm-ood-audit-measurements",
        "description": ("Measurements released with the NeurIPS 2026 Evaluations and Datasets Track paper \"" + TITLE + "\": "
                        "per-seed AUROC and FPR95 values of post-hoc OOD detectors on LayoutLMv3-base, ViT-B/16, RoBERTa-base "
                        "and RoBERTa-large, the attention-head masks, and the near-OOD split configurations. "
                        "The release contains only these derived artifacts; no data from the source datasets is redistributed."),
        "url": REPO, "license": "https://creativecommons.org/licenses/by/4.0/", "version": "1.0.0",
        "datePublished": "2026-11-06", "isAccessibleForFree": True,
        "creator": {"@type": "sc:Person", "name": "Vishnu Bindu Balachandran", "url": "https://www.vishnubindubalachandran.com"},
        "keywords": ["out-of-distribution detection", "attention head masking", "transformers", "evaluation", "audit",
                     "LayoutLMv3", "ViT", "RoBERTa"],
        "citeAs": cite,
        "rai:dataLimitations": ("Results cover LayoutLMv3-base, ViT-B/16, RoBERTa-base and RoBERTa-large, five fine-tuning "
                                "seeds per configuration (fifteen for the two headline comparisons). They describe these "
                                "models, datasets and protocols and should not be read as properties of other architectures, "
                                "scales or data. Implementation details that affect specific scores are stated in the paper. "
                                "Tobacco3482 split members are rows of the processed dataset (see split_configs/README.md)."),
        "rai:dataBiases": ("The measurements inherit the composition of the source datasets: English-language text "
                           "(SST-2, MultiNLI, AG News, 20 Newsgroups), scanned business documents from specific sources "
                           "and periods (Tobacco3482, FinanceDocs) and natural-image benchmarks (CIFAR, SVHN, DTD)."),
        "rai:personalSensitiveInformation": ("None. The release contains numbers (AUROC, FPR95), binary head masks and integer "
                                             "split indices only; no documents, images or text are included."),
        "rai:dataUseCases": ("Reproducing and re-analysing the paper's results (for example with other statistics or effect "
                             "sizes) and comparing post-hoc OOD detectors across architectures. Not intended as training data."),
        "rai:dataSocialImpact": ("Supports more reliable evaluation of OOD detectors before deployment by making cross-"
                                 "architecture results checkable; no direct social risk was identified."),
        "rai:hasSyntheticData": False,
        "rai:dataReleaseMaintenancePlan": ("Maintained by the author in the GitHub repository (issues for questions and "
                                           "corrections); the fine-tuned checkpoints are archived on Zenodo (DOI 10.5281/zenodo.23338711)."),
        "prov:wasDerivedFrom": [{"@type": "sc:Dataset", "name": n, "url": u} for n, u in SOURCES],
        "prov:wasGeneratedBy": ("Each backbone was fine-tuned per seed on the in-distribution data and scored with post-hoc "
                                "OOD detectors by the scripts in this repository (see REPRODUCTION.md); the table is written by "
                                "statistics/export_per_seed_auroc.py and the statistics by statistics/compute_paper_statistics.py."),
        "distribution": distribution,
        "recordSet": [record_set],
    }


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.join(here, ".."))
    ap.add_argument("--out", default=os.path.join(here, "..", "croissant.json"))
    ap.add_argument("--local_csv", action="store_true")
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(build(os.path.abspath(a.root), a.local_csv), f, indent=2, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
