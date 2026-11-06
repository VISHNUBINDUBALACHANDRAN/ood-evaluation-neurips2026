"""export_per_seed_auroc.py — collect the per-seed AUROC values of the reported studies into one table.

Writes statistics/per_seed_auroc.csv with one row per (study, configuration, method, seed), read from the per-seed
result files of this repository; `source_file` gives the file each value comes from. Studies: the 11 standard
configurations, the additional seeds of the two headline comparisons, the CIFAR-100 class-count subsets, RoBERTa-large,
and near-OOD (vision, text, documents). Method names are kept exactly as stored in those files. X-Mahalanobis is the
uniform variant; the optimised variant (layer weights tuned on the evaluation and OOD data) is not included.
The MC-Dropout dropout-rate intervention and the hyperparameter ablation are in outputs/mcdropout_mechanism/ and
outputs/ablation_results/.

  python statistics/export_per_seed_auroc.py [--root .] [--out statistics/per_seed_auroc.csv]
"""
import argparse, csv, json, os

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS_EXTRA = [7, 13, 31, 77, 101, 202, 303, 404, 505, 2026]
COLUMNS = ["study", "backbone", "configuration", "ood_regime", "method", "seed", "auroc", "fpr95", "source_file"]
EXCLUDED = {"X-Maha_optimized", "xmaha_optimized"}


class Table:
    def __init__(self, root):
        self.root, self.rows = root, []

    def _load(self, rel):
        p = os.path.join(self.root, rel)
        if not os.path.exists(p):
            raise SystemExit(f"missing result file: {rel}")
        with open(p) as f:
            return json.load(f)

    def add(self, study, backbone, cfg, regime, method, seed, res, rel):
        fpr = res.get("fpr_at_95", res.get("fpr95", res.get("fpr")))
        self.rows.append([study, backbone, cfg, regime, method, seed, res["auroc"], fpr, rel])

    def dict_file(self, study, backbone, cfg, seed, rel, only=None, rename=None):
        d = self._load(rel)
        for k, v in d.items():
            if isinstance(v, dict) and "auroc" in v and k not in EXCLUDED and (only is None or k in only):
                self.add(study, backbone, cfg, "standard", (rename or {}).get(k, k), seed, v, rel)

    def mahavgavg(self, study, backbone, cfg, seed, rel):
        self.add(study, backbone, cfg, "standard", "mahAvgAvg_AHM", seed, self._load(rel), rel)

    def xmaha(self, study, backbone, cfg, seed, rel):
        self.add(study, backbone, cfg, "standard", "X-Maha_uniform", seed, self._load(rel)["xmaha_uniform"], rel)


def build(root):
    T = Table(root)
    # LayoutLMv3 (documents), standard OOD; Tobacco-Cross also with the 10 additional seeds
    for cfg in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        for study, seeds in (("main", SEEDS5), ("headline_additional_seeds", SEEDS_EXTRA if cfg == "tobacco_cross" else [])):
            for s in seeds:
                rel = f"outputs/results/{cfg}_seed{s}/ood_results.json"
                for r in T._load(rel):
                    if r.get("auroc") is not None:
                        T.add(study, "LayoutLMv3-base", cfg, "standard", r["method"], s, r, rel)
                T.xmaha(study, "LayoutLMv3-base", cfg, s, f"outputs/xmaha_results/{cfg}_seed{s}/xmaha_results.json")
                if study == "main":   # AHSS and MC-Dropout (Mahalanobis and k-NN of this file are already in ood_results.json)
                    T.dict_file(study, "LayoutLMv3-base", cfg, s, f"outputs/ahss_results/{cfg}_seed{s}/phase1_results.json",
                                only={"ahss", "mc_dropout", "ahss_mah_combo", "ahss_knn_combo"})
    # ViT-B/16 on CIFAR-10
    for cfg in ["cifar10_intra", "cifar10_svhn"]:
        for s in SEEDS5:
            T.dict_file("main", "ViT-B/16", cfg, s, f"outputs/vit_results/{cfg}_seed{s}/extended_results.json")
            T.dict_file("main", "ViT-B/16", cfg, s, f"outputs/vit_results/{cfg}_seed{s}/results.json", only={"mc_dropout"})
            T.mahavgavg("main", "ViT-B/16", cfg, s, f"outputs/vit_results/{cfg}_seed{s}/mahavgavg_ahm_results.json")
            T.xmaha("main", "ViT-B/16", cfg, s, f"outputs/xmaha_results/vit_{cfg}_seed{s}/xmaha_results.json")
    # ViT-B/16 on CIFAR-100 (standard) and the class-count subsets
    for study, cfgs in (("main", ["cifar100_svhn", "cifar100_dtd"]),
                        ("class_count", ["cifar100_c10_svhn", "cifar100_c20_svhn", "cifar100_c50_svhn"])):
        for cfg in cfgs:
            for s in SEEDS5:
                T.dict_file(study, "ViT-B/16", cfg, s, f"outputs/cifar100_results/{cfg}_seed{s}/full_results.json")
                T.mahavgavg(study, "ViT-B/16", cfg, s, f"outputs/cifar100_results/{cfg}_seed{s}/mahavgavg_ahm_results.json")
    # RoBERTa-base on SST-2; SST-2 vs 20 Newsgroups also with the 10 additional seeds
    for cfg in ["sst2_mnli", "sst2_agnews", "sst2_20news"]:
        for study, seeds in (("main", SEEDS5), ("headline_additional_seeds", SEEDS_EXTRA if cfg == "sst2_20news" else [])):
            for s in seeds:
                T.dict_file(study, "RoBERTa-base", cfg, s, f"outputs/roberta_results/{cfg}_seed{s}/full_results.json")
                T.mahavgavg(study, "RoBERTa-base", cfg, s, f"outputs/roberta_results/{cfg}_seed{s}/mahavgavg_ahm_results.json")
    # RoBERTa-large on SST-2 (result folders sst2_robertalarge_*)
    for ood in ["mnli", "agnews", "20news"]:
        for s in SEEDS5:
            T.dict_file("roberta_large", "RoBERTa-large", f"sst2_{ood}", s,
                        f"outputs/roberta_results/sst2_robertalarge_{ood}_seed{s}/full_results.json")
    # near-OOD: vision (ViT-B/16, CIFAR-100 small mammals) and text (RoBERTa-base, 20 Newsgroups comp.*)
    for study, backbone, prefix in (("near_ood_vision", "ViT-B/16", "vision_v2_seed"), ("near_ood_text", "RoBERTa-base", "text_seed")):
        for s in SEEDS5:
            rel = f"outputs/nearood_polarity/{prefix}{s}.json"
            for r in T._load(rel):
                for m in ("mahalanobis", "knn", "ahss"):
                    if m in r:
                        T.add(study, backbone, r["id_set"], r["regime"], m, s, r[m], rel)
    # near-OOD: documents (LayoutLMv3 leave-class-out models)
    for h in ["Memo", "Form", "Resume", "Scientific"]:
        for s in SEEDS5:
            rel = f"verification_2026-10/results/E1_doc_nearood/lco/leaveout_{h}_seed{s}.json"
            d = T._load(rel)
            for part, cfg in (("all_id_vs_ood", f"tobacco3482_holdout_{h}"), ("letter_vs_ood", f"tobacco3482_letter_vs_{h}")):
                for m, res in d[part].items():
                    T.add("near_ood_documents", "LayoutLMv3-base", cfg, "near", m, s, res, rel)
            if h == "Memo":
                rel = f"verification_2026-10/results/E1_doc_nearood/ahss/leaveout_Memo_seed{s}.json"
                a = T._load(rel)
                T.add("near_ood_documents", "LayoutLMv3-base", "tobacco3482_holdout_Memo", "near", "ahss", s, a["ahss_all_id_vs_ood"], rel)
                T.add("near_ood_documents", "LayoutLMv3-base", "tobacco3482_letter_vs_Memo", "near", "ahss", s, a["ahss_letter_vs_ood"], rel)
    return T.rows


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.join(here, ".."))
    ap.add_argument("--out", default=os.path.join(here, "per_seed_auroc.csv"))
    a = ap.parse_args()
    rows = build(os.path.abspath(a.root))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r[0]] = by.get(r[0], 0) + 1
    print(f"wrote {len(rows)} rows:", by)


if __name__ == "__main__":
    main()
