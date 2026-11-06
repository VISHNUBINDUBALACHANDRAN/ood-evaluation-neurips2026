"""compute_paper_statistics.py — the statistics reported in the paper, computed from the released per-seed results.

No model is run: every number is read from the per-seed JSON files under outputs/ and paired by seed.
X-Mahalanobis is the leak-free uniform variant throughout.

Outputs (written to --out):
  paper_statistics.json  every number below, machine-readable
  paper_statistics.md    the same as readable tables
Contents:
  1. the 23 paired comparisons: one-sided exact Wilcoxon signed-rank test in the stated direction (primary),
     one-sided paired t-test, Cohen's d_z; Bonferroni and Benjamini-Hochberg over the 23 tests.
     The two headline comparisons enter the family at n = 15, all others at n = 5.
  2. the two headline comparisons at n = 5 and n = 15, with a seeded bootstrap 95% CI of the mean difference.
  3. the headroom check over the 11 standard configurations (baseline Mahalanobis AUROC vs mahAvgAvg_AHM gain).
  4. uniform X-Mahalanobis AUROC (mean and std over seeds) per configuration.
  5. the number of per-seed AUROC values behind the 11 standard configurations.

  python statistics/compute_paper_statistics.py [--outputs outputs] [--out statistics]
"""
import argparse, json, os
import numpy as np
import scipy
from scipy import stats

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS15 = SEEDS5 + [7, 13, 31, 77, 101, 202, 303, 404, 505, 2026]
BOOT_SEED, BOOT_N = 0, 10000
DOC = ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]
VIT = ["cifar10_intra", "cifar10_svhn"]
C100 = ["cifar100_svhn", "cifar100_dtd"]
TXT = ["sst2_mnli", "sst2_agnews", "sst2_20news"]
STANDARD = DOC + VIT + C100 + TXT

# Each comparison is tested one-sided in the direction the paper states: AUROC(a) > AUROC(b).
COMPARISONS = [
    ("financedocs_intra", "mahAvgAvg_AHM", "X-Maha"),
    ("financedocs_cross", "mahAHM", "Mahalanobis"),
    ("tobacco_intra", "mahAHM", "Mahalanobis"),
    ("tobacco_cross", "mahAvgAvg_AHM", "X-Maha"),          # headline (n = 15)
    ("tobacco_cross", "mahAvgAvg_AHM", "Mahalanobis"),
    ("financedocs_intra", "mahAvgAvg_AHM", "mahAHM"),
    ("tobacco_cross", "mahAvgAvg_AHM", "mahAHM"),
    ("cifar10_intra", "Mahalanobis", "mahAHM"),
    ("cifar10_intra", "Mahalanobis", "mahAvgAvg_AHM"),
    ("cifar100_svhn", "X-Maha", "mahAvgAvg_AHM"),
    ("cifar100_dtd", "X-Maha", "mahAvgAvg_AHM"),
    ("sst2_mnli", "mahAHM", "Mahalanobis"),
    ("sst2_agnews", "mahAHM", "Mahalanobis"),
    ("sst2_20news", "mahAHM", "X-Maha"),
    ("sst2_mnli", "Mahalanobis", "mahAvgAvg_AHM"),
    ("sst2_agnews", "mahAvgAvg_AHM", "Mahalanobis"),
    ("sst2_20news", "mahAvgAvg_AHM", "X-Maha"),           # headline (n = 15)
    ("cifar100_svhn", "X-Maha", "mahAHM"),
    ("cifar100_dtd", "X-Maha", "mahAHM"),
    ("sst2_mnli", "Mahalanobis", "AHSS"),
    ("sst2_agnews", "Mahalanobis", "AHSS"),
    ("sst2_agnews", "mahAHM", "MC-Dropout"),
    ("sst2_20news", "mahAHM", "MC-Dropout"),
]
HEADLINE = {("tobacco_cross", "mahAvgAvg_AHM", "X-Maha"), ("sst2_20news", "mahAvgAvg_AHM", "X-Maha")}


def _json(path):
    with open(path) as f:
        return json.load(f)


class Results:
    """AUROC per (configuration, method, seed), read from the released per-seed files."""

    def __init__(self, outputs):
        self.o = outputs

    def _mahavgavg(self, folder, cfg, seed):
        p = os.path.join(self.o, folder, f"{cfg}_seed{seed}", "mahavgavg_ahm_results.json")
        return _json(p)["auroc"] if os.path.exists(p) else None

    def auroc(self, cfg, method, seed):
        o = self.o
        if cfg in DOC:
            if method == "X-Maha":
                p = os.path.join(o, "xmaha_results", f"{cfg}_seed{seed}", "xmaha_results.json")
                return _json(p)["xmaha_uniform"]["auroc"] if os.path.exists(p) else None
            p = os.path.join(o, "results", f"{cfg}_seed{seed}", "ood_results.json")
            if not os.path.exists(p):
                return None
            rows = {r["method"]: r["auroc"] for r in _json(p)}
            return rows.get({"AHSS": "ahss", "kNN": "knn", "MC-Dropout": "mc_dropout"}.get(method, method))
        if cfg in VIT:
            if method == "X-Maha":
                p = os.path.join(o, "xmaha_results", f"vit_{cfg}_seed{seed}", "xmaha_results.json")
                return _json(p)["xmaha_uniform"]["auroc"] if os.path.exists(p) else None
            if method == "mahAvgAvg_AHM":
                return self._mahavgavg("vit_results", cfg, seed)
            if method == "MC-Dropout":
                p = os.path.join(o, "vit_results", f"{cfg}_seed{seed}", "results.json")
                return _json(p)["mc_dropout"]["auroc"] if os.path.exists(p) else None
            p = os.path.join(o, "vit_results", f"{cfg}_seed{seed}", "extended_results.json")
            if not os.path.exists(p):
                return None
            d = _json(p)
            key = {"Mahalanobis": "mahalanobis", "AHSS": "ahss", "kNN": "knn"}.get(method, method)
            return d[key]["auroc"] if isinstance(d.get(key), dict) else None
        folder = "cifar100_results" if cfg in C100 else "roberta_results"
        if method == "mahAvgAvg_AHM":
            return self._mahavgavg(folder, cfg, seed)
        p = os.path.join(o, folder, f"{cfg}_seed{seed}", "full_results.json")
        if not os.path.exists(p):
            return None
        d = _json(p)
        key = {"X-Maha": "X-Maha_uniform"}.get(method, method)
        return d[key]["auroc"] if isinstance(d.get(key), dict) else None

    def per_seed(self, cfg, method, seeds):
        vals = {s: self.auroc(cfg, method, s) for s in seeds}
        missing = [s for s, v in vals.items() if v is None]
        if missing:
            raise SystemExit(f"missing AUROC: {cfg} / {method} / seeds {missing}")
        return vals


def paired(a, b, seeds, boot=False):
    a = np.array([a[s] for s in seeds], dtype=float)
    b = np.array([b[s] for s in seeds], dtype=float)
    d = a - b
    r = {"n": len(seeds), "seeds": seeds, "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
         "a_wins": int((d > 0).sum()), "min_diff": d.min(), "max_diff": d.max(),
         "cohens_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan"),
         "wilcoxon_p_one_sided": stats.wilcoxon(a, b, alternative="greater", method="exact").pvalue,
         "wilcoxon_p_two_sided": stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue,
         "t_p_one_sided": stats.ttest_rel(a, b, alternative="greater").pvalue,
         "t_p_two_sided": stats.ttest_rel(a, b).pvalue}
    if boot:
        rng = np.random.default_rng(BOOT_SEED)
        means = d[rng.integers(0, len(d), size=(BOOT_N, len(d)))].mean(axis=1)
        r["bootstrap_ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outputs", default=os.path.join(here, "..", "outputs"))
    ap.add_argument("--out", default=here)
    args = ap.parse_args()
    R = Results(os.path.abspath(args.outputs))
    out = {"software": {"numpy": np.__version__, "scipy": scipy.__version__},
           "conventions": {"x_mahalanobis": "uniform (leak-free) variant", "test_direction": "one-sided, AUROC(a) > AUROC(b) as stated",
                           "primary_test": "exact Wilcoxon signed-rank", "family_size": len(COMPARISONS),
                           "bonferroni_threshold": 0.05 / len(COMPARISONS),
                           "bootstrap": f"{BOOT_N} resamples of seeds, percentile 95% CI, numpy default_rng({BOOT_SEED})",
                           "std": "population std over seeds (ddof = 0)"}}

    # 1. the 23 comparisons
    rows = []
    for cfg, a, b in COMPARISONS:
        seeds = SEEDS15 if (cfg, a, b) in HEADLINE else SEEDS5
        r = paired(R.per_seed(cfg, a, seeds), R.per_seed(cfg, b, seeds), seeds)
        rows.append({"setting": cfg, "a": a, "b": b, "headline": (cfg, a, b) in HEADLINE, **r})
    adj = bh_adjust([r["wilcoxon_p_one_sided"] for r in rows])
    m = len(rows)
    for r, q in zip(rows, adj):
        r["bh_adjusted_p"] = float(q)
        r["bonferroni_adjusted_p"] = float(min(1.0, r["wilcoxon_p_one_sided"] * m))
    out["family_23"] = rows
    out["family_summary"] = {"raw_p_le_0.05": sum(r["wilcoxon_p_one_sided"] <= 0.05 for r in rows),
                             "bh_le_0.05": sum(r["bh_adjusted_p"] <= 0.05 for r in rows),
                             "bonferroni_le_0.05": sum(r["bonferroni_adjusted_p"] <= 0.05 for r in rows)}

    # 2. headline comparisons at n = 5 and n = 15
    out["headline"] = {}
    for cfg, a, b in sorted(HEADLINE):
        for n, seeds in (("n5", SEEDS5), ("n15", SEEDS15)):
            out["headline"][f"{cfg}:{a}>{b}:{n}"] = paired(R.per_seed(cfg, a, seeds), R.per_seed(cfg, b, seeds), seeds, boot=True)

    # 3. headroom check over the 11 standard configurations (5 seeds each)
    mean = lambda cfg, meth: float(np.mean(list(R.per_seed(cfg, meth, SEEDS5).values())))
    base = [mean(c, "Mahalanobis") for c in STANDARD]
    head = {"configurations": STANDARD, "baseline_mahalanobis": base}
    for ref in ("X-Maha", "Mahalanobis"):
        gain = [mean(c, "mahAvgAvg_AHM") - mean(c, ref) for c in STANDARD]
        pr, sp = stats.pearsonr(base, gain), stats.spearmanr(base, gain)
        head[f"gain_vs_{ref}"] = {"gain": gain, "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
                                  "spearman_rho": float(sp[0]), "spearman_p": float(sp[1])}
    out["headroom"] = head

    # 4. uniform X-Mahalanobis per configuration
    out["xmaha_uniform"] = {}
    for c in STANDARD:
        v = np.array(list(R.per_seed(c, "X-Maha", SEEDS5).values()))
        out["xmaha_uniform"][c] = {"mean": float(v.mean()), "std": float(v.std()), "n": len(v)}
    v = np.array(list(R.per_seed("tobacco_cross", "X-Maha", SEEDS15).values()))
    out["xmaha_uniform"]["tobacco_cross_n15"] = {"mean": float(v.mean()), "std": float(v.std()), "n": len(v)}

    # 5. per-seed AUROC values behind the 11 standard configurations (all methods present in the files)
    out["auroc_value_count"] = count_values(os.path.abspath(args.outputs))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "paper_statistics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, "paper_statistics.md"), "w") as f:
        f.write(to_markdown(out))
    print(to_markdown(out))


def count_values(o):
    """Count (configuration, seed, method) AUROC values in the per-seed files of the 11 standard configurations."""
    per_cfg = {}
    for c in STANDARD:
        names = set()
        for s in SEEDS5:
            files = []
            if c in DOC:
                files = [("list", os.path.join(o, "results", f"{c}_seed{s}", "ood_results.json")),
                         ("xmaha", os.path.join(o, "xmaha_results", f"{c}_seed{s}", "xmaha_results.json")),
                         ("phase1", os.path.join(o, "ahss_results", f"{c}_seed{s}", "phase1_results.json"))]
            elif c in VIT:
                files = [("dict", os.path.join(o, "vit_results", f"{c}_seed{s}", "extended_results.json")),
                         ("mc", os.path.join(o, "vit_results", f"{c}_seed{s}", "results.json")),
                         ("one", os.path.join(o, "vit_results", f"{c}_seed{s}", "mahavgavg_ahm_results.json")),
                         ("xmaha", os.path.join(o, "xmaha_results", f"vit_{c}_seed{s}", "xmaha_results.json"))]
            else:
                folder = "cifar100_results" if c in C100 else "roberta_results"
                files = [("dict", os.path.join(o, folder, f"{c}_seed{s}", "full_results.json")),
                         ("one", os.path.join(o, folder, f"{c}_seed{s}", "mahavgavg_ahm_results.json"))]
            for kind, p in files:
                if not os.path.exists(p):
                    continue
                d = _json(p)
                if kind == "list":
                    names |= {(s, r["method"]) for r in d if r.get("auroc") is not None}
                elif kind == "dict":
                    names |= {(s, k) for k, v in d.items() if isinstance(v, dict) and "auroc" in v and k != "X-Maha_optimized"}
                elif kind == "mc":
                    names |= {(s, "mc_dropout")} if "mc_dropout" in d else set()
                elif kind == "one":
                    names |= {(s, "mahAvgAvg_AHM")}
                elif kind == "xmaha":
                    names |= {(s, "X-Maha_uniform")}
                elif kind == "phase1":   # AHSS and MC-Dropout (Mahalanobis and k-NN are already counted from ood_results)
                    names |= {(s, k) for k in ("ahss", "mc_dropout", "ahss_mah_combo", "ahss_knn_combo") if k in d}
        per_cfg[c] = {"values": len(names), "methods": sorted({m for _, m in names})}
    return {"per_configuration": per_cfg, "total": sum(v["values"] for v in per_cfg.values()),
            "note": "the optimised X-Mahalanobis variant (layer weights tuned on the evaluation and OOD data) is not counted"}


def to_markdown(out):
    f = lambda x: f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"
    L = ["# Paper statistics (computed by compute_paper_statistics.py)", "",
         f"Software: numpy {out['software']['numpy']}, scipy {out['software']['scipy']}. "
         "X-Mahalanobis = uniform variant. Tests are one-sided in the stated direction (a > b); "
         f"Bonferroni threshold over 23 tests = {out['conventions']['bonferroni_threshold']:.5f}.", "",
         "## 1. The 23 paired comparisons", "",
         "| # | setting | a > b | n | Δ mean | a wins | Wilcoxon p (1-sided) | paired t p (1-sided) | d_z | BH-adjusted | Bonferroni-adjusted |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(out["family_23"], 1):
        L.append(f"| {i} | {r['setting']} | {r['a']} > {r['b']}{' (headline)' if r['headline'] else ''} | {r['n']} | {r['mean_diff']:+.4f} | "
                 f"{r['a_wins']}/{r['n']} | {f(r['wilcoxon_p_one_sided'])} | {f(r['t_p_one_sided'])} | {r['cohens_dz']:.2f} | "
                 f"{f(r['bh_adjusted_p'])} | {f(r['bonferroni_adjusted_p'])} |")
    s = out["family_summary"]
    L += ["", f"Raw p ≤ 0.05: {s['raw_p_le_0.05']}/23 · BH-adjusted ≤ 0.05: {s['bh_le_0.05']}/23 · "
              f"Bonferroni-adjusted ≤ 0.05: {s['bonferroni_le_0.05']}/23.", "", "## 2. Headline comparisons", "",
          "| comparison | n | Δ mean | a wins | range of Δ | Wilcoxon p (1-sided) | paired t p (1-sided) | d_z | bootstrap 95% CI |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["headline"].items():
        L.append(f"| {k.rsplit(':', 1)[0]} | {r['n']} | {r['mean_diff']:+.4f} | {r['a_wins']}/{r['n']} | "
                 f"[{r['min_diff']:+.4f}, {r['max_diff']:+.4f}] | {f(r['wilcoxon_p_one_sided'])} | {f(r['t_p_one_sided'])} | "
                 f"{r['cohens_dz']:.2f} | [{r['bootstrap_ci95'][0]:+.4f}, {r['bootstrap_ci95'][1]:+.4f}] |")
    h = out["headroom"]
    L += ["", "## 3. Headroom check (11 standard configurations, 5 seeds)", ""]
    for ref in ("X-Maha", "Mahalanobis"):
        g = h[f"gain_vs_{ref}"]
        L.append(f"- gain = mahAvgAvg_AHM − {ref}: Pearson r = {g['pearson_r']:+.3f} (p = {g['pearson_p']:.2f}), "
                 f"Spearman ρ = {g['spearman_rho']:+.3f} (p = {g['spearman_p']:.2f})")
    L += ["", "## 4. Uniform X-Mahalanobis AUROC (mean ± population std)", "", "| configuration | n | AUROC |", "|---|---|---|"]
    for c, v in out["xmaha_uniform"].items():
        L.append(f"| {c} | {v['n']} | {v['mean']:.4f} ± {v['std']:.4f} |")
    cnt = out["auroc_value_count"]
    L += ["", f"## 5. Per-seed AUROC values behind the 11 standard configurations: {cnt['total']}", ""]
    for c, v in cnt["per_configuration"].items():
        L.append(f"- {c}: {v['values']} ({len(v['methods'])} methods × seeds)")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
