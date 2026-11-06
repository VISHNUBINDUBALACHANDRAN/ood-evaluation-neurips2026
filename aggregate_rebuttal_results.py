"""
aggregate_rebuttal_results.py — collate the additional experiments into one file.

Per-seed files hold either a list of rows or a bare dict; shapes are normalised
and malformed files are skipped.

Produces outputs/REBUTTAL_SUMMARY.json plus a readable console table.
"""

import glob
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "outputs")


def load_rows(pattern):
    """Return only well-formed dict rows, whatever shape the file has."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
            continue
        for r in (d if isinstance(d, list) else [d]):
            if isinstance(r, dict) and "config" in r:
                rows.append(r)
    return rows


def agg(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def main():
    summary = {}

    # ---- P3/P6 near-OOD + polarity -------------------------------------
    rows = load_rows(os.path.join(OUT, "nearood_polarity", "*_seed*.json"))
    print(f"\n=== P3/P6 near-OOD + polarity  ({len(rows)} rows) ===")
    p3 = {}
    for mo in sorted({r["modality"] for r in rows}):
        rs = [r for r in rows if r["modality"] == mo]
        blk = {"id_set": rs[0]["id_set"]}
        for cfg in sorted({r["config"] for r in rs}, reverse=True):
            g = [r for r in rs if r["config"] == cfg]
            blk[cfg] = {"regime": g[0]["regime"], "n_seeds": len(g)}
            for m in ("mahalanobis", "knn", "ahss"):
                v = [r[m]["auroc"] for r in g if m in r]
                if v:
                    blk[cfg][m] = agg(v)
            e = blk[cfg]
            print(f"  {mo:7s} {cfg:22s} n={len(g)}  " + "  ".join(
                f"{m}={e[m]['mean']:.4f}" for m in ("mahalanobis", "knn", "ahss") if m in e))
        near = [r for r in rs if r["regime"] == "near"]
        far = [r for r in rs if r["regime"] == "far"]
        if near and far:
            gap = (np.mean([r["mahalanobis"]["auroc"] for r in far])
                   - np.mean([r["mahalanobis"]["auroc"] for r in near]))
            inv = sum(1 for r in near if r["knn"]["auroc"] > r["mahalanobis"]["auroc"])
            blk["near_ood_drop_pts"] = float(gap * 100)
            blk["knn_inversion_seeds"] = f"{inv}/{len(near)}"
            print(f"  {mo:7s} -> near-OOD drop {gap*100:5.1f} pts | "
                  f"kNN beats Maha on near in {inv}/{len(near)} seeds")
        p3[mo] = blk
    summary["P3_P6_nearood_polarity"] = p3

    # ---- P1 Tobacco-Cross n=15 -----------------------------------------
    from scipy.stats import wilcoxon, ttest_rel
    pairs = []
    for p in sorted(glob.glob(os.path.join(OUT, "results", "tobacco_cross_seed*", "ood_results.json"))):
        m = {x["method"]: x["auroc"] for x in json.load(open(p))}
        if "mahAvgAvg_AHM" in m and "mahAvgAvg" in m:
            pairs.append((m["mahAvgAvg_AHM"], m["mahAvgAvg"], m.get("Mahalanobis")))
    if pairs:
        a = np.array([x[0] for x in pairs]); b = np.array([x[1] for x in pairs])
        d = a - b; n = len(d)
        w = wilcoxon(a, b, alternative="greater", method="exact").pvalue
        summary["P1_tobacco_cross"] = {
            "n_seeds": n, "delta_mean": float(d.mean()),
            "seeds_positive": int((d > 0).sum()),
            "wilcoxon_p": float(w), "t_p": float(ttest_rel(a, b, alternative="greater").pvalue),
            "cohens_dz": float(d.mean() / d.std(ddof=1)),
            "clears_bonferroni_m23": bool(w <= 0.05 / 23)}
        e = summary["P1_tobacco_cross"]
        print(f"\n=== P1 Tobacco-Cross  n={n} ===")
        print(f"  delta {e['delta_mean']:+.4f}  {e['seeds_positive']}/{n} positive  "
              f"wilcoxon={e['wilcoxon_p']:.2e}  clears Bonferroni: {e['clears_bonferroni_m23']}")

    # ---- MC-Dropout mechanism ------------------------------------------
    f = os.path.join(OUT, "mcdropout_mechanism", "vit_only_backup.json")
    if os.path.exists(f):
        r = json.load(open(f))
        mc = {}
        print("\n=== MC-Dropout mechanism (ViT) ===")
        for p in sorted({x["dropout_p"] for x in r}):
            s = [x for x in r if x["dropout_p"] == p]
            mc[f"p={p}"] = {"var_id": float(np.mean([x["var_id_mean"] for x in s])),
                            "auroc": agg([x["auroc"] for x in s])}
            print(f"  p={p:<5} var={mc[f'p={p}']['var_id']:.2e}  "
                  f"AUROC={mc[f'p={p}']['auroc']['mean']:.4f}")
        summary["MC_dropout_mechanism"] = mc

    # ---- P5 class-count (partial ok) ------------------------------------
    cc = {}
    for nc in (10, 20, 50, 100):
        tag = "" if nc >= 100 else f"_c{nc}"
        v = []
        for p in glob.glob(os.path.join(OUT, "cifar100_results", f"cifar100{tag}_svhn_seed*", "full_results.json")):
            d = json.load(open(p))
            if "X-Maha_optimized" in d:
                v.append(d["X-Maha_optimized"]["auroc"])
        if v:
            cc[f"num_classes={nc}"] = {"xmaha": agg(v)}
    if cc:
        summary["P5_class_count_partial"] = cc
        print("\n=== P5 class-count (partial) ===")
        for k, e in cc.items():
            print(f"  {k:18s} X-Maha={e['xmaha']['mean']:.4f} (n={e['xmaha']['n']})")

    path = os.path.join(OUT, "REBUTTAL_SUMMARY.json")
    json.dump(summary, open(path, "w"), indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
