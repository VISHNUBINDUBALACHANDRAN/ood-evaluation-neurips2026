"""
run_aggregate_results.py — Phase 6: Aggregate and format all experimental results.

Reads results from Phase 1 (AHSS + baselines), Phase 2 (D-AHM), Phase 3 (NC),
and Phase 4 (ViT) and outputs publication-ready tables for the NeurIPS paper.

Usage:
    python run_aggregate_results.py
"""

import argparse
import os
import sys
import json
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("financedocs", "intra"),
    ("financedocs", "cross"),
    ("tobacco", "intra"),
    ("tobacco", "cross"),
]
SEEDS = [42, 123, 456, 789, 1024]


def load_phase1_results():
    """Load Phase 1 results (AHSS + baselines) for all experiments/seeds."""
    results = defaultdict(list)
    results_dir = os.path.join(BASE_DIR, "outputs", "ahss_results")

    for ds, et in EXPERIMENTS:
        for seed in SEEDS:
            exp_name = f"{ds}_{et}_seed{seed}"
            path = os.path.join(results_dir, exp_name, "phase1_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                results[f"{ds}_{et}"].append(r)

    return dict(results)


def load_phase2_results():
    """Load Phase 2 D-AHM optimization results."""
    results = {}
    dahm_dir = os.path.join(BASE_DIR, "outputs", "dahm_results")
    if not os.path.exists(dahm_dir):
        return results

    for ds, et in EXPERIMENTS:
        for seed in SEEDS:
            exp_name = f"{ds}_{et}_seed{seed}"
            path = os.path.join(dahm_dir, exp_name, "dahm_results.json")
            if os.path.exists(path):
                key = f"{ds}_{et}"
                if key not in results:
                    results[key] = []
                with open(path) as f:
                    results[key].append(json.load(f))

    return results


def load_phase3_results():
    """Load Phase 3 Neural Collapse results."""
    path = os.path.join(BASE_DIR, "outputs", "nc_results", "nc_summary.json")
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        data = json.load(f)

    results = defaultdict(list)
    for r in data:
        parts = r["experiment"].rsplit("_seed", 1)
        results[parts[0]].append(r)

    return dict(results)


def load_phase4_results():
    """Load Phase 4 ViT experiment results."""
    results = {}
    vit_dir = os.path.join(BASE_DIR, "outputs", "vit_results")
    if not os.path.exists(vit_dir):
        return results

    for exp_type in ["cifar10_intra", "cifar10_svhn"]:
        exp_results = []
        for seed in SEEDS:
            path = os.path.join(vit_dir, f"{exp_type}_seed{seed}", "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    exp_results.append(json.load(f))
        if exp_results:
            results[exp_type] = exp_results

    return results


def fmt(mean, std, decimals=4):
    """Format mean±std."""
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def generate_table1(phase1):
    """Table 1: Main AHSS results — AUROC comparison."""
    lines = []
    lines.append("=" * 95)
    lines.append("TABLE 1: OOD Detection AUROC (mean ± std across 5 seeds)")
    lines.append("=" * 95)
    header = f"{'Experiment':<22} {'AHSS':>16} {'MC-Drop':>16} {'Mahalanobis':>16} {'kNN':>16}"
    lines.append(header)
    lines.append("-" * 95)

    for exp_key in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        runs = phase1.get(exp_key, [])
        if not runs:
            lines.append(f"{exp_key:<22} {'N/A':>16} {'N/A':>16} {'N/A':>16} {'N/A':>16}")
            continue

        ahss_aurocs = [r["ahss"]["auroc"] for r in runs]
        mc_aurocs = [r["mc_dropout"]["auroc"] for r in runs]
        maha_aurocs = [r["mahalanobis"]["auroc"] for r in runs]
        knn_aurocs = [r["knn"]["auroc"] for r in runs]

        lines.append(
            f"{exp_key:<22} "
            f"{fmt(np.mean(ahss_aurocs), np.std(ahss_aurocs)):>16} "
            f"{fmt(np.mean(mc_aurocs), np.std(mc_aurocs)):>16} "
            f"{fmt(np.mean(maha_aurocs), np.std(maha_aurocs)):>16} "
            f"{fmt(np.mean(knn_aurocs), np.std(knn_aurocs)):>16}"
        )

    lines.append("-" * 95)
    return "\n".join(lines)


def generate_table2(phase1):
    """Table 2: FPR@95TPR comparison."""
    lines = []
    lines.append("=" * 95)
    lines.append("TABLE 2: FPR@95TPR (lower is better, mean ± std)")
    lines.append("=" * 95)
    header = f"{'Experiment':<22} {'AHSS':>16} {'MC-Drop':>16} {'Mahalanobis':>16} {'kNN':>16}"
    lines.append(header)
    lines.append("-" * 95)

    for exp_key in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        runs = phase1.get(exp_key, [])
        if not runs:
            lines.append(f"{exp_key:<22} {'N/A':>16} {'N/A':>16} {'N/A':>16} {'N/A':>16}")
            continue

        ahss_fprs = [r["ahss"]["fpr"] for r in runs]
        mc_fprs = [r["mc_dropout"]["fpr"] for r in runs]
        maha_fprs = [r["mahalanobis"]["fpr"] for r in runs]
        knn_fprs = [r["knn"]["fpr"] for r in runs]

        lines.append(
            f"{exp_key:<22} "
            f"{fmt(np.mean(ahss_fprs), np.std(ahss_fprs)):>16} "
            f"{fmt(np.mean(mc_fprs), np.std(mc_fprs)):>16} "
            f"{fmt(np.mean(maha_fprs), np.std(maha_fprs)):>16} "
            f"{fmt(np.mean(knn_fprs), np.std(knn_fprs)):>16}"
        )

    lines.append("-" * 95)
    return "\n".join(lines)


def generate_table3(phase3):
    """Table 3: Neural Collapse metrics."""
    if not phase3:
        return "TABLE 3: Neural Collapse — No data available"

    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 3: Neural Collapse Metrics (eval embeddings, mean ± std)")
    lines.append("=" * 90)
    header = f"{'Experiment':<22} {'NC1':>16} {'NC2':>12} {'NC4':>12} {'NC1(AHM)':>16}"
    lines.append(header)
    lines.append("-" * 90)

    for exp_key in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        runs = phase3.get(exp_key, [])
        if not runs:
            continue

        nc1s = [r["eval_nc"]["nc1"] for r in runs]
        nc2s = [r["eval_nc"]["nc2"] for r in runs]
        nc4s = [r["eval_nc"]["nc4"] for r in runs]
        nc1_ahms = [r["ahm_nc"]["nc1"] for r in runs if r.get("ahm_nc")]

        ahm_str = fmt(np.mean(nc1_ahms), np.std(nc1_ahms), 0) if nc1_ahms else "N/A"
        lines.append(
            f"{exp_key:<22} "
            f"{fmt(np.mean(nc1s), np.std(nc1s), 0):>16} "
            f"{fmt(np.mean(nc2s), np.std(nc2s)):>12} "
            f"{fmt(np.mean(nc4s), np.std(nc4s)):>12} "
            f"{ahm_str:>16}"
        )

    lines.append("-" * 90)
    return "\n".join(lines)


def generate_table4(phase4):
    """Table 4: ViT cross-architecture results."""
    if not phase4:
        return "TABLE 4: ViT Cross-Architecture — No data available"

    lines = []
    lines.append("=" * 130)
    lines.append("TABLE 4: ViT-B/16 Cross-Architecture Validation (C5)")
    lines.append("=" * 130)
    header = f"{'Experiment':<22} {'AHSS AUROC':>16} {'AHSS FPR@95':>16} {'MC-Drop':>16} {'Mahalanobis':>16} {'kNN':>16} {'Combined':>16}"
    lines.append(header)
    lines.append("-" * 130)

    for exp_key in ["cifar10_intra", "cifar10_svhn"]:
        runs = phase4.get(exp_key, [])
        if not runs:
            continue
        aurocs = [r.get("ahss_raw", {}).get("auroc", r.get("ahss_auroc", 0)) for r in runs]
        fprs = [r.get("ahss_raw", {}).get("fpr_at_95", r.get("ahss_fpr", 0)) for r in runs]
        mc_aurocs = [r.get("mc_dropout", {}).get("auroc", 0) for r in runs]
        maha_aurocs = [r.get("mahalanobis", {}).get("auroc", 0) for r in runs]
        knn_aurocs = [r.get("knn", {}).get("auroc", 0) for r in runs]
        combined_aurocs = [r.get("combined", {}).get("auroc", 0) for r in runs]
        lines.append(
            f"{exp_key:<22} "
            f"{fmt(np.mean(aurocs), np.std(aurocs)):>16} "
            f"{fmt(np.mean(fprs), np.std(fprs)):>16} "
            f"{fmt(np.mean(mc_aurocs), np.std(mc_aurocs)):>16} "
            f"{fmt(np.mean(maha_aurocs), np.std(maha_aurocs)):>16} "
            f"{fmt(np.mean(knn_aurocs), np.std(knn_aurocs)):>16} "
            f"{fmt(np.mean(combined_aurocs), np.std(combined_aurocs)):>16}"
        )

    lines.append("-" * 130)
    return "\n".join(lines)


def generate_latex_table1(phase1):
    """Generate LaTeX for Table 1 (main results)."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{OOD detection performance (AUROC / FPR@95TPR) on document benchmarks. Mean $\pm$ std over 5 seeds. Best AUROC per row in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & AHSS (Ours) & MC-Dropout & Mahalanobis & kNN \\")
    lines.append(r"\midrule")

    for exp_key, label in [
        ("financedocs_intra", "FinDocs-Intra"),
        ("financedocs_cross", "FinDocs-Cross"),
        ("tobacco_intra", "Tobacco-Intra"),
        ("tobacco_cross", "Tobacco-Cross"),
    ]:
        runs = phase1.get(exp_key, [])
        if not runs:
            lines.append(f"{label} & -- & -- & -- & -- \\\\")
            continue

        methods = {
            "ahss": ([r["ahss"]["auroc"] for r in runs], [r["ahss"]["fpr"] for r in runs]),
            "mc": ([r["mc_dropout"]["auroc"] for r in runs], [r["mc_dropout"]["fpr"] for r in runs]),
            "maha": ([r["mahalanobis"]["auroc"] for r in runs], [r["mahalanobis"]["fpr"] for r in runs]),
            "knn": ([r["knn"]["auroc"] for r in runs], [r["knn"]["fpr"] for r in runs]),
        }

        means = {k: np.mean(v[0]) for k, v in methods.items()}
        best = max(means, key=means.get)

        cells = []
        for k in ["ahss", "mc", "maha", "knn"]:
            aurocs, fprs = methods[k]
            cell = f"${np.mean(aurocs):.3f}_{{{np.std(aurocs):.3f}}}$/${np.mean(fprs):.3f}$"
            if k == best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_summary_stats(phase1, phase3):
    """Generate key summary statistics for the paper abstract/intro."""
    lines = []
    lines.append("=" * 70)
    lines.append("KEY STATISTICS FOR PAPER")
    lines.append("=" * 70)

    if phase1:
        all_ahss = []
        all_mc = []
        all_maha = []
        all_knn = []
        for runs in phase1.values():
            for r in runs:
                all_ahss.append(r["ahss"]["auroc"])
                all_mc.append(r["mc_dropout"]["auroc"])
                all_maha.append(r["mahalanobis"]["auroc"])
                all_knn.append(r["knn"]["auroc"])

        lines.append(f"Total Phase 1 runs: {len(all_ahss)}")
        lines.append(f"AHSS mean AUROC: {np.mean(all_ahss):.4f}")
        lines.append(f"MC-Dropout mean AUROC: {np.mean(all_mc):.4f}")
        lines.append(f"Mahalanobis mean AUROC: {np.mean(all_maha):.4f}")
        lines.append(f"kNN mean AUROC: {np.mean(all_knn):.4f}")
        lines.append(f"AHSS vs MC-Dropout improvement: +{(np.mean(all_ahss)-np.mean(all_mc))*100:.1f}pp")
        lines.append(f"AHSS vs Mahalanobis improvement: +{(np.mean(all_ahss)-np.mean(all_maha))*100:.1f}pp")

        # Win rate (AHSS beats each baseline)
        wins_mc = sum(1 for a, m in zip(all_ahss, all_mc) if a > m)
        wins_maha = sum(1 for a, m in zip(all_ahss, all_maha) if a > m)
        wins_knn = sum(1 for a, m in zip(all_ahss, all_knn) if a > m)
        n = len(all_ahss)
        lines.append(f"AHSS wins vs MC-Dropout: {wins_mc}/{n} ({wins_mc/n*100:.0f}%)")
        lines.append(f"AHSS wins vs Mahalanobis: {wins_maha}/{n} ({wins_maha/n*100:.0f}%)")
        lines.append(f"AHSS wins vs kNN: {wins_knn}/{n} ({wins_knn/n*100:.0f}%)")

    if phase3:
        lines.append("\nNeural Collapse (C3):")
        for key, runs in phase3.items():
            nc1_evals = [r["eval_nc"]["nc1"] for r in runs]
            nc1_ahms = [r["ahm_nc"]["nc1"] for r in runs if r.get("ahm_nc")]
            if nc1_ahms:
                pct_reduction = (1 - np.mean(nc1_ahms)/np.mean(nc1_evals)) * 100
                lines.append(f"  {key}: NC1 reduction under AHM: {pct_reduction:.1f}%")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate all results")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()

    print("Loading Phase 1 results...")
    phase1 = load_phase1_results()
    print(f"  Found: {sum(len(v) for v in phase1.values())} runs across {len(phase1)} experiments")

    print("Loading Phase 2 results...")
    phase2 = load_phase2_results()
    print(f"  Found: {sum(len(v) for v in phase2.values()) if phase2 else 0} runs")

    print("Loading Phase 3 results...")
    phase3 = load_phase3_results()
    print(f"  Found: {sum(len(v) for v in phase3.values())} NC analyses")

    print("Loading Phase 4 results...")
    phase4 = load_phase4_results()
    print(f"  Found: {sum(len(v) for v in phase4.values()) if phase4 else 0} ViT runs")

    print()

    # Generate tables
    t1 = generate_table1(phase1)
    t2 = generate_table2(phase1)
    t3 = generate_table3(phase3)
    t4 = generate_table4(phase4)
    stats = generate_summary_stats(phase1, phase3)

    output = "\n\n".join([t1, t2, t3, t4, stats])
    print(output)

    # Save
    out_dir = os.path.join(BASE_DIR, "outputs", "aggregated")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_tables.txt"), "w") as f:
        f.write(output)
    print(f"\nSaved: {os.path.join(out_dir, 'all_tables.txt')}")

    if args.latex:
        latex = generate_latex_table1(phase1)
        with open(os.path.join(out_dir, "table1.tex"), "w") as f:
            f.write(latex)
        print(f"Saved: {os.path.join(out_dir, 'table1.tex')}")

    # Save structured JSON summary
    summary = {
        "phase1_count": sum(len(v) for v in phase1.values()),
        "phase2_count": sum(len(v) for v in phase2.values()) if phase2 else 0,
        "phase3_count": sum(len(v) for v in phase3.values()),
        "phase4_count": sum(len(v) for v in phase4.values()) if phase4 else 0,
        "phase1_experiments": {k: len(v) for k, v in phase1.items()},
    }
    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {os.path.join(out_dir, 'experiment_summary.json')}")


if __name__ == "__main__":
    main()
