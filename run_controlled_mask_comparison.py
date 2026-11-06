"""
run_controlled_mask_comparison.py — Phase B Step 8: Controlled Mask Strategy Comparison.

Head-to-head comparison of 4 mask strategies with paired statistical tests:
1. Random Masks (baseline)
2. AHM (Attention Head Masking — original method)
3. D-AHM (Divergence-guided head masking)
4. NC-AHM (Neural Collapse guided head masking)

For each experiment, we already have results from NC-AHM runs which include ALL 4 strategies.
This script aggregates them and performs:
- Paired t-test (each experiment is a paired observation)
- Wilcoxon signed-rank test (non-parametric)
- Effect size (Cohen's d)
- Win/Tie/Loss matrix

Usage:
    python run_controlled_mask_comparison.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def cohens_d(a, b):
    """Compute Cohen's d effect size for paired samples."""
    diff = np.array(a) - np.array(b)
    if diff.std() == 0:
        return 0.0
    return diff.mean() / diff.std()


def win_tie_loss(a, b, threshold=0.001):
    """Count wins, ties, losses for a vs b."""
    a, b = np.array(a), np.array(b)
    wins = np.sum(a > b + threshold)
    losses = np.sum(b > a + threshold)
    ties = len(a) - wins - losses
    return int(wins), int(ties), int(losses)


def main():
    # Load NC-AHM aggregated results (contains all 4 strategies)
    nc_ahm_path = os.path.join(SCRIPT_DIR, "outputs", "nc_ahm_results", "aggregated_results.json")
    with open(nc_ahm_path) as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} experiments from NC-AHM results")
    print("=" * 80)

    strategies = ["nc_ahm", "ahm", "dahm", "random"]
    strategy_names = {"nc_ahm": "NC-AHM", "ahm": "AHM", "dahm": "D-AHM", "random": "Random"}
    metrics = ["ahss_auroc", "combined_auroc"]
    metric_names = {"ahss_auroc": "AHSS AUROC", "combined_auroc": "Combined AUROC"}

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_experiments": len(all_results),
    }

    for metric_key in metrics:
        print(f"\n{'='*80}")
        print(f"METRIC: {metric_names[metric_key]}")
        print(f"{'='*80}")

        # Collect scores for each strategy
        scores = {}
        for s in strategies:
            scores[s] = []
            for r in all_results:
                scores[s].append(r[s][metric_key])
            scores[s] = np.array(scores[s])

        # Summary statistics
        print(f"\n{'Strategy':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 56)
        for s in strategies:
            print(f"{strategy_names[s]:<12} {scores[s].mean():>10.4f} {scores[s].std():>10.4f} "
                  f"{scores[s].min():>10.4f} {scores[s].max():>10.4f}")

        # Pairwise comparisons
        metric_output = {"summary": {}, "pairwise": {}}
        for s in strategies:
            metric_output["summary"][s] = {
                "mean": float(scores[s].mean()),
                "std": float(scores[s].std()),
                "min": float(scores[s].min()),
                "max": float(scores[s].max()),
            }

        print(f"\n{'Comparison':<20} {'Δ Mean':>10} {'t-stat':>10} {'p-value':>10} {'Cohen d':>10} {'W/T/L':>10} {'Signif':>8}")
        print("-" * 82)

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                key = f"{s1}_vs_{s2}"
                diff = scores[s1] - scores[s2]
                mean_diff = diff.mean()

                # Paired t-test
                if diff.std() > 0:
                    t_stat, p_val = stats.ttest_rel(scores[s1], scores[s2])
                else:
                    t_stat, p_val = 0.0, 1.0

                # Wilcoxon signed-rank (non-parametric)
                try:
                    w_stat, w_pval = stats.wilcoxon(scores[s1], scores[s2])
                except ValueError:
                    w_stat, w_pval = 0.0, 1.0

                # Effect size
                d = cohens_d(scores[s1], scores[s2])

                # Win/Tie/Loss
                w, t, l = win_tie_loss(scores[s1], scores[s2])

                signif = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                print(f"{strategy_names[s1]:>8} vs {strategy_names[s2]:<8} "
                      f"{mean_diff:>+10.4f} {t_stat:>10.3f} {p_val:>10.4f} {d:>10.3f} {w}/{t}/{l:>8} {signif:>8}")

                metric_output["pairwise"][key] = {
                    "mean_diff": float(mean_diff),
                    "t_statistic": float(t_stat),
                    "t_p_value": float(p_val),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_p_value": float(w_pval),
                    "cohens_d": float(d),
                    "wins": w, "ties": t, "losses": l,
                    "significant_at_005": bool(p_val < 0.05),
                    "significant_at_001": bool(p_val < 0.01),
                }

        output[metric_key] = metric_output

        # Win count matrix
        print(f"\nWin Count Matrix ({metric_names[metric_key]}):")
        print(f"{'':>12}", end="")
        for s in strategies:
            print(f"{strategy_names[s]:>12}", end="")
        print()
        for s1 in strategies:
            print(f"{strategy_names[s1]:>12}", end="")
            for s2 in strategies:
                if s1 == s2:
                    print(f"{'---':>12}", end="")
                else:
                    w, _, _ = win_tie_loss(scores[s1], scores[s2])
                    print(f"{w:>12}", end="")
            print()

    # Per-dataset breakdown
    print(f"\n\n{'='*80}")
    print("PER-DATASET SUBGROUP ANALYSIS")
    print(f"{'='*80}")

    subgroups = {
        "financedocs_intra": lambda r: r["dataset"] == "financedocs" and r["experiment_type"] == "intra",
        "financedocs_cross": lambda r: r["dataset"] == "financedocs" and r["experiment_type"] == "cross",
        "tobacco_intra": lambda r: r["dataset"] == "tobacco" and r["experiment_type"] == "intra",
        "tobacco_cross": lambda r: r["dataset"] == "tobacco" and r["experiment_type"] == "cross",
    }

    subgroup_output = {}
    for sg_name, sg_filter in subgroups.items():
        sg_results = [r for r in all_results if sg_filter(r)]
        if not sg_results:
            continue

        print(f"\n{sg_name} ({len(sg_results)} seeds):")
        sg_data = {}
        for metric_key in metrics:
            print(f"  {metric_names[metric_key]}:")
            for s in strategies:
                vals = [r[s][metric_key] for r in sg_results]
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                print(f"    {strategy_names[s]:<12} {mean_v:.4f} ± {std_v:.4f}")
                sg_data[f"{s}_{metric_key}_mean"] = float(mean_v)
                sg_data[f"{s}_{metric_key}_std"] = float(std_v)

            # Best strategy for this subgroup
            best = max(strategies, key=lambda s: np.mean([r[s][metric_key] for r in sg_results]))
            print(f"    BEST: {strategy_names[best]}")
            sg_data[f"best_{metric_key}"] = best

        subgroup_output[sg_name] = sg_data

    output["subgroups"] = subgroup_output

    # Rank analysis
    print(f"\n\n{'='*80}")
    print("MEAN RANK ANALYSIS")
    print(f"{'='*80}")

    for metric_key in metrics:
        print(f"\n{metric_names[metric_key]}:")
        ranks = {s: [] for s in strategies}
        for r in all_results:
            sorted_strats = sorted(strategies, key=lambda s: r[s][metric_key], reverse=True)
            for rank, s in enumerate(sorted_strats, 1):
                ranks[s].append(rank)

        for s in sorted(strategies, key=lambda s: np.mean(ranks[s])):
            mean_rank = np.mean(ranks[s])
            print(f"  {strategy_names[s]:<12} Mean Rank: {mean_rank:.2f}")

        # Friedman test
        friedman_data = [np.array([r[s][metric_key] for r in all_results]) for s in strategies]
        try:
            f_stat, f_pval = stats.friedmanchisquare(*friedman_data)
            print(f"  Friedman χ² = {f_stat:.3f}, p = {f_pval:.4f}")
            output[f"{metric_key}_friedman"] = {"chi2": float(f_stat), "p_value": float(f_pval)}
        except Exception as e:
            print(f"  Friedman test failed: {e}")

    # Save results
    out_dir = os.path.join(SCRIPT_DIR, "outputs", "controlled_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
