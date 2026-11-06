"""
run_final_aggregate.py — Create final paper-ready aggregated results tables.

Combines ALL experiment results into unified tables:
1. LayoutLMv3 full 17-method comparison (4 dataset configs × 5 seeds)
2. ViT full 17-method comparison (2 dataset configs × 5 seeds)
3. Mask strategy comparison (Random vs AHM vs D-AHM vs NC-AHM)
4. Near-OOD results
5. Layer decomposition profile
6. Ablation studies

Usage:
    python run_final_aggregate.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    outputs_dir = os.path.join(SCRIPT_DIR, "outputs")

    print("=" * 100)
    print("FINAL AGGREGATED RESULTS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # ================================================================
    # TABLE 1: LayoutLMv3 Full Method Comparison (from existing results)
    # ================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 1: LayoutLMv3-base OOD Detection — 17 Methods (mean ± std, 5 seeds)")
    print("=" * 100)

    # Load aggregated results
    agg_path = os.path.join(outputs_dir, "aggregated")
    if os.path.exists(agg_path):
        for fname in sorted(os.listdir(agg_path)):
            if fname.endswith('.json'):
                data = load_json(os.path.join(agg_path, fname))
                print(f"\n{fname}:")
                if isinstance(data, dict):
                    for method, vals in sorted(data.items()):
                        if isinstance(vals, dict) and 'auroc_mean' in vals:
                            print(f"  {method:<20} AUROC={vals['auroc_mean']:.4f}±{vals['auroc_std']:.4f}  "
                                  f"FPR@95={vals.get('fpr_mean', vals.get('fpr95_mean', 0)):.4f}±{vals.get('fpr_std', vals.get('fpr95_std', 0)):.4f}")

    # Load per-experiment AHSS results
    results_dir = os.path.join(outputs_dir, "results")
    experiments = {}
    seeds = [42, 123, 456, 789, 1024]

    for dataset in ["financedocs", "tobacco"]:
        for exp_type in ["intra", "cross"]:
            key = f"{dataset}_{exp_type}"
            exp_results = []
            for seed in seeds:
                exp_name = f"{key}_seed{seed}"
                result_path = os.path.join(results_dir, exp_name, "results.json")
                if os.path.exists(result_path):
                    data = load_json(result_path)
                    exp_results.append(data)
            if exp_results:
                experiments[key] = exp_results

    # Print LayoutLMv3 summary
    methods = ["ahss", "mc_dropout", "mahalanobis", "knn", "msp", "maxLogit",
               "energy", "kl", "gradnorm", "vim", "residual", "neco",
               "react", "ash", "mahAHM", "knnAHM", "mahAvgAvg_AHM"]

    for key, results in sorted(experiments.items()):
        print(f"\n{key} ({len(results)} seeds):")
        print(f"  {'Method':<20} {'AUROC':>20} {'FPR@95':>20}")
        print(f"  {'-'*60}")
        for m in methods:
            aurocs = []
            fprs = []
            for r in results:
                if m in r and isinstance(r[m], dict):
                    aurocs.append(r[m].get('auroc', 0))
                    fprs.append(r[m].get('fpr_at_95', r[m].get('fpr95', 0)))
            if aurocs:
                print(f"  {m:<20} {np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
                      f"{np.mean(fprs):.4f}±{np.std(fprs):.4f}")

    # ================================================================
    # TABLE 2: ViT Extended Results
    # ================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 2: ViT-B/16 OOD Detection — 17 Methods (mean ± std, 5 seeds)")
    print("=" * 100)

    vit_methods = ["ahss", "mahalanobis", "knn", "msp", "maxLogit", "energy", "kl",
                   "vim", "residual", "neco", "react", "ash", "mahAHM", "knnAHM", "combined"]

    for exp in ["cifar10_intra", "cifar10_svhn"]:
        agg_path_vit = os.path.join(outputs_dir, "vit_results", "aggregated",
                                     f"{exp}_extended_aggregated.json")
        if os.path.exists(agg_path_vit):
            data = load_json(agg_path_vit)
            print(f"\n{exp}:")
            print(f"  {'Method':<20} {'AUROC':>20} {'FPR@95':>20}")
            print(f"  {'-'*60}")
            for m in vit_methods:
                if m in data:
                    d = data[m]
                    print(f"  {m:<20} {d['auroc_mean']:.4f}±{d['auroc_std']:.4f}  "
                          f"{d['fpr_mean']:.4f}±{d['fpr_std']:.4f}")

    # MC-Dropout from original results
    print("\n  MC-Dropout (from original results):")
    for exp in ["cifar10_intra", "cifar10_svhn"]:
        mc_aurocs = []
        for seed in seeds:
            orig = os.path.join(outputs_dir, "vit_results", f"{exp}_seed{seed}", "results.json")
            if os.path.exists(orig):
                d = load_json(orig)
                mc = d.get('mc_dropout', {})
                if 'auroc' in mc:
                    mc_aurocs.append(mc['auroc'])
        if mc_aurocs:
            print(f"  {exp}: MC-Dropout AUROC = {np.mean(mc_aurocs):.4f}±{np.std(mc_aurocs):.4f}")

    # ================================================================
    # TABLE 3: Mask Strategy Comparison
    # ================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 3: Mask Strategy Comparison (Paired Tests, 20 experiments)")
    print("=" * 100)

    comp_path = os.path.join(outputs_dir, "controlled_comparison", "comparison_results.json")
    if os.path.exists(comp_path):
        comp = load_json(comp_path)
        for metric in ["ahss_auroc", "combined_auroc"]:
            if metric in comp:
                print(f"\n{metric}:")
                summary = comp[metric].get("summary", {})
                for s in ["nc_ahm", "ahm", "dahm", "random"]:
                    if s in summary:
                        d = summary[s]
                        print(f"  {s:<12} {d['mean']:.4f}±{d['std']:.4f} [{d['min']:.4f}, {d['max']:.4f}]")

                print(f"\n  Pairwise tests:")
                pairwise = comp[metric].get("pairwise", {})
                for pair, vals in pairwise.items():
                    sig = "***" if vals['t_p_value'] < 0.001 else "**" if vals['t_p_value'] < 0.01 else "*" if vals['t_p_value'] < 0.05 else "ns"
                    print(f"  {pair:<25} Δ={vals['mean_diff']:+.4f} p={vals['t_p_value']:.4f} d={vals['cohens_d']:.3f} "
                          f"W/T/L={vals['wins']}/{vals['ties']}/{vals['losses']} {sig}")

    # ================================================================
    # TABLE 4: Near-OOD Results
    # ================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 4: Near-OOD Regime (AHSS Advantage over Mahalanobis)")
    print("=" * 100)

    near_agg = os.path.join(outputs_dir, "near_ood_results", "aggregated_results.json")
    if os.path.exists(near_agg):
        near_data = load_json(near_agg)
        for r in near_data:
            print(f"\n{r['experiment']} (seed={r['seed']}):")
            maha_auroc = r.get('mahalanobis', {}).get('auroc', 0)
            ahss_auroc = r.get('ahss', {}).get('auroc', 0)
            comb_auroc = r.get('combined', {}).get('auroc', 0)
            comb_alpha = r.get('combined', {}).get('best_alpha', 'N/A')
            delta = r.get('delta_from_mah', 0)
            print(f"  Mahalanobis:     AUROC={maha_auroc:.4f}")
            print(f"  AHSS:            AUROC={ahss_auroc:.4f}")
            print(f"  Combined:        AUROC={comb_auroc:.4f} (α={comb_alpha})")
            print(f"  Δ(Combined-Maha): {delta:.4f}")

    # ================================================================
    # TABLE 5: Ablation Summary
    # ================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 5: Ablation Studies Summary")
    print("=" * 100)

    ablation_dir = os.path.join(outputs_dir, "ablation_results")
    if os.path.exists(ablation_dir):
        for fname in sorted(os.listdir(ablation_dir)):
            if fname.endswith('.json'):
                data = load_json(os.path.join(ablation_dir, fname))
                print(f"\n{fname.replace('.json', '')}:")
                for r in data.get("results", []):
                    param = r.get("T", r.get("p", r.get("F", "?")))
                    print(f"  {param}: AHSS={r['ahss_auroc']:.4f}, Combined={r['combined_auroc']:.4f}")

    # Save everything as JSON
    final_output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layoutlmv3_experiments": {k: len(v) for k, v in experiments.items()},
        "vit_experiments": ["cifar10_intra (5 seeds)", "cifar10_svhn (5 seeds)"],
    }

    final_path = os.path.join(outputs_dir, "FINAL_PAPER_TABLES.json")
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n\nSaved summary to {final_path}")


if __name__ == "__main__":
    main()
