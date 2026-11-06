"""
run_paired_significance_tests.py — Statistical tests on all "we win" claims.

For each head-to-head comparison where we claim method A > method B:
- Paired Wilcoxon signed-rank test (non-parametric)
- Paired t-test (parametric)
- Cohen's d effect size
- 95% confidence interval on mean difference

No GPU needed. Loads per-seed results from JSON files.

Usage:
    python run_paired_significance_tests.py
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
BASE_OUTPUT = os.path.join(SCRIPT_DIR, "outputs")


def load_per_seed_results():
    """Load all per-seed AUROC results from existing JSON files."""
    all_results = {}

    # LayoutLMv3 results: from results/{exp}_seed{seed}/ood_results.json (list format)
    # and xmaha_results/{exp}_seed{seed}/xmaha_results.json (dict format)
    for exp in ["financedocs_intra", "financedocs_cross", "tobacco_intra", "tobacco_cross"]:
        all_results[exp] = {}
        for seed in SEEDS:
            # OOD results (list of {"method": ..., "auroc": ..., "fpr": ...})
            ood_path = os.path.join(BASE_OUTPUT, "results", f"{exp}_seed{seed}", "ood_results.json")
            if os.path.exists(ood_path):
                with open(ood_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        method_name = item.get("method", "")
                        auroc = item.get("auroc")
                        if method_name and auroc is not None:
                            if method_name not in all_results[exp]:
                                all_results[exp][method_name] = []
                            all_results[exp][method_name].append(auroc)

            # X-Maha results (dict with xmaha_optimized, xmaha_uniform keys)
            xmaha_path = os.path.join(BASE_OUTPUT, "xmaha_results", f"{exp}_seed{seed}", "xmaha_results.json")
            if os.path.exists(xmaha_path):
                with open(xmaha_path) as f:
                    xdata = json.load(f)
                for key in ["xmaha_optimized", "xmaha_uniform"]:
                    if key in xdata and isinstance(xdata[key], dict) and "auroc" in xdata[key]:
                        if key not in all_results[exp]:
                            all_results[exp][key] = []
                        all_results[exp][key].append(xdata[key]["auroc"])

    # ViT results
    for exp in ["cifar10_intra", "cifar10_svhn"]:
        all_results[exp] = {}
        for seed in SEEDS:
            ext_path = os.path.join(BASE_OUTPUT, "vit_results", f"{exp}_seed{seed}", "extended_results.json")
            if os.path.exists(ext_path):
                with open(ext_path) as f:
                    data = json.load(f)
                for method_name, method_data in data.items():
                    if isinstance(method_data, dict) and "auroc" in method_data:
                        if method_name not in all_results[exp]:
                            all_results[exp][method_name] = []
                        all_results[exp][method_name].append(method_data["auroc"])

            # mahAvgAvg_AHM (new)
            mahavg_path = os.path.join(BASE_OUTPUT, "vit_results", f"{exp}_seed{seed}",
                                       "mahavgavg_ahm_results.json")
            if os.path.exists(mahavg_path):
                with open(mahavg_path) as f:
                    data = json.load(f)
                if "auroc" in data:
                    if "mahAvgAvg_AHM" not in all_results[exp]:
                        all_results[exp]["mahAvgAvg_AHM"] = []
                    all_results[exp]["mahAvgAvg_AHM"].append(data["auroc"])

    # CIFAR-100 results
    for ood in ["svhn", "dtd"]:
        exp = f"cifar100_{ood}"
        all_results[exp] = {}
        for seed in SEEDS:
            res_path = os.path.join(BASE_OUTPUT, "cifar100_results", f"{exp}_seed{seed}", "full_results.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    data = json.load(f)
                for method_name, method_data in data.items():
                    if isinstance(method_data, dict) and "auroc" in method_data:
                        if method_name not in all_results[exp]:
                            all_results[exp][method_name] = []
                        all_results[exp][method_name].append(method_data["auroc"])

            # mahAvgAvg_AHM (new)
            mahavg_path = os.path.join(BASE_OUTPUT, "cifar100_results", f"{exp}_seed{seed}",
                                       "mahavgavg_ahm_results.json")
            if os.path.exists(mahavg_path):
                with open(mahavg_path) as f:
                    data = json.load(f)
                if "auroc" in data:
                    if "mahAvgAvg_AHM" not in all_results[exp]:
                        all_results[exp]["mahAvgAvg_AHM"] = []
                    all_results[exp]["mahAvgAvg_AHM"].append(data["auroc"])

    # RoBERTa results
    for ood in ["mnli", "agnews", "20news"]:
        exp = f"sst2_{ood}"
        all_results[exp] = {}
        for seed in SEEDS:
            res_path = os.path.join(BASE_OUTPUT, "roberta_results", f"{exp}_seed{seed}", "full_results.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    data = json.load(f)
                for method_name, method_data in data.items():
                    if isinstance(method_data, dict) and "auroc" in method_data:
                        if method_name not in all_results[exp]:
                            all_results[exp][method_name] = []
                        all_results[exp][method_name].append(method_data["auroc"])

            # mahAvgAvg_AHM (new)
            mahavg_path = os.path.join(BASE_OUTPUT, "roberta_results", f"{exp}_seed{seed}",
                                       "mahavgavg_ahm_results.json")
            if os.path.exists(mahavg_path):
                with open(mahavg_path) as f:
                    data = json.load(f)
                if "auroc" in data:
                    if "mahAvgAvg_AHM" not in all_results[exp]:
                        all_results[exp]["mahAvgAvg_AHM"] = []
                    all_results[exp]["mahAvgAvg_AHM"].append(data["auroc"])

    return all_results


def paired_test(a_scores, b_scores, method_a, method_b, setting):
    """
    Run paired Wilcoxon + t-test on two matched score arrays.
    Returns dict with test results.
    """
    a = np.array(a_scores)
    b = np.array(b_scores)
    n = min(len(a), len(b))

    if n < 3:
        return {"error": f"Only {n} paired samples, need ≥3", "n": n}

    a = a[:n]
    b = b[:n]
    diff = a - b

    result = {
        "setting": setting,
        "method_a": method_a,
        "method_b": method_b,
        "n": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
        "a_wins": int(np.sum(diff > 0)),
        "b_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }

    # Paired t-test
    if np.std(diff) > 0:
        t_stat, t_pval = stats.ttest_rel(a, b)
        result["t_statistic"] = float(t_stat)
        result["t_pvalue"] = float(t_pval)
    else:
        result["t_statistic"] = float("inf") if np.mean(diff) > 0 else float("-inf")
        result["t_pvalue"] = 0.0 if np.all(diff != 0) else 1.0

    # Wilcoxon signed-rank test
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) >= 1:
        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="greater")
            result["wilcoxon_statistic"] = float(w_stat)
            result["wilcoxon_pvalue"] = float(w_pval)
        except ValueError:
            result["wilcoxon_statistic"] = None
            result["wilcoxon_pvalue"] = None
            result["wilcoxon_error"] = "Could not compute"
    else:
        result["wilcoxon_statistic"] = None
        result["wilcoxon_pvalue"] = 1.0
        result["wilcoxon_note"] = "All differences are zero"

    # Cohen's d
    if result["std_diff"] > 0:
        result["cohens_d"] = float(np.mean(diff) / result["std_diff"])
    else:
        result["cohens_d"] = float("inf") if np.mean(diff) != 0 else 0.0

    # 95% CI on mean difference
    if n > 1 and result["std_diff"] > 0:
        se = result["std_diff"] / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        result["ci_95_lower"] = float(np.mean(diff) - t_crit * se)
        result["ci_95_upper"] = float(np.mean(diff) + t_crit * se)
    else:
        result["ci_95_lower"] = float(np.mean(diff))
        result["ci_95_upper"] = float(np.mean(diff))

    return result


def run_all_tests():
    """Run paired tests on all 'we win' claims."""
    all_results = load_per_seed_results()

    # Define all comparisons we want to test
    # Format: (setting, method_a (ours/better), method_b (baseline))
    comparisons = [
        # LayoutLMv3: mahAvgAvg_AHM vs X-Maha
        ("financedocs_intra", "mahAvgAvg_AHM", "X-Maha_optimized"),
        ("financedocs_cross", "mahAHM", "Mahalanobis"),
        ("tobacco_intra", "mahAHM", "Mahalanobis"),
        ("tobacco_cross", "mahAvgAvg_AHM", "X-Maha_optimized"),
        ("tobacco_cross", "mahAvgAvg_AHM", "Mahalanobis"),

        # LayoutLMv3: mahAvgAvg_AHM vs mahAHM
        ("financedocs_intra", "mahAvgAvg_AHM", "mahAHM"),
        ("tobacco_cross", "mahAvgAvg_AHM", "mahAHM"),

        # ViT: AHSS vs MC-Dropout (we claim MC-Dropout fails)
        ("cifar10_intra", "ahss", "mc_dropout"),
        ("cifar10_svhn", "ahss", "mc_dropout"),

        # ViT: X-Maha vs our methods (we honestly report X-Maha wins)
        ("cifar10_svhn", "X-Maha_optimized", "mahAHM"),
        ("cifar10_intra", "mahalanobis", "mahAHM"),

        # ViT mahAvgAvg_AHM (new) vs existing best
        ("cifar10_intra", "mahAvgAvg_AHM", "mahalanobis"),
        ("cifar10_svhn", "mahAvgAvg_AHM", "X-Maha_optimized"),
        ("cifar100_svhn", "mahAvgAvg_AHM", "X-Maha_optimized"),
        ("cifar100_dtd", "mahAvgAvg_AHM", "X-Maha_optimized"),

        # RoBERTa: mahAHM vs Mahalanobis
        ("sst2_mnli", "mahAHM", "Mahalanobis"),
        ("sst2_agnews", "mahAHM", "Mahalanobis"),
        ("sst2_20news", "mahAHM", "X-Maha_optimized"),

        # RoBERTa mahAvgAvg_AHM (new)
        ("sst2_mnli", "mahAvgAvg_AHM", "Mahalanobis"),
        ("sst2_agnews", "mahAvgAvg_AHM", "Mahalanobis"),
        ("sst2_20news", "mahAvgAvg_AHM", "X-Maha_optimized"),

        # CIFAR-100: X-Maha vs mahAHM (honest: X-Maha wins)
        ("cifar100_svhn", "X-Maha_optimized", "mahAHM"),
        ("cifar100_dtd", "X-Maha_optimized", "mahAHM"),

        # AHSS polarity checks on RoBERTa (negative AUROC)
        ("sst2_mnli", "Mahalanobis", "AHSS"),
        ("sst2_agnews", "Mahalanobis", "AHSS"),

        # MC-Dropout vs random (0.5) checks — using mahAHM as proxy
        ("cifar10_intra", "mahAHM", "MC-Dropout"),
        ("sst2_agnews", "mahAHM", "MC-Dropout"),
        ("sst2_20news", "mahAHM", "MC-Dropout"),
    ]

    test_results = []
    logger.info(f"\n{'='*80}")
    logger.info("PAIRED SIGNIFICANCE TESTS ON ALL HEAD-TO-HEAD CLAIMS")
    logger.info(f"{'='*80}\n")

    # Handle method name case variations
    def get_scores(exp_data, method_name):
        """Try multiple case variations for method name."""
        variations = [
            method_name,
            method_name.lower(),
            method_name.replace("-", "_"),
            method_name.replace("_", "-"),
        ]
        # Common aliases
        aliases = {
            "mc_dropout": ["MC-Dropout", "mc_dropout", "MC_Dropout", "mcdropout"],
            "mahalanobis": ["Mahalanobis", "mahalanobis", "maha"],
            "ahss": ["AHSS", "ahss"],
            "knn": ["kNN", "knn", "KNN"],
            "mahAHM": ["mahAHM", "mah_ahm", "mahahm"],
            "mahAvgAvg_AHM": ["mahAvgAvg_AHM", "mahavgavg_ahm"],
            "X-Maha_optimized": ["X-Maha_optimized", "x_maha_optimized", "X_Maha_optimized", "xmaha_optimized"],
            "X-Maha_uniform": ["X-Maha_uniform", "x_maha_uniform", "xmaha_uniform"],
        }
        all_names = variations.copy()
        for key, vals in aliases.items():
            if method_name.lower() == key.lower() or method_name in vals:
                all_names.extend(vals)
        for name in all_names:
            if name in exp_data:
                return exp_data[name]
        return None

    for setting, method_a, method_b in comparisons:
        if setting not in all_results:
            logger.warning(f"  SKIP: {setting} — no results found")
            continue

        exp_data = all_results[setting]
        a_scores = get_scores(exp_data, method_a)
        b_scores = get_scores(exp_data, method_b)

        if a_scores is None:
            logger.warning(f"  SKIP: {setting} / {method_a} — not found (available: {list(exp_data.keys())})")
            continue
        if b_scores is None:
            logger.warning(f"  SKIP: {setting} / {method_b} — not found (available: {list(exp_data.keys())})")
            continue

        result = paired_test(a_scores, b_scores, method_a, method_b, setting)
        test_results.append(result)

        sig = ""
        if result.get("wilcoxon_pvalue") is not None and result["wilcoxon_pvalue"] < 0.10:
            sig = " *"
        if result.get("wilcoxon_pvalue") is not None and result["wilcoxon_pvalue"] < 0.05:
            sig = " **"

        logger.info(f"  {setting:<25} {method_a} vs {method_b}")
        logger.info(f"    Δ = {result['mean_diff']:+.4f} ({result['a_wins']}/{result['b_wins']}/{result['ties']} W/L/T)")
        if result.get("wilcoxon_pvalue") is not None:
            logger.info(f"    Wilcoxon p = {result['wilcoxon_pvalue']:.4f}{sig}")
        if result.get("t_pvalue") is not None:
            logger.info(f"    t-test  p = {result['t_pvalue']:.4f}")
        logger.info(f"    Cohen's d = {result['cohens_d']:.3f}, "
                    f"95% CI = [{result.get('ci_95_lower', 0):.4f}, {result.get('ci_95_upper', 0):.4f}]")
        logger.info("")

    # Save results
    out_dir = os.path.join(BASE_OUTPUT, "statistical_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paired_significance_tests.json")
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"\nAll test results saved: {out_path}")

    # Summary table
    logger.info(f"\n{'='*100}")
    logger.info(f"{'Setting':<25} {'A > B':<35} {'Δ':>8} {'Wilcox-p':>10} {'t-p':>10} {'d':>8} {'Sig':>5}")
    logger.info("-" * 100)
    for r in test_results:
        if "error" in r:
            continue
        sig_mark = ""
        wp = r.get("wilcoxon_pvalue")
        if wp is not None:
            if wp < 0.01:
                sig_mark = "***"
            elif wp < 0.05:
                sig_mark = "**"
            elif wp < 0.10:
                sig_mark = "*"
        comparison = f"{r['method_a']} > {r['method_b']}"
        logger.info(f"{r['setting']:<25} {comparison:<35} {r['mean_diff']:>+8.4f} "
                    f"{wp if wp is not None else 'N/A':>10} "
                    f"{r.get('t_pvalue', 'N/A'):>10} "
                    f"{r['cohens_d']:>8.3f} {sig_mark:>5}")
    logger.info(f"{'='*100}")
    logger.info("* p<0.10, ** p<0.05, *** p<0.01 (one-sided Wilcoxon)")

    return test_results


if __name__ == "__main__":
    run_all_tests()
