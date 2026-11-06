# Paper statistics (computed by compute_paper_statistics.py)

Software: numpy 2.0.2, scipy 1.13.1. X-Mahalanobis = uniform variant. Tests are one-sided in the stated direction (a > b); Bonferroni threshold over 23 tests = 0.00217.

## 1. The 23 paired comparisons

| # | setting | a > b | n | Δ mean | a wins | Wilcoxon p (1-sided) | paired t p (1-sided) | d_z | BH-adjusted | Bonferroni-adjusted |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | financedocs_intra | mahAvgAvg_AHM > X-Maha | 5 | +0.0004 | 3/5 | 0.4062 | 0.3679 | 0.16 | 0.4247 | 1.0000 |
| 2 | financedocs_cross | mahAHM > Mahalanobis | 5 | +0.0004 | 3/5 | 0.2188 | 0.1091 | 0.65 | 0.2396 | 1.0000 |
| 3 | tobacco_intra | mahAHM > Mahalanobis | 5 | +0.0076 | 5/5 | 0.0312 | 4.78e-04 | 3.90 | 0.0359 | 0.7188 |
| 4 | tobacco_cross | mahAvgAvg_AHM > X-Maha (headline) | 15 | +0.0136 | 15/15 | 3.05e-05 | 1.64e-07 | 2.33 | 3.51e-04 | 7.02e-04 |
| 5 | tobacco_cross | mahAvgAvg_AHM > Mahalanobis | 5 | +0.0733 | 5/5 | 0.0312 | 0.0024 | 2.52 | 0.0359 | 0.7188 |
| 6 | financedocs_intra | mahAvgAvg_AHM > mahAHM | 5 | +0.0182 | 5/5 | 0.0312 | 0.0058 | 1.97 | 0.0359 | 0.7188 |
| 7 | tobacco_cross | mahAvgAvg_AHM > mahAHM | 5 | +0.0657 | 5/5 | 0.0312 | 8.23e-04 | 3.38 | 0.0359 | 0.7188 |
| 8 | cifar10_intra | Mahalanobis > mahAHM | 5 | +0.2570 | 5/5 | 0.0312 | 0.0013 | 2.97 | 0.0359 | 0.7188 |
| 9 | cifar10_intra | Mahalanobis > mahAvgAvg_AHM | 5 | +0.0074 | 5/5 | 0.0312 | 0.0025 | 2.49 | 0.0359 | 0.7188 |
| 10 | cifar100_svhn | X-Maha > mahAvgAvg_AHM | 5 | +0.0937 | 5/5 | 0.0312 | 1.34e-04 | 5.41 | 0.0359 | 0.7188 |
| 11 | cifar100_dtd | X-Maha > mahAvgAvg_AHM | 5 | +0.0001 | 5/5 | 0.0312 | 0.0092 | 1.72 | 0.0359 | 0.7188 |
| 12 | sst2_mnli | mahAHM > Mahalanobis | 5 | +0.0092 | 5/5 | 0.0312 | 3.37e-04 | 4.27 | 0.0359 | 0.7188 |
| 13 | sst2_agnews | mahAHM > Mahalanobis | 5 | +0.0074 | 5/5 | 0.0312 | 0.0090 | 1.73 | 0.0359 | 0.7188 |
| 14 | sst2_20news | mahAHM > X-Maha | 5 | -0.0076 | 3/5 | 0.5938 | 0.6314 | -0.16 | 0.5938 | 1.0000 |
| 15 | sst2_mnli | Mahalanobis > mahAvgAvg_AHM | 5 | +0.0341 | 5/5 | 0.0312 | 0.0449 | 1.00 | 0.0359 | 0.7188 |
| 16 | sst2_agnews | mahAvgAvg_AHM > Mahalanobis | 5 | +0.0163 | 5/5 | 0.0312 | 0.0361 | 1.09 | 0.0359 | 0.7188 |
| 17 | sst2_20news | mahAvgAvg_AHM > X-Maha (headline) | 15 | +0.0577 | 15/15 | 3.05e-05 | 6.39e-07 | 2.08 | 3.51e-04 | 7.02e-04 |
| 18 | cifar100_svhn | X-Maha > mahAHM | 5 | +0.2709 | 5/5 | 0.0312 | 3.68e-04 | 4.17 | 0.0359 | 0.7188 |
| 19 | cifar100_dtd | X-Maha > mahAHM | 5 | +0.3549 | 5/5 | 0.0312 | 0.0038 | 2.23 | 0.0359 | 0.7188 |
| 20 | sst2_mnli | Mahalanobis > AHSS | 5 | +0.5149 | 5/5 | 0.0312 | 0.0022 | 2.58 | 0.0359 | 0.7188 |
| 21 | sst2_agnews | Mahalanobis > AHSS | 5 | +0.6293 | 5/5 | 0.0312 | 0.0013 | 2.97 | 0.0359 | 0.7188 |
| 22 | sst2_agnews | mahAHM > MC-Dropout | 5 | +0.5757 | 5/5 | 0.0312 | 0.0022 | 2.58 | 0.0359 | 0.7188 |
| 23 | sst2_20news | mahAHM > MC-Dropout | 5 | +0.8337 | 5/5 | 0.0312 | 6.16e-06 | 11.79 | 0.0359 | 0.7188 |

Raw p ≤ 0.05: 20/23 · BH-adjusted ≤ 0.05: 20/23 · Bonferroni-adjusted ≤ 0.05: 2/23.

## 2. Headline comparisons

| comparison | n | Δ mean | a wins | range of Δ | Wilcoxon p (1-sided) | paired t p (1-sided) | d_z | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|---|
| sst2_20news:mahAvgAvg_AHM>X-Maha | 5 | +0.0718 | 5/5 | [+0.0299, +0.1186] | 0.0312 | 0.0038 | 2.23 | [+0.0475, +0.0976] |
| sst2_20news:mahAvgAvg_AHM>X-Maha | 15 | +0.0577 | 15/15 | [+0.0251, +0.1186] | 3.05e-05 | 6.39e-07 | 2.08 | [+0.0448, +0.0719] |
| tobacco_cross:mahAvgAvg_AHM>X-Maha | 5 | +0.0157 | 5/5 | [+0.0113, +0.0211] | 0.0312 | 5.31e-04 | 3.79 | [+0.0126, +0.0191] |
| tobacco_cross:mahAvgAvg_AHM>X-Maha | 15 | +0.0136 | 15/15 | [+0.0026, +0.0218] | 3.05e-05 | 1.64e-07 | 2.33 | [+0.0107, +0.0163] |

## 3. Headroom check (11 standard configurations, 5 seeds)

- gain = mahAvgAvg_AHM − X-Maha: Pearson r = -0.035 (p = 0.92), Spearman ρ = -0.273 (p = 0.42)
- gain = mahAvgAvg_AHM − Mahalanobis: Pearson r = -0.384 (p = 0.24), Spearman ρ = -0.136 (p = 0.69)

## 4. Uniform X-Mahalanobis AUROC (mean ± population std)

| configuration | n | AUROC |
|---|---|---|
| financedocs_intra | 5 | 0.9940 ± 0.0024 |
| financedocs_cross | 5 | 0.9949 ± 0.0027 |
| tobacco_intra | 5 | 0.9341 ± 0.0072 |
| tobacco_cross | 5 | 0.9178 ± 0.0083 |
| cifar10_intra | 5 | 0.9837 ± 0.0025 |
| cifar10_svhn | 5 | 0.9997 ± 0.0002 |
| cifar100_svhn | 5 | 0.9860 ± 0.0054 |
| cifar100_dtd | 5 | 1.0000 ± 0.0000 |
| sst2_mnli | 5 | 0.8920 ± 0.0425 |
| sst2_agnews | 5 | 0.9311 ± 0.0469 |
| sst2_20news | 5 | 0.9280 ± 0.0288 |
| tobacco_cross_n15 | 15 | 0.9180 ± 0.0101 |

## 5. Per-seed AUROC values behind the 11 standard configurations: 985

- financedocs_intra: 110 (22 methods × seeds)
- financedocs_cross: 110 (22 methods × seeds)
- tobacco_intra: 110 (22 methods × seeds)
- tobacco_cross: 110 (22 methods × seeds)
- cifar10_intra: 90 (18 methods × seeds)
- cifar10_svhn: 90 (18 methods × seeds)
- cifar100_svhn: 85 (17 methods × seeds)
- cifar100_dtd: 85 (17 methods × seeds)
- sst2_mnli: 65 (13 methods × seeds)
- sst2_agnews: 65 (13 methods × seeds)
- sst2_20news: 65 (13 methods × seeds)
