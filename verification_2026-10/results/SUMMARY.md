# Verification — results summary

Generated 2026-10-08 18:11 UTC — **FINAL** (all queue steps finished)

Environment: `{"python": "3.9.6", "torch": "2.5.1", "transformers": "4.57.6", "numpy": "2.0.2", "sklearn": "1.6.1", "scipy": "1.13.1", "device": "gpu"}`

## Gate 0 — re-run vs original parity

- **layoutlmv3**: PASS=True; min cosine 0.999999882
- **roberta**: PASS=True; min cosine 1.000000000
- **vit**: PASS=True; min cosine 0.999972699

## E1-R — original Table 4 construction (model trained on the held-out class) reproduced in the re-run

| seed | Maha (re-run) | Maha (original) | kNN k=50 (re-run) | kNN (original) | gate |
|---|---|---|---|---|---|
| 42 | 0.4958 | 0.4958 | 0.5823 | 0.5824 | True |
| 123 | 0.5293 | 0.5293 | 0.6864 | 0.6864 | True |
| 456 | 0.7193 | 0.7193 | 0.8286 | 0.8286 | True |
| 789 | 0.7224 | 0.7224 | 0.7867 | 0.7868 | True |
| 1024 | 0.3063 | 0.3063 | 0.2847 | 0.2847 | True |

Mean: Maha 0.5546 ± 0.1556, kNN 0.6338 ± 0.1942 (paper Table 4: 0.555, 0.634). ALL_GATES_PASS=True

## E1-C — corrected near-OOD with leave-class-out models (OOD class never trained on)

Std.-OOD reference (submitted Table 1, Tob-Intra, Advertisement): Maha 0.970, kNN 0.961.

| held-out class | seeds | split gate | Maha | kNN k=50 | kNN k=10 | MSP | Energy | Letter-only Maha | Letter-only kNN50 |
|---|---|---|---|---|---|---|---|---|---|
| Memo | 5 | 5/5 | 0.9371 ± 0.0094 | 0.9240 ± 0.0156 | 0.9308 ± 0.0124 | 0.8413 ± 0.0439 | 0.8203 ± 0.0510 | 0.9657 ± 0.0109 | 0.9594 ± 0.0143 |
| Form | 5 | 5/5 | 0.9140 ± 0.0150 | 0.9021 ± 0.0155 | 0.9080 ± 0.0137 | 0.8256 ± 0.0479 | 0.8174 ± 0.0441 | 0.9353 ± 0.0184 | 0.9258 ± 0.0167 |
| Resume | 5 | 5/5 | 0.8801 ± 0.0089 | 0.8825 ± 0.0109 | 0.8852 ± 0.0090 | 0.8566 ± 0.0169 | 0.8287 ± 0.0277 | 0.9343 ± 0.0071 | 0.9342 ± 0.0174 |
| Scientific | 5 | 5/5 | 0.8624 ± 0.0171 | 0.8489 ± 0.0275 | 0.8545 ± 0.0249 | 0.8336 ± 0.0358 | 0.7997 ± 0.0319 | 0.8739 ± 0.0159 | 0.8390 ± 0.0353 |

**E1-A AHSS (leave-Memo-out, 25 per-layer random masks, 5 seeds):** all-ID 0.9001 ± 0.0128; Letter-only 0.7841 ± 0.0643 (submitted Table 4: 0.069).

## E2_roberta_base — 5 seeds

**Full data, unmasked (reproduction gate: Maha / kNN / X-Maha-uniform within ±0.002 of the original run)**

| OOD | gate | Maha | kNN10 | X-Maha-unif | mahAvgAvg pad-incl | mahAvgAvg pad-excl | LENGTH ONLY | mean tokens ID/OOD |
|---|---|---|---|---|---|---|---|---|
| 20news | 5/5 | 0.9265 ± 0.0230 | 0.9236 ± 0.0274 | 0.9280 ± 0.0288 | 0.9937 ± 0.0027 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 14.4 / 126.5 |
| agnews | 5/5 | 0.9398 ± 0.0175 | 0.9276 ± 0.0220 | 0.9311 ± 0.0469 | 0.9196 ± 0.0151 | 0.9770 ± 0.0059 | 0.9871 ± 0.0006 | 14.4 / 53.7 |
| mnli | 5/5 | 0.8987 ± 0.0254 | 0.8856 ± 0.0294 | 0.8920 ± 0.0425 | 0.8544 ± 0.0315 | 0.9328 ± 0.0112 | 0.7510 ± 0.0043 | 14.4 / 25.6 |
| 20news_noheader | – | 0.8848 ± 0.0483 | 0.8818 ± 0.0462 | 0.9044 ± 0.0459 | 0.9268 ± 0.0158 | 0.9806 ± 0.0049 | 0.9917 ± 0.0003 | 14.4 / 101.5 |

Full-data paired differences vs X-Maha-uniform, PROTOCOL §E2 question (a) (descriptive; seeds with difference > 0):

- 20news: pool WITH padding − X-Maha +0.0657 ± 0.0274 (5/5); pool WITHOUT padding − X-Maha +0.0720 ± 0.0288 (5/5)
- agnews: pool WITH padding − X-Maha -0.0115 ± 0.0329 (2/5); pool WITHOUT padding − X-Maha +0.0459 ± 0.0472 (4/5)
- mnli: pool WITH padding − X-Maha -0.0377 ± 0.0177 (0/5); pool WITHOUT padding − X-Maha +0.0408 ± 0.0371 (4/5)
- 20news_noheader: pool WITH padding − X-Maha +0.0223 ± 0.0367 (3/5); pool WITHOUT padding − X-Maha +0.0762 ± 0.0424 (5/5)

**Subsample (3,000 train / 1,000 eval / 1,000 OOD), 25 original masks: asym (original) vs sym (corrected)**

| OOD | maha_unmasked | mahAHM_asym | mahAHM_sym | mahAHM_top5_sym | knnAHM10_asym | knnAHM10_sym | knnAHM10_top5_sym | mahAvgAvg_avg_pad_unmasked | mahAvgAvg_AHM_avg_pad_asym | mahAvgAvg_AHM_avg_pad_sym | mahAvgAvg_AHM_avg_pad_top5_sym | mahAvgAvg_avg_nopad_unmasked | mahAvgAvg_AHM_avg_nopad_asym | mahAvgAvg_AHM_avg_nopad_sym | mahAvgAvg_AHM_avg_nopad_top5_sym | ahss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20news | 0.9226 ± 0.0277 | 0.9160 ± 0.0239 | 0.9039 ± 0.0316 | 0.9092 ± 0.0327 | 0.9070 ± 0.0468 | 0.8928 ± 0.0519 | 0.8980 ± 0.0509 | 0.9895 ± 0.0039 | 0.9996 ± 0.0004 | 0.9990 ± 0.0009 | 0.9987 ± 0.0014 | 0.9999 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.1883 ± 0.1555 |
|  ↳ original full-data (original asym) | 0.9265 ± 0.0230 | 0.9204 ± 0.0226 |  |  | 0.9021 ± 0.0444 |  |  |  | 0.9998 ± 0.0002 |  |  |  |  |  |  | 0.1924 ± 0.1581 |
| agnews | 0.9362 ± 0.0201 | 0.9426 ± 0.0161 | 0.9395 ± 0.0163 | 0.9307 ± 0.0149 | 0.9243 ± 0.0336 | 0.9181 ± 0.0288 | 0.9090 ± 0.0220 | 0.9120 ± 0.0138 | 0.9491 ± 0.0142 | 0.9408 ± 0.0129 | 0.9367 ± 0.0157 | 0.9749 ± 0.0057 | 0.9811 ± 0.0043 | 0.9769 ± 0.0042 | 0.9755 ± 0.0040 | 0.3088 ± 0.1757 |
|  ↳ original full-data (original asym) | 0.9398 ± 0.0175 | 0.9472 ± 0.0141 |  |  | 0.9235 ± 0.0190 |  |  |  | 0.9561 ± 0.0126 |  |  |  |  |  |  | 0.3104 ± 0.1812 |
| mnli | 0.8957 ± 0.0276 | 0.9037 ± 0.0285 | 0.9027 ± 0.0287 | 0.8970 ± 0.0255 | 0.8820 ± 0.0385 | 0.8790 ± 0.0359 | 0.8745 ± 0.0280 | 0.8506 ± 0.0305 | 0.8590 ± 0.0326 | 0.8699 ± 0.0278 | 0.8607 ± 0.0243 | 0.9277 ± 0.0116 | 0.9314 ± 0.0116 | 0.9311 ± 0.0088 | 0.9270 ± 0.0116 | 0.3781 ± 0.1634 |
|  ↳ original full-data (original asym) | 0.8987 ± 0.0254 | 0.9079 ± 0.0260 |  |  | 0.8838 ± 0.0293 |  |  |  | 0.8647 ± 0.0337 |  |  |  |  |  |  | 0.3838 ± 0.1655 |
| 20news_noheader | 0.8808 ± 0.0478 | 0.8661 ± 0.0361 | 0.8515 ± 0.0380 | 0.8526 ± 0.0462 | 0.8612 ± 0.0612 | 0.8434 ± 0.0579 | 0.8410 ± 0.0553 | 0.9095 ± 0.0190 | 0.9568 ± 0.0118 | 0.9501 ± 0.0135 | 0.9493 ± 0.0147 | 0.9771 ± 0.0041 | 0.9817 ± 0.0038 | 0.9808 ± 0.0030 | 0.9808 ± 0.0042 | 0.3405 ± 0.1781 |

Pooled over 20news/agnews/mnli × seeds: maha_unmasked 0.9182; mahAHM_asym 0.9207; mahAHM_sym 0.9154; mahAHM_top5_sym 0.9123; knnAHM10_asym 0.9044; knnAHM10_sym 0.8966; knnAHM10_top5_sym 0.8938; mahAvgAvg_avg_pad_unmasked 0.9174; mahAvgAvg_AHM_avg_pad_asym 0.9359; mahAvgAvg_AHM_avg_pad_sym 0.9365; mahAvgAvg_AHM_avg_pad_top5_sym 0.9320; mahAvgAvg_avg_nopad_unmasked 0.9675; mahAvgAvg_AHM_avg_nopad_asym 0.9708; mahAvgAvg_AHM_avg_nopad_sym 0.9693; mahAvgAvg_AHM_avg_nopad_top5_sym 0.9675; ahss 0.2917

**Masked-part reproduction gate (PROTOCOL §E2: asym within ±0.03 of the stored original full-data value)**

- mahAHM_asym vs original mahAHM: 15/15 within ±0.03 (max |Δ| 0.0110)
- knnAHM10_asym vs original knnAHM: 14/15 within ±0.03 (max |Δ| 0.0396); outside (seed, OOD, Δ): [(456, 'agnews', 0.0396)]
- mahAvgAvg_AHM_avg_pad_asym vs original mahAvgAvg_AHM: 15/15 within ±0.03 (max |Δ| 0.0115)

**Paired per-seed differences, same subsample: mean ± std (seeds with difference > 0)**

| contrast | status | 20news | agnews | mnli | 20news_noheader |
|---|---|---|---|---|---|
| sym − asym, mahAHM | pre-registered comparison, descriptive | -0.0121 ± 0.0088 (0/5) | -0.0031 ± 0.0033 (1/5) | -0.0009 ± 0.0026 (1/5) | -0.0146 ± 0.0114 (1/5) |
| sym − asym, knnAHM | pre-registered comparison, descriptive | -0.0143 ± 0.0077 (0/5) | -0.0062 ± 0.0060 (0/5) | -0.0030 ± 0.0038 (2/5) | -0.0178 ± 0.0116 (0/5) |
| masking gain, mahAHM asym | pre-registered comparison, descriptive | -0.0066 ± 0.0138 (1/5) | +0.0064 ± 0.0050 (5/5) | +0.0080 ± 0.0019 (5/5) | -0.0147 ± 0.0144 (1/5) |
| masking gain, mahAHM sym | pre-registered comparison, descriptive | -0.0187 ± 0.0198 (1/5) | +0.0033 ± 0.0041 (3/5) | +0.0070 ± 0.0036 (5/5) | -0.0293 ± 0.0153 (0/5) |
| masking gain, pool WITH padding, asym | pre-registered comparison, descriptive | +0.0101 ± 0.0036 (5/5) | +0.0372 ± 0.0081 (5/5) | +0.0083 ± 0.0058 (5/5) | +0.0473 ± 0.0080 (5/5) |
| masking gain, pool WITHOUT padding, asym | pre-registered comparison, descriptive | +0.0001 ± 0.0001 (5/5) | +0.0062 ± 0.0017 (5/5) | +0.0037 ± 0.0021 (4/5) | +0.0047 ± 0.0008 (5/5) |
| masking gain, pool WITHOUT padding, sym | pre-registered comparison, descriptive | +0.0001 ± 0.0001 (5/5) | +0.0021 ± 0.0025 (4/5) | +0.0034 ± 0.0038 (4/5) | +0.0038 ± 0.0012 (5/5) |
| no-padding pool (no masks) − paper's mahAvgAvg_AHM | EXPLORATORY (post hoc, not pre-registered) | +0.0003 ± 0.0003 (5/5) | +0.0257 ± 0.0143 (5/5) | +0.0687 ± 0.0306 (5/5) | +0.0202 ± 0.0114 (5/5) |

## E3_roberta_large — 5 seeds


**Subsample (3,000 train / 1,000 eval / 1,000 OOD), 25 original masks: asym (original) vs sym (corrected)**

| OOD | maha_unmasked | mahAHM_asym | mahAHM_sym | mahAHM_top5_sym | knnAHM10_asym | knnAHM10_sym | knnAHM10_top5_sym | mahAvgAvg_avg_pad_unmasked | mahAvgAvg_AHM_avg_pad_asym | mahAvgAvg_AHM_avg_pad_sym | mahAvgAvg_AHM_avg_pad_top5_sym | mahAvgAvg_avg_nopad_unmasked | mahAvgAvg_AHM_avg_nopad_asym | mahAvgAvg_AHM_avg_nopad_sym | mahAvgAvg_AHM_avg_nopad_top5_sym | ahss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20news | 0.9538 ± 0.0042 | 0.4517 ± 0.1314 | 0.9513 ± 0.0093 | 0.9477 ± 0.0125 | 0.7315 ± 0.0690 | 0.9339 ± 0.0126 | 0.9292 ± 0.0219 | 0.9926 ± 0.0034 | 0.9994 ± 0.0006 | 0.9998 ± 0.0003 | 0.9996 ± 0.0003 | 0.9937 ± 0.0024 | 0.9967 ± 0.0019 | 0.9983 ± 0.0011 | 0.9979 ± 0.0011 | 0.4196 ± 0.1402 |
|  ↳ original full-data (original asym) | 0.9563 ± 0.0042 | 0.3743 ± 0.1862 |  |  | 0.4534 ± 0.1007 |  |  |  |  |  |  |  |  |  |  | 0.4243 ± 0.1342 |
| agnews | 0.9056 ± 0.0253 | 0.3577 ± 0.1786 | 0.8917 ± 0.0376 | 0.8837 ± 0.0435 | 0.6313 ± 0.0986 | 0.8861 ± 0.0310 | 0.8680 ± 0.0560 | 0.9684 ± 0.0080 | 0.9813 ± 0.0030 | 0.9874 ± 0.0034 | 0.9819 ± 0.0113 | 0.9610 ± 0.0036 | 0.9633 ± 0.0048 | 0.9691 ± 0.0041 | 0.9676 ± 0.0037 | 0.5709 ± 0.0832 |
|  ↳ original full-data (original asym) | 0.9072 ± 0.0221 | 0.3158 ± 0.1638 |  |  | 0.4494 ± 0.1163 |  |  |  |  |  |  |  |  |  |  | 0.5729 ± 0.0720 |
| mnli | 0.8502 ± 0.0346 | 0.3703 ± 0.1532 | 0.8473 ± 0.0438 | 0.8376 ± 0.0484 | 0.6184 ± 0.0594 | 0.8356 ± 0.0276 | 0.8192 ± 0.0496 | 0.9119 ± 0.0149 | 0.9222 ± 0.0074 | 0.9375 ± 0.0078 | 0.9297 ± 0.0208 | 0.9077 ± 0.0100 | 0.9060 ± 0.0107 | 0.9136 ± 0.0092 | 0.9127 ± 0.0103 | 0.6253 ± 0.0693 |
|  ↳ original full-data (original asym) | 0.8521 ± 0.0247 | 0.3443 ± 0.1484 |  |  | 0.4993 ± 0.0826 |  |  |  |  |  |  |  |  |  |  | 0.6218 ± 0.0660 |

Pooled over 20news/agnews/mnli × seeds: maha_unmasked 0.9032; mahAHM_asym 0.3932; mahAHM_sym 0.8968; mahAHM_top5_sym 0.8896; knnAHM10_asym 0.6604; knnAHM10_sym 0.8852; knnAHM10_top5_sym 0.8722; mahAvgAvg_avg_pad_unmasked 0.9576; mahAvgAvg_AHM_avg_pad_asym 0.9677; mahAvgAvg_AHM_avg_pad_sym 0.9749; mahAvgAvg_AHM_avg_pad_top5_sym 0.9704; mahAvgAvg_avg_nopad_unmasked 0.9541; mahAvgAvg_AHM_avg_nopad_asym 0.9553; mahAvgAvg_AHM_avg_nopad_sym 0.9604; mahAvgAvg_AHM_avg_nopad_top5_sym 0.9594; ahss 0.5386

**Reproduction gate (PROTOCOL §E3): pooled mahAHM_asym 0.3932 vs reported 0.3448 (±0.05) → PASS**

Pre-stated test (sym > asym, one-sided exact Wilcoxon, n=15): mahAHM p=3.0517578125e-05, knnAHM p=3.0517578125e-05
Same test for the top-5 (Algorithm-1-style) variant: mahAHM_top5_sym > mahAHM_asym p=3.0517578125e-05

**Paired per-seed differences, same subsample: mean ± std (seeds with difference > 0)**

| contrast | status | 20news | agnews | mnli |
|---|---|---|---|---|
| sym − asym, mahAHM | pre-registered comparison, descriptive | +0.4996 ± 0.1240 (5/5) | +0.5341 ± 0.1413 (5/5) | +0.4770 ± 0.1102 (5/5) |
| sym − asym, knnAHM | pre-registered comparison, descriptive | +0.2025 ± 0.0610 (5/5) | +0.2548 ± 0.0723 (5/5) | +0.2172 ± 0.0378 (5/5) |
| masking gain, mahAHM asym | pre-registered comparison, descriptive | -0.5021 ± 0.1305 (0/5) | -0.5480 ± 0.1586 (0/5) | -0.4798 ± 0.1218 (0/5) |
| masking gain, mahAHM sym | pre-registered comparison, descriptive | -0.0025 ± 0.0090 (3/5) | -0.0139 ± 0.0211 (2/5) | -0.0028 ± 0.0146 (3/5) |
| masking gain, pool WITH padding, asym | pre-registered comparison, descriptive | +0.0068 ± 0.0029 (5/5) | +0.0130 ± 0.0053 (5/5) | +0.0103 ± 0.0078 (4/5) |
| masking gain, pool WITHOUT padding, asym | pre-registered comparison, descriptive | +0.0030 ± 0.0009 (5/5) | +0.0024 ± 0.0017 (5/5) | -0.0017 ± 0.0028 (1/5) |
| masking gain, pool WITHOUT padding, sym | pre-registered comparison, descriptive | +0.0046 ± 0.0015 (5/5) | +0.0082 ± 0.0017 (5/5) | +0.0059 ± 0.0024 (5/5) |
| no-padding pool (no masks) − paper's mahAvgAvg_AHM | EXPLORATORY (post hoc, not pre-registered) | -0.0057 ± 0.0019 (0/5) | -0.0204 ± 0.0058 (0/5) | -0.0146 ± 0.0156 (2/5) |

## E4 — LayoutLMv3 MC-Dropout, fine-tuned weights loaded strictly (tobacco_intra)

| seed | weights gate | AUROC (re-run) | AUROC (paper) | |Δ| | var ID | var OOD |
|---|---|---|---|---|---|---|
| 1024 | True | 0.9476 | 0.9512 | 0.0035 | 1.300e-01 | 2.147e-01 |
| 123 | True | 0.9289 | 0.9247 | 0.0042 | 1.192e-01 | 1.894e-01 |
| 42 | True | 0.9308 | 0.9286 | 0.0023 | 1.185e-01 | 1.853e-01 |
| 456 | True | 0.9142 | 0.9200 | 0.0058 | 1.325e-01 | 1.929e-01 |
| 789 | True | 0.9393 | 0.9344 | 0.0049 | 1.192e-01 | 2.078e-01 |

Mean AUROC 0.9322 ± 0.0112 (paper 0.9318 over 5 seeds); mean ID variance 1.239e-01.

## E5_vit (subsample 3,000/1,000/1,000; 25 original masks)

| config | seeds | maha_unmasked | knn10_unmasked | mahAHM_asym | mahAHM_sym | mahAHM_top5_sym | knnAHM10_asym | knnAHM10_sym | knnAHM10_top5_sym | mahAvgAvg_avg_pad_unmasked | mahAvgAvg_AHM_avg_pad_asym | mahAvgAvg_AHM_avg_pad_sym | mahAvgAvg_AHM_avg_pad_top5_sym | ahss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c100_dtd | 5 | 0.9978 ± 0.0008 | 0.9854 ± 0.0041 | 0.6505 ± 0.1403 | 0.9996 ± 0.0004 | 0.9992 ± 0.0005 | 0.6921 ± 0.1175 | 0.8601 ± 0.0462 | 0.8042 ± 0.1082 | 0.9997 ± 0.0001 | 0.9999 ± 0.0001 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.5367 ± 0.0417 |
| c100_svhn | 5 | 0.9384 ± 0.0163 | 0.9043 ± 0.0140 | 0.7145 ± 0.0453 | 0.8089 ± 0.0412 | 0.8674 ± 0.0507 | 0.7222 ± 0.0508 | 0.4707 ± 0.1197 | 0.5075 ± 0.1868 | 0.9487 ± 0.0109 | 0.8888 ± 0.0297 | 0.7955 ± 0.0488 | 0.9029 ± 0.0280 | 0.9025 ± 0.0169 |
| c10_intra | 5 | 0.9899 ± 0.0022 | 0.9847 ± 0.0046 | 0.7337 ± 0.0788 | 0.9854 ± 0.0038 | 0.9891 ± 0.0033 | 0.8328 ± 0.0481 | 0.9625 ± 0.0129 | 0.9841 ± 0.0050 | 0.9811 ± 0.0038 | 0.9816 ± 0.0040 | 0.9652 ± 0.0085 | 0.9766 ± 0.0069 | 0.8714 ± 0.0389 |
| c10_svhn | 5 | 0.9991 ± 0.0002 | 0.9935 ± 0.0014 | 0.8884 ± 0.0446 | 0.9966 ± 0.0007 | 0.9987 ± 0.0003 | 0.9212 ± 0.0225 | 0.9700 ± 0.0053 | 0.9898 ± 0.0023 | 0.9983 ± 0.0004 | 0.9978 ± 0.0007 | 0.9908 ± 0.0023 | 0.9952 ± 0.0015 | 0.9856 ± 0.0023 |

**E5_vit reproduction gate (asym within ±0.03 of the stored original full-data value)**

| config | mahAHM | knnAHM | mahAvgAvg_AHM | outside (seed, method, Δ) |
|---|---|---|---|---|
| c100_dtd | 5/5 (max \|Δ\| 0.020) | 4/5 (max \|Δ\| 0.036) | 5/5 (max \|Δ\| 0.000) | [(1024, 'knnAHM', -0.036)] |
| c100_svhn | 5/5 (max \|Δ\| 0.018) | 4/5 (max \|Δ\| 0.034) | 5/5 (max \|Δ\| 0.030) | [(789, 'knnAHM', 0.034)] |
| c10_intra | 5/5 (max \|Δ\| 0.011) | 5/5 (max \|Δ\| 0.017) | 5/5 (max \|Δ\| 0.004) | — |
| c10_svhn | 5/5 (max \|Δ\| 0.008) | 5/5 (max \|Δ\| 0.015) | 5/5 (max \|Δ\| 0.001) | — |

**E5_vit paired per-seed differences: mean ± std (seeds with difference > 0)**

| config | mahAHM sym − asym | knnAHM sym − asym | mahAvgAvg_AHM sym − asym | mahAHM asym − Mahalanobis | mahAHM sym − Mahalanobis | mahAvgAvg_AHM sym − unmasked pool |
|---|---|---|---|---|---|---|
| c100_dtd | +0.3491 ± 0.1402 (5/5) | +0.1680 ± 0.0724 (5/5) | +0.0000 ± 0.0001 (3/5) | -0.3473 ± 0.1407 (0/5) | +0.0018 ± 0.0007 (5/5) | +0.0002 ± 0.0001 (5/5) |
| c100_svhn | +0.0944 ± 0.0730 (4/5) | -0.2515 ± 0.0945 (0/5) | -0.0933 ± 0.0241 (0/5) | -0.2239 ± 0.0559 (0/5) | -0.1295 ± 0.0367 (0/5) | -0.1532 ± 0.0412 (0/5) |
| c10_intra | +0.2517 ± 0.0813 (5/5) | +0.1297 ± 0.0598 (5/5) | -0.0164 ± 0.0050 (0/5) | -0.2561 ± 0.0805 (0/5) | -0.0045 ± 0.0028 (1/5) | -0.0158 ± 0.0054 (0/5) |
| c10_svhn | +0.1082 ± 0.0446 (5/5) | +0.0488 ± 0.0238 (5/5) | -0.0070 ± 0.0018 (0/5) | -0.1106 ± 0.0445 (0/5) | -0.0025 ± 0.0007 (0/5) | -0.0075 ± 0.0022 (0/5) |

## E6_vit_classcount (subsample 3,000/1,000/1,000; 25 original masks)

| config | seeds | maha_unmasked | knn10_unmasked | mahAHM_asym | mahAHM_sym | mahAHM_top5_sym | knnAHM10_asym | knnAHM10_sym | knnAHM10_top5_sym | mahAvgAvg_avg_pad_unmasked | mahAvgAvg_AHM_avg_pad_asym | mahAvgAvg_AHM_avg_pad_sym | mahAvgAvg_AHM_avg_pad_top5_sym | ahss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c100_c10_svhn | 3 | 0.9968 ± 0.0017 | 0.9944 ± 0.0002 | 0.8212 ± 0.0443 | 0.9792 ± 0.0081 | 0.9935 ± 0.0005 | 0.8828 ± 0.0335 | 0.9770 ± 0.0035 | 0.9892 ± 0.0041 | 0.9952 ± 0.0009 | 0.9834 ± 0.0090 | 0.9629 ± 0.0087 | 0.9840 ± 0.0015 | 0.9803 ± 0.0059 |
| c100_c20_svhn | 3 | 0.9841 ± 0.0062 | 0.9761 ± 0.0128 | 0.8190 ± 0.0536 | 0.9398 ± 0.0111 | 0.9779 ± 0.0044 | 0.8838 ± 0.0444 | 0.9183 ± 0.0237 | 0.9633 ± 0.0149 | 0.9821 ± 0.0077 | 0.9679 ± 0.0016 | 0.8982 ± 0.0252 | 0.9581 ± 0.0143 | 0.9779 ± 0.0067 |
| c100_c50_svhn | 3 | 0.9447 ± 0.0160 | 0.9190 ± 0.0276 | 0.7766 ± 0.0393 | 0.7996 ± 0.0657 | 0.9054 ± 0.0358 | 0.7943 ± 0.0403 | 0.7203 ± 0.0678 | 0.8062 ± 0.0542 | 0.9597 ± 0.0004 | 0.8856 ± 0.0324 | 0.8051 ± 0.0212 | 0.9119 ± 0.0116 | 0.9382 ± 0.0143 |

**E6_vit_classcount reproduction gate (asym within ±0.03 of the stored original full-data value)**

| config | mahAHM | knnAHM | mahAvgAvg_AHM | outside (seed, method, Δ) |
|---|---|---|---|---|
| c100_c10_svhn | 3/3 (max \|Δ\| 0.000) | 3/3 (max \|Δ\| 0.002) | 3/3 (max \|Δ\| 0.001) | — |
| c100_c20_svhn | 3/3 (max \|Δ\| 0.005) | 3/3 (max \|Δ\| 0.010) | 3/3 (max \|Δ\| 0.006) | — |
| c100_c50_svhn | 3/3 (max \|Δ\| 0.009) | 3/3 (max \|Δ\| 0.017) | 3/3 (max \|Δ\| 0.026) | — |

**E6_vit_classcount paired per-seed differences: mean ± std (seeds with difference > 0)**

| config | mahAHM sym − asym | knnAHM sym − asym | mahAvgAvg_AHM sym − asym | mahAHM asym − Mahalanobis | mahAHM sym − Mahalanobis | mahAvgAvg_AHM sym − unmasked pool |
|---|---|---|---|---|---|---|
| c100_c10_svhn | +0.1580 ± 0.0411 (3/3) | +0.0942 ± 0.0305 (3/3) | -0.0205 ± 0.0007 (0/3) | -0.1756 ± 0.0433 (0/3) | -0.0176 ± 0.0064 (0/3) | -0.0324 ± 0.0079 (0/3) |
| c100_c20_svhn | +0.1208 ± 0.0434 (3/3) | +0.0345 ± 0.0368 (2/3) | -0.0696 ± 0.0238 (0/3) | -0.1651 ± 0.0510 (0/3) | -0.0443 ± 0.0077 (0/3) | -0.0839 ± 0.0175 (0/3) |
| c100_c50_svhn | +0.0230 ± 0.0689 (2/3) | -0.0740 ± 0.0285 (0/3) | -0.0806 ± 0.0333 (0/3) | -0.1681 ± 0.0407 (0/3) | -0.1451 ± 0.0498 (0/3) | -0.1546 ± 0.0211 (0/3) |

## E6 class-count analysis (descriptive; 100 classes = E5 C100/SVHN, same seeds 42/123/456)

| classes | n | masking cost, code's fit (asym) | masking cost, described fit (sym) | pooling cost (pool − Maha) | mahAHM sym − Maha | original-run gap X-Maha − mahAvgAvg_AHM |
|---|---|---|---|---|---|---|
| 10 | 3 | -0.0118 ± 0.0082 | -0.0324 ± 0.0079 | -0.0015 ± 0.0008 | -0.0176 ± 0.0064 | +0.0150 ± 0.0084 |
| 20 | 3 | -0.0142 ± 0.0064 | -0.0839 ± 0.0175 | -0.0020 ± 0.0017 | -0.0443 ± 0.0077 | +0.0257 ± 0.0014 |
| 50 | 3 | -0.0741 ± 0.0327 | -0.1546 ± 0.0211 | +0.0150 ± 0.0161 | -0.1451 ± 0.0498 | +0.0947 ± 0.0286 |
| 100 | 3 | -0.0714 ± 0.0160 | -0.1764 ± 0.0206 | +0.0044 ± 0.0098 | -0.1498 ± 0.0214 | +0.1063 ± 0.0020 |

Spearman ρ (class count vs mean masking cost): asym -0.800, sym -1.000 (negative = masking costs more with more classes). Descriptive only (4 points).
