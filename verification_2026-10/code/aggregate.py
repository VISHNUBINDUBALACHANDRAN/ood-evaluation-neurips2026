"""
aggregate.py — builds summary/SUMMARY.md and summary/summary.json from the per-seed
results, re-checking every gate. Pure post-processing (CPU only); safe to re-run anytime.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
import common as C

R = C.RESULTS
SUMD = os.path.join(C.FINAL, "summary")
FINAL_RUN = "--final" in sys.argv       # only the queue's last step passes --final; interim runs never touch SUMMARY.md
OUT_MD, OUT_JS = ("SUMMARY.md", "summary.json") if FINAL_RUN else ("INTERIM_SUMMARY.md", "interim_summary.json")
os.makedirs(SUMD, exist_ok=True)
J = lambda p: json.load(open(p))
md, js = [], {}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=0):.4f}" if len(v) else "n/a"


def wil1(a, b):
    """one-sided exact Wilcoxon a > b (paired); None if all differences are zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return None
    return float(stats.wilcoxon(a, b, alternative="greater", method="exact" if len(a) <= 25 else "auto").pvalue)


md.append("# Verification — results summary\n")
md.append(f"Generated {__import__('datetime').datetime.now(__import__('datetime').timezone.utc):%Y-%m-%d %H:%M} UTC — "
          + ("**FINAL** (all queue steps finished)\n" if FINAL_RUN else "**INTERIM** (queue still running; numbers may change)\n"))
md.append(f"Environment: `{json.dumps(C.env_info())}`\n")

# ------------------------------------------------------------------ gates
g = J(os.path.join(R, "gate0_parity.json"))
md.append("## Gate 0 — re-run vs original parity\n")
for k, v in g["checks"].items():
    c = v.get("cls") or v.get("cls_layers")
    md.append(f"- **{k}**: PASS={v['PASS']}; min cosine {c['cos_min']:.9f}")
js["gate0_all_pass"] = g.get("ALL_PASS")

# ------------------------------------------------------------------ E1
p = os.path.join(R, "E1_doc_nearood", "E1R_repro_submitted_table4.json")
if os.path.exists(p):
    e = J(p)
    md.append("\n## E1-R — original Table 4 construction (model trained on the held-out class) reproduced in the re-run\n")
    md.append("| seed | Maha (re-run) | Maha (original) | kNN k=50 (re-run) | kNN (original) | gate |\n|---|---|---|---|---|---|")
    for s, r in e["seeds"].items():
        md.append(f"| {s} | {r['mahalanobis']['auroc']:.4f} | {r['stored_original']['mahalanobis']:.4f} | "
                  f"{r['knn_k50']['auroc']:.4f} | {r['stored_original']['knn']:.4f} | {r['GATE_PASS']} |")
    m = [r['mahalanobis']['auroc'] for r in e["seeds"].values()]; k = [r['knn_k50']['auroc'] for r in e["seeds"].values()]
    md.append(f"\nMean: Maha {ms(m)}, kNN {ms(k)} (paper Table 4: 0.555, 0.634). ALL_GATES_PASS={e.get('ALL_GATES_PASS')}")
    js["E1R"] = {"maha": np.mean(m), "knn": np.mean(k), "all_pass": e.get("ALL_GATES_PASS")}

files = sorted(glob.glob(os.path.join(R, "E1_doc_nearood", "lco", "*.json")))
if files:
    md.append("\n## E1-C — corrected near-OOD with leave-class-out models (OOD class never trained on)\n")
    md.append("Std.-OOD reference (submitted Table 1, Tob-Intra, Advertisement): Maha 0.970, kNN 0.961.\n")
    md.append("| held-out class | seeds | split gate | Maha | kNN k=50 | kNN k=10 | MSP | Energy | Letter-only Maha | Letter-only kNN50 |\n|---|---|---|---|---|---|---|---|---|---|")
    js["E1C"] = {}
    for h in ["Memo", "Form", "Resume", "Scientific"]:
        rs = [J(f) for f in files if f"/leaveout_{h}_" in f]
        if not rs:
            continue
        A = lambda key, sub="all_id_vs_ood": [r[sub][key]["auroc"] for r in rs]
        gates = sum(r["SPLIT_GATE_PASS"] for r in rs)
        md.append(f"| {h} | {len(rs)} | {gates}/{len(rs)} | {ms(A('mahalanobis'))} | {ms(A('knn_k50'))} | {ms(A('knn_k10'))} | "
                  f"{ms(A('msp'))} | {ms(A('energy'))} | {ms(A('mahalanobis','letter_vs_ood'))} | {ms(A('knn_k50','letter_vs_ood'))} |")
        js["E1C"][h] = {k: float(np.mean(A(k))) for k in ("mahalanobis", "knn_k50", "knn_k10", "msp", "energy")}
        js["E1C"][h]["split_gates"] = f"{gates}/{len(rs)}"
        js["E1C"][h]["letter_maha"] = float(np.mean(A("mahalanobis", "letter_vs_ood")))
        js["E1C"][h]["letter_knn50"] = float(np.mean(A("knn_k50", "letter_vs_ood")))

files = sorted(glob.glob(os.path.join(R, "E1_doc_nearood", "ahss", "*.json")))
if files:
    rs = [J(f) for f in files]
    md.append(f"\n**E1-A AHSS (leave-Memo-out, 25 per-layer random masks, {len(rs)} seeds):** all-ID {ms([r['ahss_all_id_vs_ood']['auroc'] for r in rs])}; "
              f"Letter-only {ms([r['ahss_letter_vs_ood']['auroc'] for r in rs])} (submitted Table 4: 0.069).")
    js["E1A_ahss"] = float(np.mean([r['ahss_all_id_vs_ood']['auroc'] for r in rs]))

# ------------------------------------------------------------------ E2 / E3
for size, tag in (("base", "E2_roberta_base"), ("large", "E3_roberta_large")):
    files = sorted(glob.glob(os.path.join(R, tag, "seed*.json")))
    if not files:
        continue
    rs = [J(f) for f in files]
    md.append(f"\n## {tag} — {len(rs)} seeds\n")
    js[tag] = {}
    if size == "base" and all("full" in r for r in rs):
        md.append("**Full data, unmasked (reproduction gate: Maha / kNN / X-Maha-uniform within ±0.002 of the original run)**\n")
        md.append("| OOD | gate | Maha | kNN10 | X-Maha-unif | mahAvgAvg pad-incl | mahAvgAvg pad-excl | LENGTH ONLY | mean tokens ID/OOD |\n|---|---|---|---|---|---|---|---|---|")
        for o in rs[0]["full"]:
            F = [r["full"][o] for r in rs]
            gate = sum(1 for x in F if x.get("REPRO_GATE_PASS")) if "REPRO_GATE_PASS" in F[0] else "–"
            g_s = f"{gate}/{len(F)}" if gate != "–" else "–"
            md.append(f"| {o} | {g_s} | {ms([x['Mahalanobis']['auroc'] for x in F])} | {ms([x['kNN10']['auroc'] for x in F])} | "
                      f"{ms([x['X-Maha_uniform']['auroc'] for x in F])} | {ms([x['mahAvgAvg_pad_unmasked']['auroc'] for x in F])} | "
                      f"{ms([x['mahAvgAvg_nopad_unmasked']['auroc'] for x in F])} | {ms([x['length_only']['auroc'] for x in F])} | "
                      f"{np.mean([x['mean_tokens']['id'] for x in F]):.1f} / {np.mean([x['mean_tokens']['ood'] for x in F]):.1f} |")
            js[tag][f"full_{o}"] = {k: float(np.mean([x[k]['auroc'] for x in F])) for k in
                                    ("Mahalanobis", "kNN10", "X-Maha_uniform", "mahAvgAvg_pad_unmasked", "mahAvgAvg_nopad_unmasked", "length_only")}
        md.append("\nFull-data paired differences vs X-Maha-uniform, PROTOCOL §E2 question (a) (descriptive; seeds with difference > 0):\n")
        for o in rs[0]["full"]:
            F = [r["full"][o] for r in rs]
            parts_ = []
            for k, lab in (("mahAvgAvg_pad_unmasked", "pool WITH padding"), ("mahAvgAvg_nopad_unmasked", "pool WITHOUT padding")):
                d = [x[k]["auroc"] - x["X-Maha_uniform"]["auroc"] for x in F]
                parts_.append(f"{lab} − X-Maha {np.mean(d):+.4f} ± {np.std(d):.4f} ({sum(v > 0 for v in d)}/{len(d)})")
            md.append(f"- {o}: " + "; ".join(parts_))
    md.append("\n**Subsample (3,000 train / 1,000 eval / 1,000 OOD), 25 original masks: asym (original) vs sym (corrected)**\n")
    cols = ["maha_unmasked", "mahAHM_asym", "mahAHM_sym", "mahAHM_top5_sym", "knnAHM10_asym", "knnAHM10_sym", "knnAHM10_top5_sym",
            "mahAvgAvg_avg_pad_unmasked", "mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_AHM_avg_pad_sym", "mahAvgAvg_AHM_avg_pad_top5_sym",
            "mahAvgAvg_avg_nopad_unmasked", "mahAvgAvg_AHM_avg_nopad_asym", "mahAvgAvg_AHM_avg_nopad_sym",
            "mahAvgAvg_AHM_avg_nopad_top5_sym", "ahss"]
    md.append("| OOD | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    oods = list(rs[0]["subsample"]["scores"])
    pooled = {c: [] for c in cols}
    for o in oods:
        S = [r["subsample"]["scores"][o] for r in rs]
        row = []
        for c in cols:
            v = [x[c]["auroc"] for x in S if c in x]
            row.append(ms(v))
            if o in ("20news", "agnews", "mnli"):
                pooled[c] += v
        md.append(f"| {o} | " + " | ".join(row) + " |")
        js[tag][f"sub_{o}"] = {c: float(np.mean([x[c]['auroc'] for x in S if c in x])) for c in cols if c in S[0]}
        st = [x.get("stored_original_full", {}) for x in S]
        if st and st[0]:
            md.append(f"|  ↳ original full-data (original asym) | " + " | ".join(
                [ms([y.get('Mahalanobis') for y in st]) if c == 'maha_unmasked' else
                 ms([y.get('mahAHM') for y in st]) if c == 'mahAHM_asym' else
                 ms([y.get('knnAHM') for y in st]) if c == 'knnAHM10_asym' else
                 ms([y.get('mahAvgAvg_AHM') for y in st]) if (c == 'mahAvgAvg_AHM_avg_pad_asym' and 'mahAvgAvg_AHM' in st[0]) else
                 ms([y.get('AHSS') for y in st]) if c == 'ahss' else "" for c in cols]) + " |")
    md.append("\nPooled over 20news/agnews/mnli × seeds: " + "; ".join(f"{c} {np.mean(v):.4f}" for c, v in pooled.items() if v))
    js[tag]["pooled"] = {c: float(np.mean(v)) for c, v in pooled.items() if v}
    # reproduction gates (PROTOCOL §E2 / §E3): original asym definition vs stored original full-data values
    if size == "base":
        md.append("\n**Masked-part reproduction gate (PROTOCOL §E2: asym within ±0.03 of the stored original full-data value)**\n")
        gres = {}
        for mine, theirs in (("mahAHM_asym", "mahAHM"), ("knnAHM10_asym", "knnAHM"), ("mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_AHM")):
            cells = [(r["seed"], o, r["subsample"]["scores"][o][mine]["auroc"], r["subsample"]["scores"][o]["stored_original_full"][theirs])
                     for r in rs for o in ("20news", "agnews", "mnli")
                     if theirs in r["subsample"]["scores"][o].get("stored_original_full", {})]
            fails = [(s_, o, round(a - b, 4)) for s_, o, a, b in cells if abs(a - b) > 0.03]
            gres[mine] = {"pass": len(cells) - len(fails), "n": len(cells),
                          "max_abs_diff": float(max(abs(a - b) for *_, a, b in cells)), "outside": fails}
            md.append(f"- {mine} vs original {theirs}: {len(cells) - len(fails)}/{len(cells)} within ±0.03 "
                      f"(max |Δ| {gres[mine]['max_abs_diff']:.4f})" + (f"; outside (seed, OOD, Δ): {fails}" if fails else ""))
        js[tag]["masked_repro_gate"] = gres
    elif pooled["mahAHM_asym"]:
        m = float(np.mean(pooled["mahAHM_asym"]))
        ok = abs(m - 0.3448) <= 0.05
        md.append(f"\n**Reproduction gate (PROTOCOL §E3): pooled mahAHM_asym {m:.4f} vs reported 0.3448 (±0.05) → "
                  f"{'PASS' if ok else 'FAIL'}**")
        js[tag]["repro_gate"] = {"pooled_mahAHM_asym": m, "reported": 0.3448, "PASS": bool(ok)}
    if size == "large" and pooled["mahAHM_sym"]:
        # the only E2/E3 test with a pre-stated direction (PROTOCOL §E3); E2 has none (PROTOCOL §0.6)
        p_m = wil1(pooled["mahAHM_sym"], pooled["mahAHM_asym"]); p_k = wil1(pooled["knnAHM10_sym"], pooled["knnAHM10_asym"])
        md.append(f"\nPre-stated test (sym > asym, one-sided exact Wilcoxon, n={len(pooled['mahAHM_sym'])}): mahAHM p={p_m}, knnAHM p={p_k}")
        if pooled.get("mahAHM_top5_sym"):
            p_t = wil1(pooled["mahAHM_top5_sym"], pooled["mahAHM_asym"])
            md.append(f"Same test for the top-5 (Algorithm-1-style) variant: mahAHM_top5_sym > mahAHM_asym p={p_t}")
        js[tag]["wilcoxon_sym_gt_asym"] = {"mahAHM": p_m, "knnAHM": p_k}
    # paired per-seed differences on the same subsample: descriptive only (no test without a pre-stated direction)
    PRE, EXP = "pre-registered comparison, descriptive", "EXPLORATORY (post hoc, not pre-registered)"
    contrasts = [("mahAHM_sym", "mahAHM_asym", "sym − asym, mahAHM", PRE),
                 ("knnAHM10_sym", "knnAHM10_asym", "sym − asym, knnAHM", PRE),
                 ("mahAHM_asym", "maha_unmasked", "masking gain, mahAHM asym", PRE),
                 ("mahAHM_sym", "maha_unmasked", "masking gain, mahAHM sym", PRE),
                 ("mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_avg_pad_unmasked", "masking gain, pool WITH padding, asym", PRE),
                 ("mahAvgAvg_AHM_avg_nopad_asym", "mahAvgAvg_avg_nopad_unmasked", "masking gain, pool WITHOUT padding, asym", PRE),
                 ("mahAvgAvg_AHM_avg_nopad_sym", "mahAvgAvg_avg_nopad_unmasked", "masking gain, pool WITHOUT padding, sym", PRE),
                 ("mahAvgAvg_avg_nopad_unmasked", "mahAvgAvg_AHM_avg_pad_asym", "no-padding pool (no masks) − paper's mahAvgAvg_AHM", EXP)]
    md.append("\n**Paired per-seed differences, same subsample: mean ± std (seeds with difference > 0)**\n")
    md.append("| contrast | status | " + " | ".join(oods) + " |\n|---|---|" + "---|" * len(oods))
    js[tag]["paired_differences"] = {}
    for a, b, lab, status in contrasts:
        cells, jj = [], {}
        for o in oods:
            d = [r["subsample"]["scores"][o][a]["auroc"] - r["subsample"]["scores"][o][b]["auroc"] for r in rs
                 if a in r["subsample"]["scores"][o] and b in r["subsample"]["scores"][o]]
            cells.append(f"{np.mean(d):+.4f} ± {np.std(d):.4f} ({sum(x > 0 for x in d)}/{len(d)})" if d else "n/a")
            jj[o] = d
        md.append(f"| {lab} | {status} | " + " | ".join(cells) + " |")
        js[tag]["paired_differences"][lab] = {"status": status, "per_ood": jj}

# ------------------------------------------------------------------ E4
files = sorted(glob.glob(os.path.join(R, "E4_mcdropout_layoutlmv3", "seed*.json")))
if files:
    rs = [J(f) for f in files]
    md.append("\n## E4 — LayoutLMv3 MC-Dropout, fine-tuned weights loaded strictly (tobacco_intra)\n")
    md.append("| seed | weights gate | AUROC (re-run) | AUROC (paper) | |Δ| | var ID | var OOD |\n|---|---|---|---|---|---|---|")
    for r in rs:
        md.append(f"| {r['seed']} | {r['WEIGHTS_GATE_PASS']} | {r['auroc']:.4f} | {r['paper_auroc_this_seed']:.4f} | "
                  f"{r['abs_diff_vs_paper']:.4f} | {r['var_id_mean']:.3e} | {r['var_ood_mean']:.3e} |")
    md.append(f"\nMean AUROC {ms([r['auroc'] for r in rs])} (paper 0.9318 over 5 seeds); mean ID variance "
              f"{np.mean([r['var_id_mean'] for r in rs]):.3e}.")
    js["E4"] = {"auroc": float(np.mean([r['auroc'] for r in rs])), "var_id": float(np.mean([r['var_id_mean'] for r in rs])),
                "var_ood": float(np.mean([r['var_ood_mean'] for r in rs])), "n": len(rs)}

# ------------------------------------------------------------------ E5 / E6
for tag in ("E5_vit", "E6_vit_classcount"):
    cfgs = sorted(glob.glob(os.path.join(R, tag, "*")))
    if not cfgs:
        continue
    md.append(f"\n## {tag} (subsample 3,000/1,000/1,000; 25 original masks)\n")
    cols = ["maha_unmasked", "knn10_unmasked", "mahAHM_asym", "mahAHM_sym", "mahAHM_top5_sym", "knnAHM10_asym", "knnAHM10_sym",
            "knnAHM10_top5_sym", "mahAvgAvg_avg_pad_unmasked", "mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_AHM_avg_pad_sym",
            "mahAvgAvg_AHM_avg_pad_top5_sym", "ahss"]
    md.append("| config | seeds | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    js[tag] = {}
    for cd in cfgs:
        rs = [J(f) for f in sorted(glob.glob(os.path.join(cd, "seed*.json")))]
        if not rs:
            continue
        md.append(f"| {os.path.basename(cd)} | {len(rs)} | " + " | ".join(ms([r['scores'][c]['auroc'] for r in rs]) for c in cols) + " |")
        js[tag][os.path.basename(cd)] = {c: float(np.mean([r['scores'][c]['auroc'] for r in rs])) for c in cols}
    # reproduction gate (PROTOCOL §E5; E6 uses the E5 design): asym within ±0.03 of the stored original full-data value.
    # The original files name the keys per config family: CIFAR-10 'extended_results:<m>', CIFAR-100 'full_results:<m>'.
    def orig(st, m):
        for k in (f"full_results:{m}", f"extended_results:{m}", f"results:{m}", m):
            if k in st:
                return st[k]
        return None
    md.append(f"\n**{tag} reproduction gate (asym within ±0.03 of the stored original full-data value)**\n")
    md.append("| config | mahAHM | knnAHM | mahAvgAvg_AHM | outside (seed, method, Δ) |\n|---|---|---|---|---|")
    js[tag]["repro_gate"] = {}
    for cd in cfgs:
        rs = [J(f) for f in sorted(glob.glob(os.path.join(cd, "seed*.json")))]
        if not rs:
            continue
        cells, outside = [], []
        for mine, theirs in (("mahAHM_asym", "mahAHM"), ("knnAHM10_asym", "knnAHM"), ("mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_AHM")):
            d = [(r["seed"], r["scores"][mine]["auroc"] - orig(r["stored_original_full"], theirs)) for r in rs
                 if orig(r["stored_original_full"], theirs) is not None]
            ok = sum(abs(x) <= 0.03 for _, x in d)
            cells.append(f"{ok}/{len(d)} (max \\|Δ\\| {max(abs(x) for _, x in d):.3f})" if d else "n/a")
            outside += [(s_, theirs, round(x, 3)) for s_, x in d if abs(x) > 0.03]
        md.append(f"| {os.path.basename(cd)} | " + " | ".join(cells) + f" | {outside if outside else '—'} |")
        js[tag]["repro_gate"][os.path.basename(cd)] = {"cells": cells, "outside": outside}
    # paired per-seed differences (descriptive; no direction pre-stated for E5/E6)
    md.append(f"\n**{tag} paired per-seed differences: mean ± std (seeds with difference > 0)**\n")
    pc = [("mahAHM_sym", "mahAHM_asym", "mahAHM sym − asym"), ("knnAHM10_sym", "knnAHM10_asym", "knnAHM sym − asym"),
          ("mahAvgAvg_AHM_avg_pad_sym", "mahAvgAvg_AHM_avg_pad_asym", "mahAvgAvg_AHM sym − asym"),
          ("mahAHM_asym", "maha_unmasked", "mahAHM asym − Mahalanobis"), ("mahAHM_sym", "maha_unmasked", "mahAHM sym − Mahalanobis"),
          ("mahAvgAvg_AHM_avg_pad_sym", "mahAvgAvg_avg_pad_unmasked", "mahAvgAvg_AHM sym − unmasked pool")]
    md.append("| config | " + " | ".join(l for *_, l in pc) + " |\n|---|" + "---|" * len(pc))
    js[tag]["paired_differences"] = {}
    for cd in cfgs:
        rs = [J(f) for f in sorted(glob.glob(os.path.join(cd, "seed*.json")))]
        if not rs:
            continue
        row, jj = [], {}
        for a, b, lab in pc:
            d = [r["scores"][a]["auroc"] - r["scores"][b]["auroc"] for r in rs]
            row.append(f"{np.mean(d):+.4f} ± {np.std(d):.4f} ({sum(x > 0 for x in d)}/{len(d)})")
            jj[lab] = d
        md.append(f"| {os.path.basename(cd)} | " + " | ".join(row) + " |")
        js[tag]["paired_differences"][os.path.basename(cd)] = jj

# ------------------------------------------------------------------ E6 class-count analysis (descriptive)
# Claim checked: "label-space size hurts the head-mask-averaged variant specifically" (gap X-Maha − mahAvgAvg_AHM grows
# 10→100 classes). Here the masking effect is isolated on the same subsample: masked pool − unmasked pool (asym and sym);
# 100 classes = E5 c100_svhn with the E6 seeds (42/123/456).
cc_cfg = [(10, os.path.join(R, "E6_vit_classcount", "c100_c10_svhn")), (20, os.path.join(R, "E6_vit_classcount", "c100_c20_svhn")),
          (50, os.path.join(R, "E6_vit_classcount", "c100_c50_svhn")), (100, os.path.join(R, "E5_vit", "c100_svhn"))]
rows_cc = []
for ncls, cd in cc_cfg:
    rs = [J(os.path.join(cd, f"seed{s}.json")) for s in (42, 123, 456) if os.path.exists(os.path.join(cd, f"seed{s}.json"))]
    if not rs:
        continue
    g = lambda k: np.array([r["scores"][k]["auroc"] for r in rs])
    orig = lambda m: np.array([next((r["stored_original_full"][k] for k in (f"full_results:{m}", m) if k in r["stored_original_full"]), np.nan)
                               for r in rs])
    rows_cc.append({"classes": ncls, "n": len(rs),
                    "mask_cost_asym": g("mahAvgAvg_AHM_avg_pad_asym") - g("mahAvgAvg_avg_pad_unmasked"),
                    "mask_cost_sym": g("mahAvgAvg_AHM_avg_pad_sym") - g("mahAvgAvg_avg_pad_unmasked"),
                    "pool_cost": g("mahAvgAvg_avg_pad_unmasked") - g("maha_unmasked"),
                    "mahAHM_sym_minus_maha": g("mahAHM_sym") - g("maha_unmasked"),
                    "original_gap": orig("X-Maha_uniform") - orig("mahAvgAvg_AHM")})
if rows_cc:
    md.append("\n## E6 class-count analysis (descriptive; 100 classes = E5 C100/SVHN, same seeds 42/123/456)\n")
    md.append("| classes | n | masking cost, code's fit (asym) | masking cost, described fit (sym) | pooling cost (pool − Maha) | "
              "mahAHM sym − Maha | original-run gap X-Maha − mahAvgAvg_AHM |\n|---|---|---|---|---|---|---|")
    for x in rows_cc:
        f = lambda v: f"{np.nanmean(v):+.4f} ± {np.nanstd(v):.4f}"
        md.append(f"| {x['classes']} | {x['n']} | {f(x['mask_cost_asym'])} | {f(x['mask_cost_sym'])} | {f(x['pool_cost'])} | "
                  f"{f(x['mahAHM_sym_minus_maha'])} | {f(x['original_gap'])} |")
    if len(rows_cc) >= 3:
        cls = [x["classes"] for x in rows_cc]
        rho_a = stats.spearmanr(cls, [np.mean(x["mask_cost_asym"]) for x in rows_cc]).correlation
        rho_s = stats.spearmanr(cls, [np.mean(x["mask_cost_sym"]) for x in rows_cc]).correlation
        md.append(f"\nSpearman ρ (class count vs mean masking cost): asym {rho_a:+.3f}, sym {rho_s:+.3f} "
                  f"(negative = masking costs more with more classes). Descriptive only (4 points).")
        js["E6_classcount_analysis"] = {"rows": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x.items()} for x in rows_cc],
                                        "spearman_masking_cost_asym": float(rho_a), "spearman_masking_cost_sym": float(rho_s)}

open(os.path.join(SUMD, OUT_MD), "w").write("\n".join(md) + "\n")
C.save_json(js, os.path.join(SUMD, OUT_JS))
print("\n".join(md))
