"""verify_blas_warnings.py — are the 'divide by zero / overflow / invalid value encountered in matmul'
RuntimeWarnings printed during the re-runs harmless?

They are raised by plain matrix products (scipy.linalg.pinvh inside sklearn's ShrunkCovariance, and
`diff @ precision` in the original MahalanobisScorer). A matrix product never divides, and numpy here
uses the platform's BLAS library. This script checks it with numbers, not assumptions:
  1. does the warning appear on a product of plain random numbers (no inf/nan anywhere)?
  2. for the 5 saved Memo leave-class-out embedding sets, recompute the Mahalanobis scores WITHOUT
     BLAS (numpy einsum, optimize=False, own loops) and compare with the original scorer:
     scores, precision matrix (P @ Cov = I), and the stored AUROCs must agree;
  3. scan every result JSON for NaN / inf.
Writes results/_audit/blas_warning_check.json.
"""
import glob, json, math, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402  (puts the original project on sys.path)
import numpy as np  # noqa: E402
from ood_methods.distance_based import MahalanobisScorer  # noqa: E402
from evaluation.metrics import compute_auroc  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
out = {}


def count_warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fn()
    return res, sum("matmul" in str(x.message) for x in w)


# 1. benign product
rng = np.random.default_rng(0)
A, B = rng.standard_normal((2000, 768)), rng.standard_normal((768, 768))
prod, nw = count_warnings(lambda: A @ B)
ref = np.einsum("ni,ij->nj", A, B, optimize=False)
out["1_random_product"] = {"matmul_warnings": nw, "all_finite": bool(np.isfinite(prod).all()),
                           "max_abs_diff_vs_einsum": float(np.abs(prod - ref).max())}

# 2. real embeddings, independent (non-BLAS) Mahalanobis
def maha_independent(tr, y, x):
    classes = np.unique(y)
    means = {c: tr[y == c].mean(0) for c in classes}
    cen = np.empty_like(tr)
    for c in classes:
        cen[y == c] = tr[y == c] - means[c]
    xc = cen - cen.mean(0)  # ShrunkCovariance(assume_centered=False) re-centres
    emp = np.einsum("ni,nj->ij", xc, xc, optimize=False) / len(xc)
    mu = np.trace(emp) / emp.shape[0]
    cov = (1 - 0.01) * emp + 0.01 * mu * np.eye(emp.shape[0])
    d = np.stack([np.einsum("ni,ni->n", np.einsum("ni,ij->nj", x - means[c], PREC, optimize=False), x - means[c],
                            optimize=False) for c in classes], 1)
    return -d.min(1), cov


rows = []
for p in sorted(glob.glob(os.path.join(R, "E1_doc_nearood", "embeddings", "leaveout_*_seed*.npz"))):
    z = np.load(p)
    tr, y = z["tr_cls"].astype(np.float64), z["tr_y"]
    ev, ood = z["ev_cls"].astype(np.float64), z["ood_cls"].astype(np.float64)
    m = MahalanobisScorer()
    _, w_fit = count_warnings(lambda: m.fit(tr, y))
    (s_ev, s_ood), w_score = count_warnings(lambda: (m.score(ev), m.score(ood)))
    PREC = m.precision_matrix
    i_ev, cov = maha_independent(tr, y, ev)
    i_ood, _ = maha_independent(tr, y, ood)
    eye_err = float(np.abs(np.einsum("ij,jk->ik", PREC, cov, optimize=False) - np.eye(len(cov))).max())
    cov_err = float(np.abs(cov - m.shared_cov).max() / np.abs(cov).max())
    rel = float(max(np.abs(s_ev - i_ev).max(), np.abs(s_ood - i_ood).max()) / np.abs(np.r_[i_ev, i_ood]).max())
    stored = json.load(open(p.replace("embeddings", "lco").replace(".npz", ".json")))["all_id_vs_ood"]["mahalanobis"]["auroc"]
    rows.append({"file": os.path.basename(p), "warnings_fit": w_fit, "warnings_score": w_score,
                 "scores_finite": bool(np.isfinite(np.r_[s_ev, s_ood]).all()),
                 "cov_rel_err_vs_independent": cov_err, "max|P@Cov - I|": eye_err,
                 "score_rel_err_vs_independent": rel,
                 "auroc_original": compute_auroc(s_ev, s_ood), "auroc_independent": compute_auroc(i_ev, i_ood),
                 "auroc_stored": stored})
out["2_real_embeddings"] = rows

# 3. NaN / inf anywhere in any result JSON
bad = []
def walk(o, path, f):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}", f)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", f)
    elif isinstance(o, float) and not math.isfinite(o):
        bad.append(f"{f}:{path}")
files = [f for f in glob.glob(os.path.join(R, "**", "*.json"), recursive=True) if "_audit" not in f]
for f in files:
    walk(json.load(open(f)), "", os.path.relpath(f, R))
out["3_nonfinite_scan"] = {"files_scanned": len(files), "nonfinite_values": bad}

ok = (out["1_random_product"]["all_finite"]
      and all(r["scores_finite"] and r["score_rel_err_vs_independent"] < 1e-9 and r["max|P@Cov - I|"] < 1e-8
              and r["auroc_original"] == r["auroc_independent"] == r["auroc_stored"] for r in rows)
      and not bad)
out["VERDICT_WARNINGS_HARMLESS"] = bool(ok)
os.makedirs(os.path.join(R, "_audit"), exist_ok=True)
C.save_json(out, os.path.join(R, "_audit", "blas_warning_check.json"))
print(json.dumps(out, indent=1))
