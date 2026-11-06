"""
masked_sym.py — the "asym vs sym" fit comparison shared by E2 (RoBERTa-base),
E3 (RoBERTa-large) and E5/E6 (ViT). See PROTOCOL.md §E2, §E3, §E5.

For one model and one seed:
  * unmasked features of train_sub / eval_sub / each OOD_sub
  * features under each of the 25 masks of the ORIGINAL cross-architecture generator,
    averaged over masks (running mean), plus the AHSS displacement accumulated per sample
  * scores:
      asym : fit on UNMASKED train_sub, score MASK-AVERAGED eval/OOD   (original code)
      sym  : fit on MASK-AVERAGED train_sub, score MASK-AVERAGED eval/OOD (corrected)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
import common as C
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr


def ev(id_s, ood_s):
    return {"auroc": float(compute_auroc(id_s, ood_s)), "fpr95": float(compute_fpr_at_tpr(id_s, ood_s))}


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def masked_block(model, parts, input_keys, masks, feats, batch_size, top_f=5, sel_k=10):
    """parts: ordered dict name -> Dataset (must contain 'train' and 'eval').
    Returns (unmasked, masked_mean_all, ahss_disp, labels, masked_mean_topF, selection) where the
    feature containers are dicts part -> {feat: array}.
      * masked_mean_all : mean over ALL masks (the original cross-architecture definition)
      * masked_mean_topF: mean over the top-F masks ranked by the ORIGINAL AHM criterion
                          (models/attention_masking.AHMOptimizer._compute_similarity_score:
                          mean top-K cosine similarity of eval to train CLS under the mask)
    AHSS displacement uses L2-normalised last-layer CLS, float64, as AHSSScorer.compute_ahss."""
    from models.attention_masking import AHMOptimizer
    names = list(parts)
    sizes = [len(parts[n]) for n in names]
    off = dict(zip(names, np.concatenate([[0], np.cumsum(sizes)[:-1]])))
    sl = lambda n: slice(off[n], off[n] + len(parts[n]))
    cat = ConcatDataset([parts[n] for n in names])
    dl = DataLoader(cat, batch_size=batch_size, shuffle=False)
    split = lambda arr: dict(zip(names, np.split(arr, np.cumsum(sizes)[:-1])))
    masker = C.HeadMasker(model)
    sel = AHMOptimizer.__new__(AHMOptimizer); sel.neighbors_K = sel_k     # original selection code, unchanged
    t0 = time.time()
    un = C.run_features(model, dl, input_keys, want=feats)
    un_cls_n = l2n(un["cls_last"])
    msum = {f: np.zeros_like(un[f]) for f in feats}
    per_mask = {f: [] for f in feats}
    sel_scores = []
    disp = np.zeros(un["cls_last"].shape[0])
    for i, mk in enumerate(masks):
        m = C.run_features(model, dl, input_keys, want=feats, masker=masker, mask=mk)
        for f in feats:
            msum[f] += m[f]
            per_mask[f].append(m[f].astype(np.float32))
        disp += np.sum((l2n(m["cls_last"]) - un_cls_n) ** 2, axis=1)
        sel_scores.append(float(sel._compute_similarity_score(m["cls_last"][sl("train")], m["cls_last"][sl("eval")])))
        if i == 0:
            C.log(f"    mask 1/{len(masks)} done in {time.time() - t0:.0f}s (incl. unmasked pass)")
    mmean = {f: msum[f] / len(masks) for f in feats}
    disp /= len(masks)
    top = [int(j) for j in np.argsort(-np.asarray(sel_scores), kind="stable")[:top_f]]
    mtop = {f: np.mean(np.stack([per_mask[f][j].astype(np.float64) for j in top]), axis=0) for f in feats}
    del per_mask

    def by(d):                                   # feat -> array  ==>  part -> feat -> array
        s_ = {f: split(d[f]) for f in feats}
        return {n: {f: s_[f][n] for f in feats} for n in names}

    labels = split(un["labels"]) if "labels" in un else None
    selection = {"criterion": "AHMOptimizer._compute_similarity_score (mean top-K cos, eval->train, CLS)",
                 "K": sel_k, "F": top_f, "scores": sel_scores, "top_mask_indices": top}
    return by(un), by(mmean), split(disp), labels, by(mtop), selection


def score_block(un, mm, disp, train_y, ood_names, avg_feats=("avg_pad", "avg_nopad"), knn_k=10, extra_id=None, mtop=None):
    """All asym/sym/unmasked scores for each OOD set. un/mm/mtop: part -> feat -> array.
    If mtop (top-F AHM-selected masks) is given, the *_top5_sym / *_top5_asym variants are added."""
    out = {}
    tr_u, tr_m, ev_u, ev_m = un["train"], mm["train"], un["eval"], mm["eval"]

    def maha(fit, a, b):
        s = MahalanobisScorer(); s.fit(fit, train_y); return s.score(a), s.score(b)

    def knn(fit, a, b):
        s = KNNScorer(k=knn_k, metric="euclidean"); s.fit(fit); return s.score(a), s.score(b)

    for o in ood_names:
        o_u, o_m = un[o], mm[o]
        r = {}
        r["maha_unmasked"] = ev(*maha(tr_u["cls_last"], ev_u["cls_last"], o_u["cls_last"]))
        r[f"knn{knn_k}_unmasked"] = ev(*knn(tr_u["cls_last"], ev_u["cls_last"], o_u["cls_last"]))
        r["mahAHM_asym"] = ev(*maha(tr_u["cls_last"], ev_m["cls_last"], o_m["cls_last"]))
        r["mahAHM_sym"] = ev(*maha(tr_m["cls_last"], ev_m["cls_last"], o_m["cls_last"]))
        r[f"knnAHM{knn_k}_asym"] = ev(*knn(tr_u["cls_last"], ev_m["cls_last"], o_m["cls_last"]))
        r[f"knnAHM{knn_k}_sym"] = ev(*knn(tr_m["cls_last"], ev_m["cls_last"], o_m["cls_last"]))
        for f in avg_feats:
            if f not in tr_u:
                continue
            r[f"mahAvgAvg_{f}_unmasked"] = ev(*maha(tr_u[f], ev_u[f], o_u[f]))
            r[f"mahAvgAvg_AHM_{f}_asym"] = ev(*maha(tr_u[f], ev_m[f], o_m[f]))
            r[f"mahAvgAvg_AHM_{f}_sym"] = ev(*maha(tr_m[f], ev_m[f], o_m[f]))
        r["ahss"] = ev(-disp["eval"], -disp[o])
        if mtop is not None:          # Algorithm-1-style: top-F masks by the original AHM criterion
            tt, et, ot = mtop["train"], mtop["eval"], mtop[o]
            r["mahAHM_top5_sym"] = ev(*maha(tt["cls_last"], et["cls_last"], ot["cls_last"]))
            r["mahAHM_top5_asym"] = ev(*maha(tr_u["cls_last"], et["cls_last"], ot["cls_last"]))
            r[f"knnAHM{knn_k}_top5_sym"] = ev(*knn(tt["cls_last"], et["cls_last"], ot["cls_last"]))
            r[f"knnAHM{knn_k}_top5_asym"] = ev(*knn(tr_u["cls_last"], et["cls_last"], ot["cls_last"]))
            for f in avg_feats:
                if f not in tt:
                    continue
                r[f"mahAvgAvg_AHM_{f}_top5_sym"] = ev(*maha(tt[f], et[f], ot[f]))
                r[f"mahAvgAvg_AHM_{f}_top5_asym"] = ev(*maha(tr_u[f], et[f], ot[f]))
        if extra_id is not None:
            r.update(extra_id(o))
        out[o] = r
    return out
