"""
E1 — Document near-OOD on Tobacco-3482 (see PROTOCOL.md §E1).

  python e1_doc_nearood.py repro            # E1-R: reproduce the original Table 4 construction
  python e1_doc_nearood.py lco [holdouts]   # E1-C: corrected leave-class-out models
  python e1_doc_nearood.py ahss             # E1-A: AHSS on the leave-Memo-out models
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import common as C
from gate0_parity import DocView, doc_split

from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from utils.data import create_dataloader
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from ood_methods.confidence_based import MSPScorer, EnergyScorer
from ood_methods.sensitivity_based import AHSSScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr

KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")
OUTD = os.path.join(C.RESULTS, "E1_doc_nearood")
EMB = os.path.join(OUTD, "embeddings")
HOLDOUTS = ["Memo", "Form", "Resume", "Scientific"]
_DS = None


def ds():
    global _DS
    if _DS is None:
        _DS = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
    return _DS


def feats(model, rows, l2i, want=("cls_last", "logits"), masker=None, mask=None, release_cache=False):
    dl = create_dataloader(DocView(ds(), rows, l2i), batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    return C.run_features(model, dl, KEYS, want, masker=masker, mask=mask, release_cache=release_cache)


def ev(id_s, ood_s):
    return {"auroc": compute_auroc(id_s, ood_s), "fpr95": compute_fpr_at_tpr(id_s, ood_s)}


# --------------------------------------------------------------------- E1-R
def repro():
    labels = list(ds()["labels"])
    letter = [i for i, l in enumerate(labels) if l == "Letter"]
    memo = [i for i, l in enumerate(labels) if l == "Memo"]
    out = {"design": "submitted Table 4: intra model (trained on Letter AND Memo), ID=20% Letter, OOD=all Memo",
           "env": C.env_info(), "seeds": {}}
    for s in C.SEEDS:
        t0 = time.time()
        ck = os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{s}", "best_model")
        model, info = C.build_layoutlmv3(ck)
        assert "Memo" in info["label2id"] and "Letter" in info["label2id"], "intra model must contain Memo+Letter"
        _, ev_idx = train_test_split(np.arange(len(letter)), train_size=0.8, random_state=s)   # original non-stratified branch
        id_rows = [letter[i] for i in ev_idx]
        f_id = feats(model, id_rows, {})
        f_ood = feats(model, memo, {})
        emb = os.path.join(C.OUT, "embeddings", f"tobacco_intra_seed{s}")
        tr, trl = np.load(os.path.join(emb, "train_embeddings.npy")), np.load(os.path.join(emb, "train_labels.npy"))
        mah = MahalanobisScorer(); mah.fit(tr, trl)
        knn = KNNScorer(k=50, metric="euclidean"); knn.fit(tr)
        r = {"n_id": len(id_rows), "n_ood": len(memo),
             "mahalanobis": ev(mah.score(f_id["cls_last"]), mah.score(f_ood["cls_last"])),
             "knn_k50": ev(knn.score(f_id["cls_last"]), knn.score(f_ood["cls_last"]))}
        stored = json.load(open(os.path.join(C.OUT, "near_ood_results", f"tobacco_near_seed{s}", "near_ood_results.json")))
        r["stored_original"] = {"mahalanobis": stored["mahalanobis"]["auroc"], "knn": stored["knn"]["auroc"],
                            "n_id_eval": stored["n_id_eval"], "n_ood": stored["n_ood"]}
        r["abs_diff"] = {"mahalanobis": abs(r["mahalanobis"]["auroc"] - stored["mahalanobis"]["auroc"]),
                         "knn": abs(r["knn_k50"]["auroc"] - stored["knn"]["auroc"])}
        r["GATE_PASS"] = bool(r["n_id"] == stored["n_id_eval"] and r["n_ood"] == stored["n_ood"]
                              and max(r["abs_diff"].values()) <= 0.002)
        r["load"] = {k: v for k, v in info.items() if k != "label2id"}; r["seconds"] = time.time() - t0
        out["seeds"][s] = r
        C.log(f"E1-R seed {s}: maha {r['mahalanobis']['auroc']:.4f} (original {stored['mahalanobis']['auroc']:.4f})  "
              f"knn {r['knn_k50']['auroc']:.4f} (original {stored['knn']['auroc']:.4f})  GATE={r['GATE_PASS']}")
        C.save_json(out, os.path.join(OUTD, "E1R_repro_submitted_table4.json"))
        del model; C.empty_device_cache()
    out["ALL_GATES_PASS"] = all(v["GATE_PASS"] for v in out["seeds"].values())
    C.save_json(out, os.path.join(OUTD, "E1R_repro_submitted_table4.json"))
    C.log(f"E1-R ALL_GATES_PASS={out['ALL_GATES_PASS']}")


# --------------------------------------------------------------------- E1-C
def lco(holdouts):
    labels = list(ds()["labels"])
    for h in holdouts:
        for s in C.SEEDS:
            path = os.path.join(OUTD, "lco", f"leaveout_{h}_seed{s}.json")
            if os.path.exists(path):
                C.log(f"skip (done) {path}"); continue
            t0 = time.time()
            run = os.path.join(C.CPMOD, "checkpoints", f"leaveout_{h}_seed{s}")
            meta = json.load(open(os.path.join(run, "training_metadata.json")))
            model, info = C.build_layoutlmv3(os.path.join(run, "best_model"))
            train_classes = list(meta["train_classes"])
            assert h not in train_classes and h not in info["label2id"], "holdout must be absent from training"
            l2i, tr, evr = doc_split(labels, train_classes, s)
            assert l2i == {k: int(v) for k, v in meta["label2id"].items()} == {k: int(v) for k, v in info["label2id"].items()}
            ood = [i for i, l in enumerate(labels) if l == h]
            f_tr, f_ev, f_ood = feats(model, tr, l2i), feats(model, evr, l2i), feats(model, ood, {})
            acc = float((f_ev["logits"].argmax(1) == f_ev["labels"]).mean())
            n_ev = len(evr)
            split_gate = abs(acc - float(meta["eval_accuracy"])) <= 1.0 / n_ev + 1e-12
            mah = MahalanobisScorer(); mah.fit(f_tr["cls_last"], f_tr["labels"])
            k50 = KNNScorer(k=50, metric="euclidean"); k50.fit(f_tr["cls_last"])
            k10 = KNNScorer(k=10, metric="euclidean"); k10.fit(f_tr["cls_last"])
            sc = {"mahalanobis": (mah.score(f_ev["cls_last"]), mah.score(f_ood["cls_last"])),
                  "knn_k50": (k50.score(f_ev["cls_last"]), k50.score(f_ood["cls_last"])),
                  "knn_k10": (k10.score(f_ev["cls_last"]), k10.score(f_ood["cls_last"])),
                  "msp": (MSPScorer().score(f_ev["logits"]), MSPScorer().score(f_ood["logits"])),
                  "energy": (EnergyScorer().score(f_ev["logits"]), EnergyScorer().score(f_ood["logits"]))}
            letter_mask = np.array([labels[i] == "Letter" for i in evr])
            r = {"holdout": h, "seed": s, "train_classes": train_classes, "n_train": len(tr), "n_eval": n_ev,
                 "n_ood": len(ood), "n_eval_letter": int(letter_mask.sum()),
                 "eval_acc_rerun": acc, "eval_acc_logged": meta["eval_accuracy"], "SPLIT_GATE_PASS": bool(split_gate),
                 "all_id_vs_ood": {k: ev(a, b) for k, (a, b) in sc.items()},
                 "letter_vs_ood": {k: ev(a[letter_mask], b) for k, (a, b) in sc.items()},
                 "load": {k: v for k, v in info.items() if k != "label2id"}, "seconds": time.time() - t0}
            os.makedirs(EMB, exist_ok=True)
            np.savez_compressed(os.path.join(EMB, f"leaveout_{h}_seed{s}.npz"),
                                tr_cls=f_tr["cls_last"].astype(np.float32), tr_y=f_tr["labels"], tr_rows=np.array(tr),
                                ev_cls=f_ev["cls_last"].astype(np.float32), ev_y=f_ev["labels"], ev_rows=np.array(evr),
                                ev_logits=f_ev["logits"].astype(np.float32),
                                ood_cls=f_ood["cls_last"].astype(np.float32), ood_rows=np.array(ood),
                                ood_logits=f_ood["logits"].astype(np.float32))
            C.save_json(r, path)
            a = r["all_id_vs_ood"]
            C.log(f"E1-C {h} seed {s}: acc {acc:.4f} (logged {meta['eval_accuracy']:.4f}) SPLIT_GATE={split_gate} | "
                  f"maha {a['mahalanobis']['auroc']:.4f} knn50 {a['knn_k50']['auroc']:.4f} knn10 {a['knn_k10']['auroc']:.4f} "
                  f"msp {a['msp']['auroc']:.4f} | letter-only maha {r['letter_vs_ood']['mahalanobis']['auroc']:.4f} "
                  f"knn50 {r['letter_vs_ood']['knn_k50']['auroc']:.4f}  ({r['seconds']:.0f}s)")
            del model; C.empty_device_cache()


# --------------------------------------------------------------------- E1-A
def ahss():
    labels = list(ds()["labels"])
    h = "Memo"
    for s in C.SEEDS:
        path = os.path.join(OUTD, "ahss", f"leaveout_{h}_seed{s}.json")
        if os.path.exists(path):
            C.log(f"skip (done) {path}"); continue
        t0 = time.time()
        z = np.load(os.path.join(EMB, f"leaveout_{h}_seed{s}.npz"))
        evr, ood = [int(i) for i in z["ev_rows"]], [int(i) for i in z["ood_rows"]]
        model, info = C.build_layoutlmv3(os.path.join(C.CPMOD, "checkpoints", f"leaveout_{h}_seed{s}", "best_model"))
        masker = C.HeadMasker(model)
        masks = C.per_layer_random_masks(25, 12, 12, seed=s)
        # re-extract the unmasked reference in this same session (and check it equals the saved one)
        u_ev = feats(model, evr, {}, want=("cls_last",), release_cache=True)
        u_ood = feats(model, ood, {}, want=("cls_last",), release_cache=True)
        drift = float(max(np.abs(u_ev["cls_last"] - z["ev_cls"]).max(), np.abs(u_ood["cls_last"] - z["ood_cls"]).max()))
        m_ev, m_ood = [], []
        for mk in masks:
            m_ev.append(feats(model, evr, {}, want=("cls_last",), masker=masker, mask=mk, release_cache=True)["cls_last"])
            m_ood.append(feats(model, ood, {}, want=("cls_last",), masker=masker, mask=mk, release_cache=True)["cls_last"])
        a_ev = -AHSSScorer.compute_ahss(u_ev["cls_last"], m_ev, normalize=True)
        a_ood = -AHSSScorer.compute_ahss(u_ood["cls_last"], m_ood, normalize=True)
        letter_mask = np.array([labels[i] == "Letter" for i in evr])
        r = {"holdout": h, "seed": s, "n_masks": len(masks),
             "mask_heads_zeroed": [int(144 - m.sum()) for m in masks],
             "unmasked_recompute_max_abs_drift": drift,
             "ahss_all_id_vs_ood": ev(a_ev, a_ood), "ahss_letter_vs_ood": ev(a_ev[letter_mask], a_ood),
             "seconds": time.time() - t0}
        C.save_json(r, path)
        C.log(f"E1-A {h} seed {s}: AHSS all-ID {r['ahss_all_id_vs_ood']['auroc']:.4f} letter {r['ahss_letter_vs_ood']['auroc']:.4f} "
              f"drift {drift:.2e} ({r['seconds']:.0f}s)")
        del model; C.empty_device_cache()


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "repro":
        repro()
    elif part == "lco":
        lco(sys.argv[2:] or HOLDOUTS)
    elif part == "ahss":
        ahss()
