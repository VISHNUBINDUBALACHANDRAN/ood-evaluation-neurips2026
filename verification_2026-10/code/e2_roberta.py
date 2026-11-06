"""
E2 (RoBERTa-base) and E3 (RoBERTa-large) — see PROTOCOL.md §E2, §E3.

  python e2_roberta.py base  [seeds...]   # full-data unmasked pass + masked subsample
  python e2_roberta.py large [seeds...]   # masked subsample only
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import Subset, DataLoader
import common as C
from masked_sym import masked_block, score_block, ev

from transformers import RobertaTokenizer
from run_roberta_experiments import load_sst2_data, load_ood_text_dataset, TextDataset, XMahalanobisScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer

SMOKE = os.environ.get("SMOKE") == "1"   # tiny end-to-end test; never used for reported numbers
N_TRAIN_SUB, N_EVAL_SUB, N_OOD_SUB = (64, 32, 32) if SMOKE else (3000, 1000, 1000)
OODS = ["20news", "agnews", "mnli"]


def load_20news_noheader(tok, max_length=128, max_samples=5000):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="test", data_home=os.path.join(C.PROJECT_DIR, "data"),
                           remove=("headers", "footers", "quotes"), download_if_missing=False)
    texts = [t[:512] for t in d.data if len("".join(t.split())) >= 50][:max_samples]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return TextDataset(enc, torch.tensor([-1] * len(texts)))


def ntok(dset):
    return np.array([int(dset[i]["attention_mask"].sum()) for i in range(len(dset))])


def run(size, seeds):
    outd = os.path.join(C.RESULTS, ("SMOKE_" if SMOKE else "") + f"E{'2' if size == 'base' else '3'}_roberta_{size}")
    os.makedirs(outd, exist_ok=True)
    snap = os.path.join(C.PROJECT_DIR, "data", "hf_cache", f"models--roberta-{size}", "snapshots")
    tok = RobertaTokenizer.from_pretrained(os.path.join(snap, sorted(os.listdir(snap))[0]))
    ood_names = OODS + (["20news_noheader"] if size == "base" else [])
    for s in seeds:
        path = os.path.join(outd, f"seed{s}.json")
        if os.path.exists(path):
            C.log(f"skip (done) {path}"); continue
        t0 = time.time()
        C.set_seed(s)
        train_ds, eval_ds = load_sst2_data(s, tok)
        oods = {o: load_ood_text_dataset(o, tok) for o in OODS}
        if size == "base":
            oods["20news_noheader"] = load_20news_noheader(tok)
        if SMOKE:   # exercise every code path on small slices
            train_ds = Subset(train_ds, range(300)); eval_ds = Subset(eval_ds, range(100))
            oods = {o: Subset(d, range(100)) for o, d in oods.items()}
        ck = os.path.join(C.OUT, "roberta_checkpoints", f"sst2_{'' if size == 'base' else 'robertalarge_'}seed{s}", "best_model")
        model, info = C.build_roberta(ck, size=size)
        L, H = model.config.num_hidden_layers, model.config.num_attention_heads
        res = {"size": size, "seed": s, "load": info, "L": L, "H": H, "env": C.env_info(),
               "n_full": {"train": len(train_ds), "eval": len(eval_ds), **{o: len(d) for o, d in oods.items()}}}
        keys = ("input_ids", "attention_mask")
        bs = 128 if size == "base" else 64

        # ---------------- full-data unmasked pass (base only): reproduction gate + padding analysis
        if size == "base":
            dl = lambda d: DataLoader(d, batch_size=bs, shuffle=False)
            want = ("cls_last", "cls_avg", "avg_pad", "avg_nopad")
            F_tr = C.run_features(model, dl(train_ds), keys, want)
            F_ev = C.run_features(model, dl(eval_ds), keys, want)
            F_o = {o: C.run_features(model, dl(d), keys, want) for o, d in oods.items()}
            y = F_tr["labels"]
            mah = MahalanobisScorer(); mah.fit(F_tr["cls_last"], y)
            knn = KNNScorer(k=10); knn.fit(F_tr["cls_last"])
            xm = XMahalanobisScorer(); xm._fit_maha(F_tr["cls_avg"], y)
            mpad = MahalanobisScorer(); mpad.fit(F_tr["avg_pad"], y)
            mnop = MahalanobisScorer(); mnop.fit(F_tr["avg_nopad"], y)
            len_ev = ntok(eval_ds)
            full = {}
            for o, Fo in F_o.items():
                r = {"Mahalanobis": ev(mah.score(F_ev["cls_last"]), mah.score(Fo["cls_last"])),
                     "kNN10": ev(knn.score(F_ev["cls_last"]), knn.score(Fo["cls_last"])),
                     "X-Maha_uniform": ev(xm._score(F_ev["cls_avg"]), xm._score(Fo["cls_avg"])),
                     "mahAvgAvg_pad_unmasked": ev(mpad.score(F_ev["avg_pad"]), mpad.score(Fo["avg_pad"])),
                     "mahAvgAvg_nopad_unmasked": ev(mnop.score(F_ev["avg_nopad"]), mnop.score(Fo["avg_nopad"])),
                     "length_only": ev(-len_ev, -ntok(oods[o])),
                     "mean_tokens": {"id": float(len_ev.mean()), "ood": float(ntok(oods[o]).mean())}}
                if o in OODS:   # reproduction gate vs stored original per-seed values
                    st = json.load(open(os.path.join(C.OUT, "roberta_results", f"sst2_{o}_seed{s}", "full_results.json")))
                    sa = json.load(open(os.path.join(C.OUT, "roberta_results", f"sst2_{o}_seed{s}", "mahavgavg_ahm_results.json")))
                    r["stored_original"] = {"Mahalanobis": st["Mahalanobis"]["auroc"], "kNN": st["kNN"]["auroc"],
                                        "X-Maha_uniform": st["X-Maha_uniform"]["auroc"], "mahAHM": st["mahAHM"]["auroc"],
                                        "knnAHM": st["knnAHM"]["auroc"], "AHSS": st["AHSS"]["auroc"],
                                        "mahAvgAvg_AHM": sa["auroc"]}
                    d = {k: abs(r[k2]["auroc"] - st[k]["auroc"]) for k, k2 in
                         (("Mahalanobis", "Mahalanobis"), ("kNN", "kNN10"), ("X-Maha_uniform", "X-Maha_uniform"))}
                    r["repro_abs_diff"] = d
                    r["REPRO_GATE_PASS"] = bool(max(d.values()) <= 0.002)
                full[o] = r
                C.log(f"  [{size} s{s}] FULL {o:16s} Maha {r['Mahalanobis']['auroc']:.4f} kNN {r['kNN10']['auroc']:.4f} "
                      f"XMu {r['X-Maha_uniform']['auroc']:.4f} AvgPad {r['mahAvgAvg_pad_unmasked']['auroc']:.4f} "
                      f"AvgNoPad {r['mahAvgAvg_nopad_unmasked']['auroc']:.4f} LEN {r['length_only']['auroc']:.4f} "
                      f"gate={r.get('REPRO_GATE_PASS')}")
            res["full"] = full
            del F_tr, F_ev, F_o

        # ---------------- masked subsample: asym vs sym
        rng = np.random.RandomState(s + 7)
        idx = {"train": np.sort(rng.choice(len(train_ds), N_TRAIN_SUB, replace=False)),
               "eval": np.sort(rng.choice(len(eval_ds), N_EVAL_SUB, replace=False))}
        for o, d in oods.items():
            idx[o] = np.sort(rng.choice(len(d), N_OOD_SUB, replace=False))
        parts = {"train": Subset(train_ds, idx["train"].tolist()), "eval": Subset(eval_ds, idx["eval"].tolist())}
        parts.update({o: Subset(d, idx[o].tolist()) for o, d in oods.items()})
        masks = C.global_random_masks(2 if SMOKE else 25, L, H, seed=s)
        res["masks"] = {"n": len(masks), "heads_zeroed": sorted({int(L * H - m.sum()) for m in masks}),
                        "mean_fraction_zeroed": float(np.mean([1 - m.mean() for m in masks]))}
        un, mm, disp, labels, mtop, selection = masked_block(model, parts, keys, masks, ("cls_last", "avg_pad", "avg_nopad"), bs)
        sub = score_block(un, mm, disp, labels["train"], list(oods), avg_feats=("avg_pad", "avg_nopad"), knn_k=10, mtop=mtop)
        if size == "base":
            for o in ood_names:
                sub[o]["length_only"] = ev(-ntok(parts["eval"]), -ntok(parts[o]))
        tag = "" if size == "base" else "robertalarge_"
        for o in OODS:   # stored original full-data values (original asym definitions) for the approximate gate
            d = os.path.join(C.OUT, "roberta_results", f"sst2_{tag}{o}_seed{s}")
            st = json.load(open(os.path.join(d, "full_results.json")))
            sub[o]["stored_original_full"] = {k: st[k]["auroc"] for k in ("Mahalanobis", "kNN", "mahAHM", "knnAHM", "AHSS") if k in st}
            if os.path.exists(os.path.join(d, "mahavgavg_ahm_results.json")):
                sub[o]["stored_original_full"]["mahAvgAvg_AHM"] = json.load(open(os.path.join(d, "mahavgavg_ahm_results.json")))["auroc"]
        res["subsample"] = {"sizes": {k: len(v) for k, v in idx.items()}, "scores": sub, "ahm_selection": selection}
        np.savez_compressed(os.path.join(outd, f"seed{s}_subsample_indices.npz"), **{k: v for k, v in idx.items()})
        res["seconds"] = time.time() - t0
        C.save_json(res, path)
        for o in ood_names:
            q = sub[o]
            C.log(f"  [{size} s{s}] SUB {o:16s} mahAHM asym {q['mahAHM_asym']['auroc']:.4f} sym {q['mahAHM_sym']['auroc']:.4f} | "
                  f"knnAHM asym {q['knnAHM10_asym']['auroc']:.4f} sym {q['knnAHM10_sym']['auroc']:.4f} | "
                  f"AvgAHM pad asym {q['mahAvgAvg_AHM_avg_pad_asym']['auroc']:.4f} nopad asym {q['mahAvgAvg_AHM_avg_nopad_asym']['auroc']:.4f} "
                  f"nopad sym {q['mahAvgAvg_AHM_avg_nopad_sym']['auroc']:.4f} | AHSS {q['ahss']['auroc']:.4f}")
        C.log(f"[{size} seed {s}] done in {res['seconds']/60:.1f} min")
        del model; C.empty_device_cache()


if __name__ == "__main__":
    size = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or C.SEEDS
    run(size, seeds)
