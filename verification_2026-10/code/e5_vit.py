"""
E5 / E6 — ViT-B/16: asym vs sym fitting of the AHM methods (PROTOCOL.md §E5, §E6).

  python e5_vit.py <config,...> <seed> [seed...]
  configs: c10_intra c10_svhn c100_svhn c100_dtd  (E5)
           c100_c10_svhn c100_c20_svhn c100_c50_svhn (E6; c100_svhn is the 100-class cell)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import Subset
import common as C
from masked_sym import masked_block, score_block

SMOKE = os.environ.get("SMOKE") == "1"
N_TRAIN_SUB, N_EVAL_SUB, N_OOD_SUB = (64, 32, 32) if SMOKE else (3000, 1000, 1000)


def load(cfg, s):
    O = C.OUT
    if cfg == "c10_intra":
        from run_vit_experiments import load_cifar10_splits
        tr, ev, ood, _, _ = load_cifar10_splits(s, ood_class=0)
        return tr, ev, ood, os.path.join(O, "vit_checkpoints", f"cifar10_intra_seed{s}", "best_model"), 9, \
            [os.path.join(O, "vit_results", f"cifar10_intra_seed{s}")]
    if cfg == "c10_svhn":
        from run_vit_experiments import load_cifar10_cross
        tr, ev, ood, _, _ = load_cifar10_cross(s, ood_source="svhn")
        return tr, ev, ood, os.path.join(O, "vit_checkpoints", f"cifar10_svhn_seed{s}", "best_model"), 10, \
            [os.path.join(O, "vit_results", f"cifar10_svhn_seed{s}")]
    if cfg.startswith("c100"):
        from run_cifar100_benchmark import load_cifar100_data, load_ood_dataset
        parts = cfg.split("_")
        nc = int(parts[1][1:]) if len(parts) == 3 else 100
        src = parts[-1]
        tr, ev, _, tf = load_cifar100_data(s, num_classes=nc)
        ood = load_ood_dataset(src, tf)
        tag = "" if nc == 100 else f"_c{nc}"
        return tr, ev, ood, os.path.join(O, "cifar100_checkpoints", f"cifar100{tag}_seed{s}", "best_model"), nc, \
            [os.path.join(O, "cifar100_results", f"cifar100{tag}_{src}_seed{s}")]
    raise ValueError(cfg)


def stored(dirs):
    out = {}
    for d in dirs:
        for fn in ("results.json", "extended_results.json", "full_results.json"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                for k, v in json.load(open(p)).items():
                    if isinstance(v, dict) and isinstance(v.get("auroc"), (int, float)):
                        out[f"{fn.split('.')[0]}:{k}"] = v["auroc"]
        p = os.path.join(d, "mahavgavg_ahm_results.json")
        if os.path.exists(p):
            out["mahAvgAvg_AHM"] = json.load(open(p))["auroc"]
    return out


def run(cfgs, seeds):
    for cfg in cfgs:
        outd = os.path.join(C.RESULTS, ("SMOKE_" if SMOKE else "") + ("E6_vit_classcount" if "_c" in cfg[4:] else "E5_vit"), cfg)
        os.makedirs(outd, exist_ok=True)
        for s in seeds:
            path = os.path.join(outd, f"seed{s}.json")
            if os.path.exists(path):
                C.log(f"skip (done) {path}"); continue
            t0 = time.time(); C.set_seed(s)
            tr, ev, ood, ck, nl, sdirs = load(cfg, s)
            model, info = C.build_vit(ck, nl)
            rng = np.random.RandomState(s + 7)
            # sizes capped at what exists (PROTOCOL Addendum B): only c100_c10_svhn is affected (eval split = 500 < 1,000);
            # for every other config min() returns the target, so the random draws are unchanged
            idx = {"train": np.sort(rng.choice(len(tr), min(N_TRAIN_SUB, len(tr)), replace=False)),
                   "eval": np.sort(rng.choice(len(ev), min(N_EVAL_SUB, len(ev)), replace=False)),
                   "ood": np.sort(rng.choice(len(ood), min(N_OOD_SUB, len(ood)), replace=False))}
            parts = {"train": Subset(tr, idx["train"].tolist()), "eval": Subset(ev, idx["eval"].tolist()),
                     "ood": Subset(ood, idx["ood"].tolist())}
            masks = C.global_random_masks(2 if SMOKE else 25, 12, 12, seed=s)
            un, mm, disp, labels, mtop, selection = masked_block(model, parts, ("pixel_values",), masks, ("cls_last", "avg_pad"), 128)
            sc = score_block(un, mm, disp, labels["train"], ["ood"], avg_feats=("avg_pad",), knn_k=10, mtop=mtop)["ood"]
            res = {"config": cfg, "seed": s, "num_labels": nl, "load": info, "env": C.env_info(),
                   "n_full": {"train": len(tr), "eval": len(ev), "ood": len(ood)},
                   "subsample_sizes": {k: len(v) for k, v in idx.items()},
                   "masks": {"n": len(masks), "heads_zeroed": sorted({int(144 - m.sum()) for m in masks})},
                   "scores": sc, "ahm_selection": selection, "stored_original_full": stored(sdirs), "seconds": time.time() - t0}
            np.savez_compressed(os.path.join(outd, f"seed{s}_subsample_indices.npz"), **idx)
            C.save_json(res, path)
            C.log(f"E5 {cfg} s{s}: maha {sc['maha_unmasked']['auroc']:.4f} | mahAHM asym {sc['mahAHM_asym']['auroc']:.4f} "
                  f"sym {sc['mahAHM_sym']['auroc']:.4f} | knnAHM asym {sc['knnAHM10_asym']['auroc']:.4f} sym {sc['knnAHM10_sym']['auroc']:.4f} | "
                  f"AvgAHM asym {sc['mahAvgAvg_AHM_avg_pad_asym']['auroc']:.4f} sym {sc['mahAvgAvg_AHM_avg_pad_sym']['auroc']:.4f} "
                  f"unmasked {sc['mahAvgAvg_avg_pad_unmasked']['auroc']:.4f} | AHSS {sc['ahss']['auroc']:.4f} ({res['seconds']/60:.1f} min)")
            del model; C.empty_device_cache()


if __name__ == "__main__":
    run(sys.argv[1].split(","), [int(x) for x in sys.argv[2:]])
