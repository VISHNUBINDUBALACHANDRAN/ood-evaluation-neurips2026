"""
run_nearood_polarity.py — experiments P3 and P6.

P3  near-OOD outside the document modality.
P6  does the AHSS polarity inversion track ID input length?

Both ride on the same fine-tunes, so the pair costs one training run per seed
per modality rather than two.

DESIGN. A near-OOD regime requires the held-out class to sit *inside* the
training manifold. It is therefore not enough to re-slice which ID samples are
scored: the model itself must be trained only on the near neighbours, so the
class-conditional Gaussian reference is built from them.

With the reference fixed by training, near vs far is then varied on the OOD
side, which is the standard construction:

  VISION   ViT-B/16 fine-tuned on four CIFAR-100 small mammals
             (hamster, mouse, rabbit, shrew)
             near-OOD : squirrel   (5th sibling of the same superclass)
             far-OOD  : rocket     (vehicles_2, a distant superclass)
  TEXT     RoBERTa-base fine-tuned on 20-Newsgroups comp.* (5 computing classes)
             near-OOD : sci.electronics        (technical, adjacent)
             far-OOD  : talk.politics.mideast  (distant)

P6 falls out of the text arm: its ID set is long-form newsgroup prose, whereas
the paper's RoBERTa ID set (SST-2) is single short sentences. Same backbone,
same AHSS estimator, only ID length differs. If AHSS behaves normally here
(AUROC > 0.5) while it inverts on SST-2 (0.192-0.384), the input-length
hypothesis is supported over the pre-training-objective one.

Usage:
    python run_nearood_polarity.py --arch vision --seeds 42        # smoke
    python run_nearood_polarity.py --arch both
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from evaluation.metrics import compute_auroc, compute_fpr_at_tpr          # noqa: E402
from ood_methods.distance_based import MahalanobisScorer, KNNScorer       # noqa: E402
from ood_methods.sensitivity_based import AHSSScorer                      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]

# VISION SPLIT (CIFAR-100, fine-grained inside one superclass): the near class
# is a sibling from the SAME superclass, so the contrast is genuinely hard.
#   superclass small_mammals = hamster, mouse, rabbit, shrew, squirrel
CIFAR100_ID_NAMES = ["hamster", "mouse", "rabbit", "shrew"]
CIFAR100_NEAR_NAME = "squirrel"    # 5th sibling of the SAME superclass
CIFAR100_FAR_NAME = "rocket"       # vehicles_2, a distant superclass

NEWS_ID_PREFIX = "comp."
NEWS_NEAR_OOD = "sci.electronics"
NEWS_FAR_OOD = "talk.politics.mideast"


# ---------------------------------------------------------------------------

def score_block(train_emb, train_lab, id_emb, ood_emb,
                id_un=None, id_mk=None, ood_un=None, ood_mk=None):
    out = {}
    mah = MahalanobisScorer(); mah.fit(train_emb, train_lab)
    a, b = mah.score(id_emb), mah.score(ood_emb)
    out["mahalanobis"] = {"auroc": float(compute_auroc(a, b)),
                          "fpr_at_95": float(compute_fpr_at_tpr(a, b))}
    knn = KNNScorer(k=50, metric="euclidean"); knn.fit(train_emb)
    a, b = knn.score(id_emb), knn.score(ood_emb)
    out["knn"] = {"auroc": float(compute_auroc(a, b)),
                  "fpr_at_95": float(compute_fpr_at_tpr(a, b))}
    if id_mk is not None:
        sc = AHSSScorer(normalize=True)
        a = -sc.compute_ahss(id_un, id_mk, normalize=True)
        b = -sc.compute_ahss(ood_un, ood_mk, normalize=True)
        out["ahss"] = {"auroc": float(compute_auroc(a, b)),
                       "fpr_at_95": float(compute_fpr_at_tpr(a, b))}
    return out


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def _cifar_subsets(seed, transform, n_train, n_eval):
    """CIFAR-100 fine-grained: ID = 4 siblings, near = 5th sibling, far = distant class."""
    from torchvision.datasets import CIFAR100
    root = os.path.join(SCRIPT_DIR, "data", "cifar100")
    tr_raw = CIFAR100(root=root, train=True, download=False)
    te_raw = CIFAR100(root=root, train=False, download=False)
    name2id = {c: i for i, c in enumerate(tr_raw.classes)}
    for nm in CIFAR100_ID_NAMES + [CIFAR100_NEAR_NAME, CIFAR100_FAR_NAME]:
        if nm not in name2id:
            raise RuntimeError(f"CIFAR-100 class {nm!r} not found")
    lmap = {name2id[c]: i for i, c in enumerate(CIFAR100_ID_NAMES)}

    class _DS(Dataset):
        def __init__(self, base, idx, remap):
            self.b, self.i, self.r = base, idx, remap
        def __len__(self): return len(self.i)
        def __getitem__(self, k):
            img, lab = self.b[self.i[k]]
            return {"pixel_values": transform(img),
                    "labels": self.r[lab] if self.r else -1}

    rng = np.random.RandomState(seed)
    tr_idx = [i for i, t in enumerate(tr_raw.targets) if t in lmap]
    rng.shuffle(tr_idx)
    cut = int(0.9 * len(tr_idx))
    train_i, eval_i = tr_idx[:cut], tr_idx[cut:]
    if n_train and len(train_i) > n_train: train_i = train_i[:n_train]
    if n_eval and len(eval_i) > n_eval:   eval_i = eval_i[:n_eval]

    class _Sub(Dataset):
        def __init__(self, base, idx): self.b, self.i = base, idx
        def __len__(self): return len(self.i)
        def __getitem__(self, k): return self.b[self.i[k]]

    class _Concat(Dataset):
        """OOD class pooled from BOTH splits: the test split alone gives only 100
        images per CIFAR-100 class. These classes are held out of ID training
        entirely, so their train-split images are equally unseen and valid as OOD."""
        def __init__(self, parts): self.parts = parts
        def __len__(self): return sum(len(p) for p in self.parts)
        def __getitem__(self, k):
            for p in self.parts:
                if k < len(p): return p[k]
                k -= len(p)
            raise IndexError(k)

    def ood_of(name):
        cid = name2id[name]
        te_idx = [i for i, t in enumerate(te_raw.targets) if t == cid]
        tr_idx = [i for i, t in enumerate(tr_raw.targets) if t == cid]
        parts = [_DS(te_raw, te_idx, None), _DS(tr_raw, tr_idx, None)]
        ds = _Concat(parts)
        if n_eval and len(ds) > n_eval:
            keep = np.random.RandomState(seed).choice(len(ds), n_eval, replace=False)
            return _Sub(ds, keep.tolist())
        return ds


    return (_DS(tr_raw, train_i, lmap), _DS(tr_raw, eval_i, lmap),
            ood_of(CIFAR100_NEAR_NAME), ood_of(CIFAR100_FAR_NAME), len(lmap))


def run_vision(args):
    from models.vit_classifier import (ViTClassifier,
                                       extract_vit_embeddings_from_dataloader,
                                       extract_vit_embeddings_with_multiple_masks)
    from run_vit_experiments import finetune_vit, get_vit_transform, generate_random_masks
    from utils.model_io import model_weights_exist

    dev, rows = args.device, []
    tf = get_vit_transform()
    for seed in args.seeds:
        out_dir = os.path.join(SCRIPT_DIR, "outputs", "nearood_polarity",
                               f"vit_smallmammals_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        tr, ev, ood_near, ood_far, nlab = _cifar_subsets(
            seed, tf, args.n_train, args.n_eval)
        logger.info("seed %s vision: train=%d eval=%d near=%d far=%d (%d ID classes)",
                    seed, len(tr), len(ev), len(ood_near), len(ood_far), nlab)

        model = finetune_vit(tr, ev, nlab, seed, out_dir,
                             epochs=args.epochs_vit, lr=2e-5, batch_size=args.bs)
        model.to(dev).eval()
        ld = dict(batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        masks = generate_random_masks(args.n_masks, seed=seed)

        tr_e, tr_l = extract_vit_embeddings_from_dataloader(model, DataLoader(tr, **ld), device=dev)
        ev_u, ev_m, _ = extract_vit_embeddings_with_multiple_masks(model, DataLoader(ev, **ld), masks, device=dev)
        for tag, ds in ((f"near_{CIFAR100_NEAR_NAME}", ood_near), (f"far_{CIFAR100_FAR_NAME}", ood_far)):
            od_u, od_m, _ = extract_vit_embeddings_with_multiple_masks(model, DataLoader(ds, **ld), masks, device=dev)
            t0 = time.time()
            r = score_block(tr_e, tr_l, ev_u, od_u, ev_u, ev_m, od_u, od_m)
            rows.append({"modality": "vision", "architecture": "ViT-B/16",
                         "id_set": "cifar100_small_mammals", "config": tag,
                         "regime": "near" if tag.startswith("near") else "far",
                         "seed": seed, "n_id": len(ev_u), "n_ood": len(od_u),
                         "elapsed_seconds": time.time() - t0, **r})
            logger.info("  seed=%-5s %-14s Maha=%.4f kNN=%.4f AHSS=%.4f",
                        seed, tag, r["mahalanobis"]["auroc"], r["knn"]["auroc"], r["ahss"]["auroc"])
        del model; torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _news(prefixes, tok, max_len, n, seed, exact=False):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", data_home=os.path.join(SCRIPT_DIR, "data"),
                           remove=("headers", "footers", "quotes"))
    names = d.target_names
    keep = [(t, names[y]) for t, y in zip(d.data, d.target)
            if (names[y] in prefixes if exact else any(names[y].startswith(p) for p in prefixes))
            and len(t.strip()) > 50]
    if not keep:
        raise RuntimeError(f"20news selection {prefixes} produced 0 documents")
    rng = np.random.RandomState(seed)
    if n and len(keep) > n:
        keep = [keep[i] for i in rng.choice(len(keep), n, replace=False)]
    cats = sorted({c for _, c in keep})
    cmap = {c: i for i, c in enumerate(cats)}
    enc = tok([t for t, _ in keep], truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    labs = torch.tensor([cmap[c] for _, c in keep])

    class _DS(Dataset):
        def __len__(self): return labs.shape[0]
        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i], "labels": labs[i]}
    return _DS(), len(cats)


def run_text(args):
    from transformers import RobertaTokenizer
    from run_roberta_experiments import (RoBERTaClassifier, finetune_roberta,
                                         extract_logits_and_embeddings,
                                         extract_multi_mask_embeddings,
                                         generate_random_masks)
    from torch.utils.data import Subset

    dev, rows = args.device, []
    tok = RobertaTokenizer.from_pretrained(
        "roberta-base", cache_dir=os.path.join(SCRIPT_DIR, "data", "hf_cache"))

    for seed in args.seeds:
        out_dir = os.path.join(SCRIPT_DIR, "outputs", "nearood_polarity",
                               f"roberta_20news_comp_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        full, ncls = _news([NEWS_ID_PREFIX], tok, args.max_len, None, seed)
        idx = np.random.RandomState(seed).permutation(len(full))
        cut = int(0.9 * len(idx))
        tr = Subset(full, idx[:cut].tolist())
        ev = Subset(full, idx[cut:][:args.n_eval].tolist())
        near, _ = _news([NEWS_NEAR_OOD], tok, args.max_len, args.n_eval, seed, exact=True)
        far, _ = _news([NEWS_FAR_OOD], tok, args.max_len, args.n_eval, seed, exact=True)
        logger.info("seed %s text: train=%d eval=%d near=%d far=%d (%d ID classes)",
                    seed, len(tr), len(ev), len(near), len(far), ncls)

        # 20news comp.* has 5 classes; finetune_roberta defaults to 2 (SST-2).
        model = finetune_roberta(tr, ev, seed, out_dir, epochs=args.epochs_txt,
                                 lr=2e-5, batch_size=args.bs_txt, num_labels=ncls)
        model.to(dev).eval()
        ld = dict(batch_size=args.bs_txt, shuffle=False, num_workers=2)
        masks = generate_random_masks(args.n_masks, seed=seed)

        _, tr_e, tr_l = extract_logits_and_embeddings(model, DataLoader(tr, **ld), dev)
        ev_u, ev_m, _ = extract_multi_mask_embeddings(model, DataLoader(ev, **ld), masks, dev)
        for tag, ds in (("near_sci_electronics", near), ("far_talk_politics", far)):
            od_u, od_m, _ = extract_multi_mask_embeddings(model, DataLoader(ds, **ld), masks, dev)
            t0 = time.time()
            r = score_block(tr_e, tr_l, ev_u, od_u, ev_u, ev_m, od_u, od_m)
            rows.append({"modality": "text", "architecture": "RoBERTa-base",
                         "id_set": "20news_comp_LONG", "config": tag,
                         "regime": "near" if tag.startswith("near") else "far",
                         "seed": seed, "n_id": len(ev_u), "n_ood": len(od_u),
                         "elapsed_seconds": time.time() - t0, **r})
            logger.info("  seed=%-5s %-22s Maha=%.4f kNN=%.4f AHSS=%.4f",
                        seed, tag, r["mahalanobis"]["auroc"], r["knn"]["auroc"], r["ahss"]["auroc"])
        del model; torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------

def summarise(rows, path):
    out = {}
    for k in sorted({(r["modality"], r["config"]) for r in rows}):
        rs = [r for r in rows if (r["modality"], r["config"]) == k]
        e = {"regime": rs[0]["regime"], "id_set": rs[0]["id_set"], "n_seeds": len(rs)}
        for m in ("mahalanobis", "knn", "ahss"):
            v = [r[m]["auroc"] for r in rs if m in r]
            if v: e[m] = {"auroc_mean": float(np.mean(v)), "auroc_std": float(np.std(v))}
        out[f"{k[0]}|{k[1]}"] = e
    json.dump(out, open(path, "w"), indent=2)
    logger.info("\n%-8s %-24s %-7s %-16s %-16s %-16s",
                "mod", "config", "regime", "Mahalanobis", "kNN", "AHSS")
    for k, e in out.items():
        mo, cf = k.split("|")
        f = lambda m: f"{e[m]['auroc_mean']:.4f}+-{e[m]['auroc_std']:.4f}" if m in e else "n/a"
        logger.info("%-8s %-24s %-7s %-16s %-16s %-16s",
                    mo, cf, e["regime"], f("mahalanobis"), f("knn"), f("ahss"))
    # sanity: near must be harder than far
    for mo in {r["modality"] for r in rows}:
        n = [e for k, e in out.items() if k.startswith(mo) and e["regime"] == "near"]
        fa = [e for k, e in out.items() if k.startswith(mo) and e["regime"] == "far"]
        if n and fa and "mahalanobis" in n[0] and "mahalanobis" in fa[0]:
            d = fa[0]["mahalanobis"]["auroc_mean"] - n[0]["mahalanobis"]["auroc_mean"]
            logger.info("SANITY %-8s far - near = %+.4f  %s", mo, d,
                        "OK (near is harder)" if d > 0 else "** INVERTED, design suspect **")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vision", "text", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=800)
    ap.add_argument("--n_masks", type=int, default=25)
    ap.add_argument("--epochs_vit", type=int, default=5)
    ap.add_argument("--epochs_txt", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--bs_txt", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "outputs", "nearood_polarity",
                                                  "nearood_polarity.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    rows = []
    if a.arch in ("vision", "both"):
        logger.info("=" * 70); logger.info("P3 VISION near-vs-far (ID = CIFAR-10 vehicles)")
        rows += run_vision(a); json.dump(rows, open(a.out, "w"), indent=2)
    if a.arch in ("text", "both"):
        logger.info("=" * 70); logger.info("P3/P6 TEXT near-vs-far (ID = 20news comp.*, LONG inputs)")
        rows += run_text(a)
    json.dump(rows, open(a.out, "w"), indent=2)
    logger.info("Wrote %d rows to %s", len(rows), a.out)
    if rows: summarise(rows, a.out.replace(".json", "_summary.json"))


if __name__ == "__main__":
    main()
