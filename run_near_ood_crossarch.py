"""
run_near_ood_crossarch.py — first design of experiment P3 (near-OOD beyond documents).
Not used for the reported near-OOD results; those come from run_nearood_polarity.py.

Extends the near-OOD analysis beyond document data.

Design. In every configuration the backbone, its weights, the OOD set and the
scoring code are held fixed; only the ID reference set changes. That isolates
near-vs-far semantic distance from every other factor, and it needs no new
fine-tuning, so it reuses checkpoints whose numbers are already in the paper.

VISION (ViT-B/16, existing cifar10_intra checkpoints; OOD = airplane throughout)
  far  : ID = the six animal classes      (animal  vs vehicle -> semantically far)
  near : ID = automobile, ship, truck     (vehicle vs vehicle -> semantically near)

TEXT (RoBERTa-base, existing sst2 checkpoints; ID = SST-2 throughout)
  far  : OOD = 20-Newsgroups comp.*   (technical prose, distant from film reviews)
  near : OOD = 20-Newsgroups rec.*    (leisure/entertainment prose, closer)

The text arm varies the OOD side rather than the ID side because SST-2 is a
single binary task with no semantic sub-structure to partition. We report it as
an OOD-distance contrast and do not overclaim it as a strict class-holdout
near-OOD split.

Scoring uses the same Mahalanobis / kNN / AHSS implementations as the paper.

Usage:
    python run_near_ood_crossarch.py --arch vision --seeds 42
    python run_near_ood_crossarch.py --arch both
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from evaluation.metrics import compute_auroc, compute_fpr_at_tpr  # noqa: E402
from ood_methods.distance_based import MahalanobisScorer, KNNScorer  # noqa: E402
from ood_methods.sensitivity_based import AHSSScorer  # noqa: E402
from utils.model_io import load_model_weights, model_weights_exist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]

# CIFAR-10: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
#           5 dog, 6 frog, 7 horse, 8 ship, 9 truck
CIFAR10_OOD_CLASS = 0                      # airplane, held out at fine-tuning time
CIFAR10_VEHICLES = [1, 8, 9]               # automobile, ship, truck  -> NEAR to airplane
CIFAR10_ANIMALS = [2, 3, 4, 5, 6, 7]       # bird..horse              -> FAR from airplane

NEWS_FAR_PREFIX = "comp."
NEWS_NEAR_PREFIX = "rec."


# ---------------------------------------------------------------------------
# Shared scoring
# ---------------------------------------------------------------------------

def score_all(train_emb, train_lab, id_emb, ood_emb,
              id_unmasked=None, id_masked=None, ood_unmasked=None, ood_masked=None):
    """Mahalanobis / kNN (+ AHSS when masked embeddings are supplied)."""
    out = {}

    mah = MahalanobisScorer()
    mah.fit(train_emb, train_lab)
    s_id, s_ood = mah.score(id_emb), mah.score(ood_emb)
    out["mahalanobis"] = {"auroc": float(compute_auroc(s_id, s_ood)),
                          "fpr_at_95": float(compute_fpr_at_tpr(s_id, s_ood))}

    knn = KNNScorer(k=50, metric="euclidean")
    knn.fit(train_emb)
    s_id, s_ood = knn.score(id_emb), knn.score(ood_emb)
    out["knn"] = {"auroc": float(compute_auroc(s_id, s_ood)),
                  "fpr_at_95": float(compute_fpr_at_tpr(s_id, s_ood))}

    if id_masked is not None:
        sc = AHSSScorer(normalize=True)
        a_id = -sc.compute_ahss(id_unmasked, id_masked, normalize=True)
        a_ood = -sc.compute_ahss(ood_unmasked, ood_masked, normalize=True)
        out["ahss"] = {"auroc": float(compute_auroc(a_id, a_ood)),
                       "fpr_at_95": float(compute_fpr_at_tpr(a_id, a_ood))}
    return out


# ---------------------------------------------------------------------------
# Vision arm
# ---------------------------------------------------------------------------

def run_vision(args):
    from models.vit_classifier import (
        ViTClassifier, extract_vit_embeddings_from_dataloader,
        extract_vit_embeddings_with_multiple_masks,
    )
    from run_vit_experiments import load_cifar10_splits, generate_random_masks

    device, results = args.device, []
    for seed in args.seeds:
        ckpt = os.path.join(SCRIPT_DIR, "outputs", "vit_checkpoints",
                            f"cifar10_intra_seed{seed}", "best_model")
        if not model_weights_exist(ckpt, "model"):
            logger.warning("SKIP vision seed %s: no checkpoint at %s", seed, ckpt)
            continue

        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_splits(
            seed, ood_class=CIFAR10_OOD_CLASS)

        model = ViTClassifier(num_labels=len(id_classes))
        model.load_state_dict(load_model_weights(ckpt, "model"), strict=False)
        model.to(device).eval()

        ld = dict(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        tr_emb, tr_lab = extract_vit_embeddings_from_dataloader(
            model, DataLoader(_cap(train_ds, args.n_train, seed), **ld), device=device)
        ev_ds = _cap(eval_ds, args.n_eval, seed)
        ev_emb, ev_lab = extract_vit_embeddings_from_dataloader(
            model, DataLoader(ev_ds, **ld), device=device)
        od_ds = _cap(ood_ds, args.n_eval, seed)
        od_emb, _ = extract_vit_embeddings_from_dataloader(
            model, DataLoader(od_ds, **ld), device=device)

        # label_map sends original CIFAR-10 ids -> contiguous training ids
        veh = {label_map[c] for c in CIFAR10_VEHICLES if c in label_map}
        ani = {label_map[c] for c in CIFAR10_ANIMALS if c in label_map}

        for tag, keep in (("far_animals_vs_airplane", ani), ("near_vehicles_vs_airplane", veh)):
            sel = np.array([i for i, l in enumerate(ev_lab) if int(l) in keep])
            if len(sel) < 20:
                logger.warning("  %s: only %d ID eval samples, skipping", tag, len(sel))
                continue
            t0 = time.time()
            r = score_all(tr_emb, tr_lab, ev_emb[sel], od_emb)
            row = {"architecture": "ViT-B/16", "modality": "vision", "config": tag,
                   "regime": "near" if tag.startswith("near") else "far",
                   "seed": seed, "n_id": int(len(sel)), "n_ood": int(len(od_emb)),
                   "elapsed_seconds": time.time() - t0, **r}
            results.append(row)
            logger.info("  ViT seed=%-5s %-28s Maha=%.4f kNN=%.4f (n_id=%d)",
                        seed, tag, r["mahalanobis"]["auroc"], r["knn"]["auroc"], len(sel))

        del model
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Text arm
# ---------------------------------------------------------------------------

def _news_subset(prefix, tokenizer, max_len, n):
    """20-Newsgroups documents whose category starts with `prefix`."""
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="test", data_home=os.path.join(SCRIPT_DIR, "data"),
                           remove=("headers", "footers", "quotes"))
    names = d.target_names
    texts = [t for t, y in zip(d.data, d.target)
             if names[y].startswith(prefix) and len(t.strip()) > 30]
    if not texts:
        raise RuntimeError(f"20news subset '{prefix}' produced 0 documents")
    rng = np.random.RandomState(0)
    if n and len(texts) > n:
        texts = [texts[i] for i in rng.choice(len(texts), n, replace=False)]
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")

    class _DS(Dataset):
        def __len__(self): return enc["input_ids"].shape[0]
        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": torch.tensor(0)}
    return _DS()


def run_text(args):
    from transformers import RobertaTokenizer
    from run_roberta_experiments import (
        RoBERTaClassifier, load_sst2_data, extract_logits_and_embeddings,
    )

    device, results = args.device, []
    tok = RobertaTokenizer.from_pretrained(
        "roberta-base", cache_dir=os.path.join(SCRIPT_DIR, "data", "hf_cache"))

    for seed in args.seeds:
        ckpt = os.path.join(SCRIPT_DIR, "outputs", "roberta_checkpoints",
                            f"sst2_seed{seed}", "best_model")
        if not model_weights_exist(ckpt, "model"):
            logger.warning("SKIP text seed %s: no checkpoint at %s", seed, ckpt)
            continue

        model = RoBERTaClassifier(num_labels=2)
        model.load_state_dict(load_model_weights(ckpt, "model"), strict=False)
        model.to(device).eval()

        train_ds, eval_ds = load_sst2_data(seed, tok)
        ld = dict(batch_size=args.batch_size_text, shuffle=False, num_workers=2)
        _, tr_emb, tr_lab = extract_logits_and_embeddings(
            model, DataLoader(_cap(train_ds, args.n_train, seed), **ld), device)
        _, ev_emb, _ = extract_logits_and_embeddings(
            model, DataLoader(_cap(eval_ds, args.n_eval, seed), **ld), device)

        for tag, prefix in (("far_sst2_vs_20news_comp", NEWS_FAR_PREFIX),
                            ("near_sst2_vs_20news_rec", NEWS_NEAR_PREFIX)):
            ood_ds = _news_subset(prefix, tok, args.max_len, args.n_eval)
            _, od_emb, _ = extract_logits_and_embeddings(model, DataLoader(ood_ds, **ld), device)
            t0 = time.time()
            r = score_all(tr_emb, tr_lab, ev_emb, od_emb)
            row = {"architecture": "RoBERTa-base", "modality": "text", "config": tag,
                   "regime": "near" if tag.startswith("near") else "far",
                   "seed": seed, "n_id": int(len(ev_emb)), "n_ood": int(len(od_emb)),
                   "elapsed_seconds": time.time() - t0, **r}
            results.append(row)
            logger.info("  RoBERTa seed=%-5s %-26s Maha=%.4f kNN=%.4f (n_ood=%d)",
                        seed, tag, r["mahalanobis"]["auroc"], r["knn"]["auroc"], len(od_emb))

        del model
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------

def _cap(ds, n, seed):
    if not n or n >= len(ds):
        return ds
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    return Subset(ds, idx.tolist())


def summarise(rows, path):
    out = {}
    for key in sorted({(r["modality"], r["config"]) for r in rows}):
        rs = [r for r in rows if (r["modality"], r["config"]) == key]
        e = {"regime": rs[0]["regime"], "n_seeds": len(rs)}
        for m in ("mahalanobis", "knn", "ahss"):
            v = [r[m]["auroc"] for r in rs if m in r]
            if v:
                e[m] = {"auroc_mean": float(np.mean(v)), "auroc_std": float(np.std(v))}
        out[f"{key[0]}|{key[1]}"] = e
    json.dump(out, open(path, "w"), indent=2)
    logger.info("\n%-14s %-30s %-8s %-18s %-18s", "modality", "config", "regime", "Mahalanobis", "kNN")
    for k, e in out.items():
        mo, cf = k.split("|")
        f = lambda m: f"{e[m]['auroc_mean']:.4f}+-{e[m]['auroc_std']:.4f}" if m in e else "n/a"
        logger.info("%-14s %-30s %-8s %-18s %-18s", mo, cf, e["regime"], f("mahalanobis"), f("knn"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vision", "text", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--batch_size_text", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "outputs",
                                                  "near_ood_crossarch",
                                                  "near_ood_crossarch.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    if args.arch in ("vision", "both"):
        logger.info("=" * 70); logger.info("VISION near-vs-far (ViT-B/16, OOD = airplane)")
        rows += run_vision(args)
        json.dump(rows, open(args.out, "w"), indent=2)
    if args.arch in ("text", "both"):
        logger.info("=" * 70); logger.info("TEXT near-vs-far (RoBERTa-base, ID = SST-2)")
        rows += run_text(args)

    json.dump(rows, open(args.out, "w"), indent=2)
    logger.info("Wrote %d rows to %s", len(rows), args.out)
    if rows:
        summarise(rows, args.out.replace(".json", "_summary.json"))


if __name__ == "__main__":
    main()
