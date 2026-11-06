"""
run_mcdropout_mechanism.py — experiment R1.

Converts the claim "MC-Dropout collapses on ViT because ViT has no learnable
dropout" from a hypothesis into a demonstrated mechanism, by intervening on the
dropout rate at inference time and measuring both the raw MC variance magnitude
and the resulting AUROC.

The design is a 2x2 intervention:

    ViT-B/16      p = 0.0 (native)  -> predict: variance ~ 0, AUROC ~ 0.5
    ViT-B/16      p > 0  (injected) -> predict: variance > 0, AUROC recovers
    LayoutLMv3    p = 0.1 (native)  -> predict: variance > 0, AUROC usable
    LayoutLMv3    p = 0.0 (ablated) -> predict: variance ~ 0, AUROC ~ 0.5

If both directions hold, the collapse is caused by the dropout rate and not by
anything else about the backbone.

Reported per cell: mean/median per-sample MC variance on ID and OOD, and the
AUROC under the paper's standard sign convention.

Usage:
    python run_mcdropout_mechanism.py --arch vit --n_eval 1000 --n_passes 25
    python run_mcdropout_mechanism.py --arch layoutlmv3
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from evaluation.metrics import compute_auroc  # noqa: E402
from utils.model_io import load_model_weights, model_weights_exist  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Dropout introspection and intervention
# ---------------------------------------------------------------------------

def audit_dropout(model):
    """Return the config-declared dropout probabilities and the live nn.Dropout rates."""
    cfg = model.model.config
    declared = {
        k: getattr(cfg, k)
        for k in ("hidden_dropout_prob", "attention_probs_dropout_prob",
                  "classifier_dropout", "classifier_dropout_prob", "drop_path_rate")
        if hasattr(cfg, k)
    }
    live = sorted({float(m.p) for m in model.modules() if isinstance(m, nn.Dropout)})
    n_modules = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    return {"config_declared": declared, "live_dropout_rates": live,
            "n_dropout_modules": n_modules}


def set_dropout(model, p):
    """Force every nn.Dropout module in the model to probability p."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)
            n += 1
    return n


# ---------------------------------------------------------------------------
# MC sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_variance(model, loader, forward_fn, n_passes, device, normalize=True):
    """
    Run n_passes stochastic forward passes with the model in train() mode and
    return the per-sample summed variance of the CLS embedding.

    Matches ood_methods/sensitivity_based.MCDropoutScorer.compute_mc_uncertainty:
    L2-normalize each pass, variance across passes, sum over feature dims.
    """
    passes = []
    for _ in range(n_passes):
        model.train()  # activates nn.Dropout (a no-op when every p == 0)
        chunks = []
        for batch in loader:
            emb = forward_fn(model, batch, device)
            chunks.append(emb.detach().cpu().numpy().astype(np.float64))
        passes.append(np.concatenate(chunks, axis=0))
    model.eval()

    stacked = np.stack(passes, axis=0)  # (T, N, D)
    if normalize:
        norms = np.linalg.norm(stacked, axis=2, keepdims=True) + 1e-8
        stacked = stacked / norms
    return np.sum(np.var(stacked, axis=0), axis=1)  # (N,)


def _vit_forward(model, batch, device):
    out = model.forward(pixel_values=batch["pixel_values"].to(device))
    return out["hidden_states"][-1][:, 0, :]


def _layoutlm_forward(model, batch, device):
    out = model.forward(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        bbox=batch["bbox"].to(device),
        pixel_values=batch["pixel_values"].to(device),
    )
    return out["hidden_states"][-1][:, 0, :]


# ---------------------------------------------------------------------------
# Per-architecture drivers
# ---------------------------------------------------------------------------

def run_vit(args):
    from models.vit_classifier import ViTClassifier
    from run_vit_experiments import load_cifar10_splits

    device = args.device
    results = []
    for seed in args.seeds:
        ckpt = os.path.join(SCRIPT_DIR, "outputs", "vit_checkpoints",
                            f"cifar10_intra_seed{seed}", "best_model")
        if not model_weights_exist(ckpt, "model"):
            logger.warning("SKIP seed %s: no checkpoint at %s", seed, ckpt)
            continue

        _, eval_ds, ood_ds, _, _ = load_cifar10_splits(seed, ood_class=0)
        eval_ds = _subsample(eval_ds, args.n_eval, seed)
        ood_ds = _subsample(ood_ds, args.n_eval, seed)

        model = ViTClassifier(num_labels=9)
        model.load_state_dict(load_model_weights(ckpt, "model"), strict=False)
        model.to(device)

        audit = audit_dropout(model)
        logger.info("seed %s ViT dropout audit: %s", seed, audit)

        ld = dict(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        eval_loader, ood_loader = DataLoader(eval_ds, **ld), DataLoader(ood_ds, **ld)

        for p in args.rates:
            n_set = set_dropout(model, p)
            t0 = time.time()
            v_id = mc_variance(model, eval_loader, _vit_forward, args.n_passes, device)
            v_ood = mc_variance(model, ood_loader, _vit_forward, args.n_passes, device)
            row = _row("ViT-B/16", "cifar10_intra", seed, p, n_set, v_id, v_ood,
                       time.time() - t0, audit)
            results.append(row)
            _log_row(row)
        del model
        torch.cuda.empty_cache()
    return results


def run_layoutlmv3(args):
    from datasets import load_from_disk
    from models.layoutlmv3_classifier import LayoutLMv3DocumentClassifier

    device = args.device
    results = []
    for seed in args.seeds:
        ckpt = os.path.join(SCRIPT_DIR, "outputs", "checkpoints",
                            f"tobacco_intra_seed{seed}", "best_model")
        if not model_weights_exist(ckpt, "model"):
            logger.warning("SKIP seed %s: no checkpoint at %s", seed, ckpt)
            continue

        meta_path = os.path.join(SCRIPT_DIR, "outputs", "checkpoints",
                                 f"tobacco_intra_seed{seed}", "training_metadata.json")
        num_labels = json.load(open(meta_path))["num_labels"]

        ds = load_from_disk(os.path.join(SCRIPT_DIR, "data", "tobacco", "colated_tensors"))
        eval_ds, ood_ds = _split_layoutlm(ds, meta_path, args.n_eval, seed)

        model = LayoutLMv3DocumentClassifier(num_labels=num_labels)
        model.load_state_dict(load_model_weights(ckpt, "model"), strict=False)
        model.to(device)

        audit = audit_dropout(model)
        logger.info("seed %s LayoutLMv3 dropout audit: %s", seed, audit)

        ld = dict(batch_size=args.batch_size_doc, shuffle=False, num_workers=2)
        eval_loader, ood_loader = DataLoader(eval_ds, **ld), DataLoader(ood_ds, **ld)

        for p in args.rates_doc:
            n_set = set_dropout(model, p)
            t0 = time.time()
            v_id = mc_variance(model, eval_loader, _layoutlm_forward, args.n_passes, device)
            v_ood = mc_variance(model, ood_loader, _layoutlm_forward, args.n_passes, device)
            row = _row("LayoutLMv3-base", "tobacco_intra", seed, p, n_set, v_id, v_ood,
                       time.time() - t0, audit)
            results.append(row)
            _log_row(row)
        del model
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample(ds, n, seed):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    return Subset(ds, idx.tolist())


def _split_layoutlm(ds, meta_path, n, seed):
    """Reproduce the tobacco_intra ID-eval / OOD split from the training metadata."""
    meta = json.load(open(meta_path))
    # `labels` holds class-name strings ("Advertisement", "Email", ...) while
    # label2id maps ID class *names* to ints, so match on keys, not values.
    id_names = set(meta["label2id"].keys())
    names = ds["label"] if "label" in ds.column_names else ds["labels"]
    id_idx = [i for i, l in enumerate(names) if l in id_names]
    ood_idx = [i for i, l in enumerate(names) if l not in id_names]
    if not id_idx or not ood_idx:
        raise RuntimeError(
            f"ID/OOD split is degenerate (id={len(id_idx)}, ood={len(ood_idx)}). "
            f"Label values look like {names[:3]!r}; expected members of {sorted(id_names)!r}."
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(id_idx)
    eval_idx = id_idx[int(0.8 * len(id_idx)):]
    return (Subset(ds, _cap(eval_idx, n, rng)), Subset(ds, _cap(ood_idx, n, rng)))


def _cap(idx, n, rng):
    if n is None or n >= len(idx):
        return list(idx)
    return list(rng.choice(idx, n, replace=False))


def _row(arch, config, seed, p, n_set, v_id, v_ood, secs, audit):
    # Paper convention: score = -variance, higher = more ID-like.
    auroc = float(compute_auroc(-v_id, -v_ood))
    return {
        "architecture": arch, "config": config, "seed": seed,
        "dropout_p": float(p), "n_dropout_modules_set": n_set,
        "var_id_mean": float(np.mean(v_id)), "var_id_median": float(np.median(v_id)),
        "var_ood_mean": float(np.mean(v_ood)), "var_ood_median": float(np.median(v_ood)),
        "var_id_max": float(np.max(v_id)), "var_ood_max": float(np.max(v_ood)),
        "frac_id_exactly_zero": float(np.mean(v_id == 0.0)),
        "auroc": auroc, "auroc_flipped": 1.0 - auroc,
        "n_id": int(len(v_id)), "n_ood": int(len(v_ood)),
        "elapsed_seconds": secs, "dropout_audit": audit,
    }


def _log_row(r):
    logger.info(
        "  %-16s seed=%-5s p=%-5.2f | var_ID=%.3e var_OOD=%.3e | AUROC=%.4f | %.0fs",
        r["architecture"], r["seed"], r["dropout_p"],
        r["var_id_mean"], r["var_ood_mean"], r["auroc"], r["elapsed_seconds"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vit", "layoutlmv3", "both"], default="both")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--rates_doc", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--n_passes", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--batch_size_doc", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "outputs",
                                                  "mcdropout_mechanism",
                                                  "mcdropout_mechanism.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    if args.arch in ("vit", "both"):
        logger.info("=" * 70)
        logger.info("ViT-B/16: dropout-rate intervention")
        results += run_vit(args)
        json.dump(results, open(args.out, "w"), indent=2)
    if args.arch in ("layoutlmv3", "both"):
        logger.info("=" * 70)
        logger.info("LayoutLMv3-base: dropout-rate intervention")
        results += run_layoutlmv3(args)

    json.dump(results, open(args.out, "w"), indent=2)
    logger.info("Wrote %d rows to %s", len(results), args.out)

    # Aggregate across seeds.
    logger.info("\n%-16s %-6s %-12s %-12s %-8s", "arch", "p", "var_ID", "var_OOD", "AUROC")
    keys = sorted({(r["architecture"], r["dropout_p"]) for r in results},
                  key=lambda k: (k[0], k[1]))
    summary = {}
    for arch, p in keys:
        rs = [r for r in results if r["architecture"] == arch and r["dropout_p"] == p]
        a = [r["auroc"] for r in rs]
        s = {"var_id_mean": float(np.mean([r["var_id_mean"] for r in rs])),
             "var_ood_mean": float(np.mean([r["var_ood_mean"] for r in rs])),
             "auroc_mean": float(np.mean(a)), "auroc_std": float(np.std(a)),
             "n_seeds": len(rs)}
        summary[f"{arch}|p={p}"] = s
        logger.info("%-16s %-6.2f %-12.3e %-12.3e %.4f+-%.4f (n=%d)",
                    arch, p, s["var_id_mean"], s["var_ood_mean"],
                    s["auroc_mean"], s["auroc_std"], s["n_seeds"])
    json.dump(summary, open(args.out.replace(".json", "_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
