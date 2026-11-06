"""
E4 — LayoutLMv3 MC-Dropout with STRICT loading of the fine-tuned weights (PROTOCOL.md §E4).
run_mcdropout_mechanism.py loads LayoutLMv3 with strict=False, and the HF-style checkpoint keys do not
match its wrapper, so no fine-tuned weights are loaded there; this script loads them strictly.

  python e4_mcdropout_layoutlmv3.py <seed> [seed...]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import common as C
from gate0_parity import DocView, doc_split
from datasets import load_from_disk
from utils.data import create_dataloader
from ood_methods.sensitivity_based import MCDropoutScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr

KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")
SMOKE = os.environ.get("SMOKE") == "1"
N_PASSES = 2 if SMOKE else 25
OUTD = os.path.join(C.RESULTS, ("SMOKE_" if SMOKE else "") + "E4_mcdropout_layoutlmv3")


@torch.no_grad()
def mc_passes(model, loader, n, tag=""):
    model.train()                                    # dropout active at the checkpoint's own rate (0.1)
    passes = []
    for i in range(n):
        embs = []
        for b in loader:
            out = model(**C.to_device(b, KEYS), output_hidden_states=True)
            embs.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
            del out
            C.release_device_cache()                    # PROTOCOL Addendum C: memory only, no change to the computation
        passes.append(np.concatenate(embs).astype(np.float64))
        if (i + 1) % 5 == 0 or i == 0:
            C.log(f"    {tag} pass {i + 1}/{n} done | GPU memory {C.device_memory_bytes() / 2**30:.2f} GB")
    model.eval()
    return passes


def main(seeds):
    os.makedirs(OUTD, exist_ok=True)
    ds = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
    labels = list(ds["labels"])
    for s in seeds:
        path = os.path.join(OUTD, f"seed{s}.json")
        if os.path.exists(path):
            C.log(f"skip (done) {path}"); continue
        t0 = time.time(); C.set_seed(s)
        ck = os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{s}")
        meta = json.load(open(os.path.join(ck, "training_metadata.json")))
        model, info = C.build_layoutlmv3(os.path.join(ck, "best_model"))
        dropouts = sorted({float(m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)})
        l2i, _, evr = doc_split(labels, sorted(meta["label2id"].keys()), s)
        ood = [i for i, l in enumerate(labels) if l == "Advertisement"]
        if SMOKE:
            evr, ood = evr[:32], ood[:32]
        mk = lambda rows: create_dataloader(DocView(ds, rows, l2i), batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
        # sanity: deterministic eval-mode accuracy equals the logged value (proves the fine-tuned weights are in use)
        f = C.run_features(model, mk(evr), KEYS, want=("logits",), release_cache=True)
        acc = float((f["logits"].argmax(1) == f["labels"]).mean())
        p_id = mc_passes(model, mk(evr), N_PASSES, tag=f"seed {s} ID")
        p_ood = mc_passes(model, mk(ood), N_PASSES, tag=f"seed {s} OOD")
        sc = MCDropoutScorer(normalize=True)
        u_id = sc.compute_mc_uncertainty(p_id, normalize=True)
        u_ood = sc.compute_mc_uncertainty(p_ood, normalize=True)
        paper = json.load(open(os.path.join(C.OUT, "ahss_results", f"tobacco_intra_seed{s}", "phase1_results.json")))["mc_dropout"]["auroc"]
        au = compute_auroc(-u_id, -u_ood)
        r = {"seed": s, "n_id": len(evr), "n_ood": len(ood), "n_passes": N_PASSES, "dropout_rates_in_model": dropouts,
             "eval_acc_eval_mode": acc, "eval_acc_logged": meta["eval_accuracy"],
             "WEIGHTS_GATE_PASS": bool(abs(acc - float(meta["eval_accuracy"])) <= 1.0 / len(evr) + 1e-12),
             "var_id_mean": float(u_id.mean()), "var_ood_mean": float(u_ood.mean()),
             "var_id_median": float(np.median(u_id)), "var_ood_median": float(np.median(u_ood)),
             "auroc": float(au), "fpr95": float(compute_fpr_at_tpr(-u_id, -u_ood)),
             "paper_auroc_this_seed": paper, "abs_diff_vs_paper": abs(au - paper),
             "REPRO_GATE_PASS": bool(abs(au - paper) <= 0.02),
             "load": {k: v for k, v in info.items() if k != "label2id"}, "seconds": time.time() - t0}
        C.save_json(r, path)
        C.log(f"E4 seed {s}: acc {acc:.4f} (logged {meta['eval_accuracy']:.4f}) | var ID {r['var_id_mean']:.3e} OOD {r['var_ood_mean']:.3e} | "
              f"AUROC {au:.4f} (paper {paper:.4f}) gates W={r['WEIGHTS_GATE_PASS']} R={r['REPRO_GATE_PASS']} ({r['seconds']/60:.1f} min)")
        del model; C.empty_device_cache()


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
