"""validate_e4_fix.py — verification required by PROTOCOL Addendum C before E4 / E1-A continue.

1. EXACTNESS: the original E4 MC-Dropout loop (no cache release, copied verbatim from the code that produced seed 42)
   and the changed loop (release after every batch) are run with the same seed on the same documents (40 ID docs:
   batches of 16, 16 and 8, i.e. including a smaller last batch), 2 passes each. Every embedding must be bit-identical.
2. MEMORY: the changed loop runs 2 full passes over all 651 ID docs and 2 over all 230 OOD docs (E4 sizes, last batches
   11 and 6); GPU memory is recorded after every pass and must stay bounded (no growth).
Writes results/_validation/E4_fix/validation.json.  Run only while no other GPU job runs (queue on E4_HOLD).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import common as C
import e4_mcdropout_layoutlmv3 as E4
from gate0_parity import DocView, doc_split
from datasets import load_from_disk
from utils.data import create_dataloader

KEYS = E4.KEYS
s = 42


@torch.no_grad()
def mc_passes_original(model, loader, n):       # verbatim copy of the loop that produced seed 42 (6 Oct 23:09 – 7 Oct 02:20 UTC)
    model.train()
    passes = []
    for _ in range(n):
        embs = []
        for b in loader:
            out = model(**C.to_device(b, KEYS), output_hidden_states=True)
            embs.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
        passes.append(np.concatenate(embs).astype(np.float64))
    model.eval()
    return passes


ds = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
labels = list(ds["labels"])
ck = os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{s}")
meta = json.load(open(os.path.join(ck, "training_metadata.json")))
l2i, _, evr = doc_split(labels, sorted(meta["label2id"].keys()), s)
ood = [i for i, l in enumerate(labels) if l == "Advertisement"]
mk = lambda rows: create_dataloader(DocView(ds, rows, l2i), batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
model, _ = C.build_layoutlmv3(os.path.join(ck, "best_model"))
out = {"seed": s, "n_id": len(evr), "n_ood": len(ood)}

# 1. exactness
C.set_seed(s); a = mc_passes_original(model, mk(evr[:40]), 2)
C.release_device_cache()
C.set_seed(s); b = E4.mc_passes(model, mk(evr[:40]), 2, tag="validation")
ident = [bool(np.array_equal(x, y)) for x, y in zip(a, b)]
maxdiff = [float(np.abs(x - y).max()) for x, y in zip(a, b)]
distinct = bool(not np.array_equal(a[0], a[1]))          # dropout really active: two passes must differ
out["exactness"] = {"docs": 40, "batches": [16, 16, 8], "passes": 2, "bit_identical_per_pass": ident,
                    "max_abs_diff_per_pass": maxdiff, "passes_differ_from_each_other (dropout active)": distinct}
C.log(f"EXACTNESS: bit-identical {ident} max|diff| {maxdiff} | dropout active {distinct}")

# 2. memory over full E4-size passes
C.release_device_cache()
mem = {"start_GB": C.device_memory_bytes() / 2**30}
t0 = time.time()
for name, rows in (("ID_651", evr), ("OOD_230", ood)):
    for p in range(2):
        C.set_seed(s + p)
        E4.mc_passes(model, mk(rows), 1, tag=f"memory {name} pass {p + 1}")
        mem[f"{name}_pass{p + 1}_GB"] = C.device_memory_bytes() / 2**30
mem["seconds"] = time.time() - t0
mem["max_GB"] = max(v for k, v in mem.items() if k.endswith("_GB"))
out["memory"] = mem
C.log(f"MEMORY: {json.dumps({k: round(v, 2) for k, v in mem.items()})}")

out["PASS"] = bool(all(ident) and distinct and mem["max_GB"] < 4.0)
os.makedirs(os.path.join(C.RESULTS, "_validation", "E4_fix"), exist_ok=True)
C.save_json(out, os.path.join(C.RESULTS, "_validation", "E4_fix", "validation.json"))
print(json.dumps(out, indent=1))
