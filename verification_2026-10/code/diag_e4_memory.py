"""diag_e4_memory.py — why did E4 (MC-Dropout, LayoutLMv3 on the GPU) grow to 23 GB? (run only when no GPU job is running)

Same model, same documents, same batch size (16) as E4. For each mode the GPU memory held by the process
(C.device_memory_bytes) is recorded after every batch:
  A  eval mode                      (as E1-C: no growth expected)
  B  train mode (dropout on)        (as E4)
  C  train mode + C.empty_device_cache() after every batch
  D  train mode, only nn.Dropout modules in train mode (equivalent computation for LayoutLMv3: it has no BatchNorm)
     + empty_cache after every batch
Prints MB per batch and the growth per batch for each mode.
  python diag_e4_memory.py [n_batches=12]
"""
import gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import common as C
from gate0_parity import DocView, doc_split
from datasets import load_from_disk
from utils.data import create_dataloader

KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 12
s = 42
ds = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
labels = list(ds["labels"])
ck = os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{s}")
meta = json.load(open(os.path.join(ck, "training_metadata.json")))
l2i, _, evr = doc_split(labels, sorted(meta["label2id"].keys()), s)
dl = create_dataloader(DocView(ds, evr[:16 * NB], l2i), batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
batches = [C.to_device(b, KEYS) for b in dl]           # pre-loaded once so data loading is not measured
model, _ = C.build_layoutlmv3(os.path.join(ck, "best_model"))
assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()), "BatchNorm present"
MB = lambda: C.device_memory_bytes() / 2**20


def mode(name, setup, clear):
    setup()
    C.release_device_cache()
    m0, mem, t = MB(), [], time.time()
    with torch.no_grad():
        for b in batches:
            out = model(**b, output_hidden_states=True)
            _ = out.hidden_states[-1][:, 0, :].float().cpu().numpy()
            del out
            if clear:
                C.release_device_cache()
            mem.append(MB())
    grow = (mem[-1] - mem[len(mem) // 2]) / max(1, len(mem) - 1 - len(mem) // 2)
    print(f"{name:52s} start {m0:7.0f} MB | after batches: {' '.join(f'{x:.0f}' for x in mem)} | "
          f"growth {grow:+.0f} MB/batch (2nd half) | {time.time() - t:.1f}s", flush=True)


def dropout_only():
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


mode("A eval mode", model.eval, False)
mode("B train mode (as E4)", model.train, False)
mode("C train mode + empty_cache each batch", model.train, True)
mode("D dropout-only train + empty_cache each batch", dropout_only, True)
model.eval()
