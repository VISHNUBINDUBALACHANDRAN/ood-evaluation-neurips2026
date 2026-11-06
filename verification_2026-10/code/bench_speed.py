"""Steady-state throughput on the GPU, and a check that returning attention maps
(output_attentions=True, as in the original wrappers) does not change hidden states
in eager mode — if bit-identical, production runs may skip returning them."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
import common as C


@torch.no_grad()
def timed(model, batches, attn):
    outs = []
    C.device_sync(); t0 = time.time()
    for b in batches:
        o = model(**b, output_hidden_states=True, output_attentions=attn)
        outs.append(torch.stack([h[:, 0, :] for h in o.hidden_states], 0).cpu())
    C.device_sync()
    return time.time() - t0, torch.cat(outs, 1)


def bench(name, model, batches, n):
    timed(model, batches[:2], True)                       # warm-up
    t1, h1 = timed(model, batches, True)
    t0, h0 = timed(model, batches, False)
    return {"name": name, "samples": n, "per_s_attn_true": n / t1, "per_s_attn_false": n / t0,
            "hidden_identical_bitwise": bool(torch.equal(h1, h0)),
            "max_abs_diff": float((h1 - h0).abs().max())}


res = []
# RoBERTa base / large
from transformers import RobertaTokenizer
from run_roberta_experiments import load_sst2_data
tok = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=os.path.join(C.PROJECT_DIR, "data", "hf_cache"))
_, ev = load_sst2_data(42, tok)
for size, ck, bs in (("base", "sst2_seed42", 128), ("large", "sst2_robertalarge_seed42", 64)):
    m, _ = C.build_roberta(os.path.join(C.OUT, "roberta_checkpoints", ck, "best_model"), size=size)
    dl = DataLoader(Subset(ev, range(bs * 6)), batch_size=bs)
    batches = [C.to_device(b, ("input_ids", "attention_mask")) for b in dl]
    res.append(bench(f"roberta-{size} bs{bs}", m, batches, bs * 6)); C.log(res[-1]); del m
# ViT
from run_vit_experiments import load_cifar10_splits
_, ev, _, _, _ = load_cifar10_splits(42, ood_class=0)
m, _ = C.build_vit(os.path.join(C.OUT, "vit_checkpoints", "cifar10_intra_seed42", "best_model"), 9)
dl = DataLoader(Subset(ev, range(128 * 4)), batch_size=128)
batches = [C.to_device(b, ("pixel_values",)) for b in dl]
res.append(bench("vit bs128", m, batches, 512)); C.log(res[-1]); del m
# LayoutLMv3
from datasets import load_from_disk
from gate0_parity import DocView
from utils.data import create_dataloader
ds = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
m, _ = C.build_layoutlmv3(os.path.join(C.OUT, "checkpoints", "tobacco_intra_seed42", "best_model"))
dl = create_dataloader(DocView(ds, range(64), {}), batch_size=16, num_workers=0, pin_memory=False)
batches = [C.to_device(b, ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")) for b in dl]
res.append(bench("layoutlmv3 bs16", m, batches, 64)); C.log(res[-1])
C.save_json(res, os.path.join(C.RESULTS, "bench_speed.json"))
