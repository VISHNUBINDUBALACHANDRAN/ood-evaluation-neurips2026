"""
common.py — shared utilities for the verification experiments.

Design rules (to keep these results comparable with the paper's original runs):
  * Data loading, splits, scorers and metrics are IMPORTED UNCHANGED from the
    original project (PROJECT_DIR); nothing here re-implements them.
  * Models are rebuilt from local config files and their fine-tuned weights are
    loaded with strict=True (a silent partial load is impossible).
  * The attention-head-masking hook is a verbatim copy of the original hook
    (zero the per-head slice of the self-attention output).
  * Forward passes use output_hidden_states=True and output_attentions=True,
    exactly as the original wrappers do (this also fixes the eager attention path).
"""
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

FINAL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))                 # verification_2026-10/
ROOT = os.path.dirname(FINAL)                                                           # repository root
PROJECT_DIR = os.environ.get("PROJECT_DIR", ROOT)
OUT = os.path.join(PROJECT_DIR, "outputs")                                              # original artefacts (read-only)
CPMOD = os.environ.get("LEAVE_CLASS_OUT_DIR", OUT)                                      # holds checkpoints/leaveout_<class>_seed<s>
RESULTS = os.path.join(FINAL, "results")
ASSETS = os.path.join(FINAL, "assets")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DEVICE = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
_acc = lambda: getattr(torch, DEVICE.type)   # backend module of the accelerator in use (e.g. torch.cuda)
SEEDS = [42, 123, 456, 789, 1024]


# ----------------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def env_info():
    import transformers, sklearn, scipy
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "device": str(DEVICE)}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# strict weight loading
# ----------------------------------------------------------------------------
def _read_state_dict(ckpt_dir):
    st = os.path.join(ckpt_dir, "model.safetensors")
    bn = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st, device="cpu"), st
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True), bn
    raise FileNotFoundError(f"no weights in {ckpt_dir}")


def load_strict(module, state_dict, strip_prefix=None, allowed_missing=("position_ids",)):
    """Load with strict checking. Only non-persistent buffer names listed in
    allowed_missing may be absent; any other missing/unexpected key is an error."""
    if strip_prefix:
        state_dict = {k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v
                      for k, v in state_dict.items()}
    res = module.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in res.missing_keys if not any(a in k for a in allowed_missing)]
    # A saved position_ids buffer (older transformers persisted it) is accepted ONLY if it is
    # identical to the model's own fixed buffer; any other unexpected key is an error.
    bad_unexpected, ok_unexpected = [], []
    for k in res.unexpected_keys:
        if k.endswith("position_ids"):
            try:
                own = module.get_buffer(k)
            except Exception:
                own = None
            if own is not None and own.shape == state_dict[k].shape and torch.equal(own.cpu(), state_dict[k].cpu()):
                ok_unexpected.append(k); continue
        bad_unexpected.append(k)
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"STRICT LOAD FAILED: missing={bad_missing[:8]} (n={len(bad_missing)}) "
                           f"unexpected={bad_unexpected[:8]} (n={len(bad_unexpected)})")
    n_loaded = len(state_dict) - len(res.unexpected_keys)
    return {"n_tensors_loaded": n_loaded, "missing_ok": list(res.missing_keys),
            "unexpected_ok_identical_buffers": ok_unexpected}


# ----------------------------------------------------------------------------
# models (rebuilt from local configs; eager attention as in the original forward)
# ----------------------------------------------------------------------------
def build_layoutlmv3(ckpt_dir):
    from transformers import LayoutLMv3Config, LayoutLMv3ForSequenceClassification
    cfg = LayoutLMv3Config.from_pretrained(ckpt_dir, output_hidden_states=True, output_attentions=True)
    model = LayoutLMv3ForSequenceClassification._from_config(cfg, attn_implementation="eager")
    sd, path = _read_state_dict(ckpt_dir)
    info = load_strict(model, sd)
    info.update({"weights": path, "weights_sha256": sha256_file(path), "num_labels": cfg.num_labels,
                 "label2id": cfg.label2id})
    return model.to(DEVICE).eval(), info


def build_roberta(ckpt_dir, size="base", num_labels=2):
    from transformers import RobertaConfig, RobertaForSequenceClassification
    snap = os.path.join(PROJECT_DIR, "data", "hf_cache", f"models--roberta-{size}", "snapshots")
    snap = os.path.join(snap, sorted(os.listdir(snap))[0])
    cfg = RobertaConfig.from_pretrained(snap, num_labels=num_labels,
                                        output_hidden_states=True, output_attentions=True)
    model = RobertaForSequenceClassification._from_config(cfg, attn_implementation="eager")
    sd, path = _read_state_dict(ckpt_dir)
    info = load_strict(model, sd, strip_prefix="model.")
    info.update({"weights": path, "weights_sha256": sha256_file(path), "config": snap})
    return model.to(DEVICE).eval(), info


def build_vit(ckpt_dir, num_labels):
    from transformers import ViTConfig, ViTForImageClassification
    cfg = ViTConfig.from_pretrained(os.path.join(ASSETS, "vit-base-patch16-224"), num_labels=num_labels,
                                    output_hidden_states=True, output_attentions=True)
    cfg.id2label = {i: str(i) for i in range(num_labels)}; cfg.label2id = {str(i): i for i in range(num_labels)}
    model = ViTForImageClassification._from_config(cfg, attn_implementation="eager")
    sd, path = _read_state_dict(ckpt_dir)
    info = load_strict(model, sd, strip_prefix="model.")
    info.update({"weights": path, "weights_sha256": sha256_file(path)})
    return model.to(DEVICE).eval(), info


def self_attention_modules(model):
    """Ordered list of the per-layer self-attention modules (the hook targets)."""
    if hasattr(model, "layoutlmv3"):
        layers = model.layoutlmv3.encoder.layer; return [l.attention.self for l in layers]
    if hasattr(model, "roberta"):
        layers = model.roberta.encoder.layer; return [l.attention.self for l in layers]
    if hasattr(model, "vit"):
        layers = model.vit.encoder.layer; return [l.attention.attention for l in layers]
    raise ValueError("unknown architecture")


class HeadMasker:
    """Verbatim re-implementation of the original forward-hook head masking."""

    def __init__(self, model):
        self.mods = self_attention_modules(model)
        self.handles = []

    def set(self, mask):
        self.clear()
        if mask is None:
            return
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
        assert mask.shape == (len(self.mods), self.mods[0].num_attention_heads), mask.shape
        for li, mod in enumerate(self.mods):
            nh, hs = mod.num_attention_heads, mod.attention_head_size

            def make_hook(lm, nh=nh, hs=hs):
                def hook_fn(module, inp, output):
                    ctx = output[0]
                    b, s, _ = ctx.shape
                    ctx = (ctx.view(b, s, nh, hs) * lm.to(ctx.device).view(1, 1, nh, 1)).view(b, s, nh * hs)
                    return (ctx,) + tuple(output[1:])
                return hook_fn
            self.handles.append(mod.register_forward_hook(make_hook(mask[li])))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------
def global_random_masks(n, n_layers, n_heads, pcts=(0.1, 0.2), seed=42):
    """Original cross-architecture generator (run_roberta_experiments / crossarch):
    mask int(L*H*pct) heads chosen uniformly over the whole L x H grid."""
    rng = np.random.RandomState(seed); out = []
    for i in range(n):
        pct = pcts[i % len(pcts)]
        m = np.ones(n_layers * n_heads, dtype=np.float32)
        m[rng.choice(n_layers * n_heads, size=int(n_layers * n_heads * pct), replace=False)] = 0.0
        out.append(m.reshape(n_layers, n_heads))
    return out


def per_layer_random_masks(n, n_layers, n_heads, pcts=(0.1, 0.2), seed=42):
    """Original AHM generator (models/attention_masking.AttentionHeadMasker.create_mask):
    in every layer mask max(1, int(H*pct)) heads. Budget split as in AHMOptimizer:
    base = n // len(pcts) masks per pct (remainder to the first pcts). Seeded here
    (the original optimizer was unseeded; seeding makes the selection reproducible)."""
    from models.attention_masking import AttentionHeadMasker
    masker = AttentionHeadMasker(n_layers, n_heads)
    rng = np.random.RandomState(seed); out = []
    base, rem = divmod(n, len(pcts))
    for pi, p in enumerate(pcts):
        for _ in range(base + (1 if pi < rem else 0)):
            out.append(masker.create_mask(p, seed=int(rng.randint(0, 2**31))).numpy())
    return out


# ----------------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------------
def to_device(batch, keys):
    return {k: batch[k].to(DEVICE) for k in keys if k in batch}


def release_device_cache():
    """Return cached-but-unused GPU memory to the system (PROTOCOL Addendum C). Changes no computation."""
    if DEVICE.type != "cpu":
        _acc().synchronize(); gc.collect(); _acc().empty_cache()


def empty_device_cache():
    if DEVICE.type != "cpu":
        _acc().empty_cache()


def device_memory_bytes():
    """Memory held by the accelerator (driver level where the backend reports it)."""
    m = _acc() if DEVICE.type != "cpu" else None
    for fn in ("driver_allocated_memory", "memory_reserved", "memory_allocated"):
        if m is not None and hasattr(m, fn):
            return getattr(m, fn)()
    return 0


@torch.no_grad()
def run_features(model, loader, input_keys, want, masker=None, mask=None, desc="", output_attentions=False,
                 release_cache=False):
    """One pass over `loader`; returns dict of numpy arrays for the requested features:
      cls_last   : CLS of the last hidden layer                        (N, D)
      cls_layers : CLS of every hidden state incl. embeddings           (L+1, N, D)
      avg_pad    : AvgAvg, sequence mean INCLUDING padding, layers 1..L (N, D)   [original crossarch]
      avg_nopad  : AvgAvg, sequence mean over attention_mask==1 only    (N, D)   [corrected]
      logits     : classifier logits                                    (N, C)
    release_cache=True releases the GPU cache after every batch (LayoutLMv3 steps E4 / E1-A, PROTOCOL Addendum C);
    the default False keeps the exact code path used for all results produced before 2026-10-07 08:50 UTC.
    """
    if masker is not None:
        masker.set(mask)
    acc = {k: [] for k in want}
    labels = []
    for batch in loader:
        inp = to_device(batch, input_keys)
        out = model(**inp, output_hidden_states=True, output_attentions=output_attentions)
        hs = out.hidden_states
        if "cls_last" in want:
            acc["cls_last"].append(hs[-1][:, 0, :].float().cpu())
        if "cls_layers" in want:
            acc["cls_layers"].append(torch.stack([h[:, 0, :] for h in hs], 0).float().cpu())
        if "cls_avg" in want:   # uniform mean of CLS over transformer layers 1..L (X-Maha-uniform input)
            acc["cls_avg"].append(torch.stack([h[:, 0, :] for h in hs[1:]], 0).mean(0).float().cpu())
        if "avg_pad" in want:
            acc["avg_pad"].append(torch.stack([h.mean(dim=1) for h in hs[1:]], 0).mean(0).float().cpu())
        if "avg_nopad" in want:
            am = inp["attention_mask"].to(hs[0].dtype).unsqueeze(-1)       # (B, S, 1)
            den = am.sum(dim=1).clamp(min=1.0)
            acc["avg_nopad"].append(torch.stack([(h[:, :am.shape[1], :] * am).sum(1) / den for h in hs[1:]], 0)
                                    .mean(0).float().cpu())
        if "logits" in want:
            acc["logits"].append(out.logits.float().cpu())
        if "labels" in batch:
            labels.append(batch["labels"].cpu())
        if release_cache:
            del out, hs
            release_device_cache()
    if masker is not None:
        masker.clear()
    res = {k: torch.cat(v, dim=1 if k == "cls_layers" else 0).numpy().astype(np.float64) for k, v in acc.items()}
    if labels:
        res["labels"] = torch.cat(labels).numpy()
    return res


def device_sync():
    if DEVICE.type != "cpu":
        _acc().synchronize()
