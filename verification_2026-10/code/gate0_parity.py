"""
GATE 0 — re-run vs original parity.

Before any new experiment is trusted, the re-run pipeline must reproduce numbers the
original runs saved to disk, using the SAME checkpoints, the SAME data splits and the
SAME code. Three architectures are checked:

  G0a LayoutLMv3 tobacco_intra seed 42 : rebuild the 651-doc eval split exactly as
       run_finetune.py did; compare CLS embeddings + logits with the original files
       outputs/embeddings/tobacco_intra_seed42/{eval_embeddings,eval_logits}.npy and
       the logged eval accuracy.
  G0b RoBERTa-base sst2 seed 42       : compare per-layer CLS of the first 512 eval
       sentences with outputs/roberta_results/sst2_20news_seed42/layer_embeddings.
  G0c ViT-B/16 cifar10_intra seed 42  : same with outputs/xmaha_results/
       vit_cifar10_intra_seed42/layer_embeddings.

PASS criterion: per-sample cosine similarity >= 0.9999 for every compared vector,
identical predicted classes, and accuracy equal to the logged value.
Throughput (samples/s) is recorded for scheduling.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset, Dataset
import common as C

N_CMP = 512
report = {"env": C.env_info(), "checks": {}}


def cos_stats(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    cs = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return {"cos_min": float(cs.min()), "cos_mean": float(cs.mean()),
            "max_abs_diff": float(np.abs(a - b).max()), "max_abs_ref": float(np.abs(b).max())}


# ---------------------------------------------------------------- LayoutLMv3
class DocView(Dataset):
    """Rows of the HF Tobacco dataset with string labels mapped to ints (-1 = not in map)."""
    def __init__(self, ds, idx, label2id):
        self.ds, self.idx, self.l2i = ds, list(idx), label2id
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        r = self.ds[int(self.idx[k])]
        out = {c: r[c] for c in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")}
        out["labels"] = self.l2i.get(r["labels"], -1)
        return out


def doc_split(ds_labels, train_classes, seed):
    """Exact replica of run_finetune.py: filter (order kept) -> encode with sorted
    label2id -> sklearn train_test_split(train_size=0.8, stratify, random_state=seed)."""
    from sklearn.model_selection import train_test_split
    from configs.config import get_label2id
    l2i = get_label2id(train_classes)
    id_rows = [i for i, l in enumerate(ds_labels) if l in train_classes]
    y = np.array([l2i[ds_labels[i]] for i in id_rows])
    tr, ev = train_test_split(np.arange(len(id_rows)), train_size=0.8, random_state=seed, stratify=y)
    return l2i, [id_rows[i] for i in tr], [id_rows[i] for i in ev]


def gate_layoutlmv3():
    from datasets import load_from_disk
    from utils.data import create_dataloader
    seed = 42
    ck = os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{seed}", "best_model")
    model, info = C.build_layoutlmv3(ck)
    meta = json.load(open(os.path.join(C.OUT, "checkpoints", f"tobacco_intra_seed{seed}", "training_metadata.json")))
    ds = load_from_disk(os.path.join(C.PROJECT_DIR, "data", "tobacco", "colated_tensors"))
    labels = list(ds["labels"])
    train_classes = sorted(meta["label2id"].keys())
    l2i, tr, ev = doc_split(labels, train_classes, seed)
    assert l2i == {k: int(v) for k, v in meta["label2id"].items()}, "label2id mismatch"
    dl = create_dataloader(DocView(ds, ev, l2i), batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    keys = ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")
    t0 = time.time()
    f = C.run_features(model, dl, keys, want=("cls_last", "logits"), output_attentions=True)
    C.device_sync(); dt = time.time() - t0
    ref_e = np.load(os.path.join(C.OUT, "embeddings", f"tobacco_intra_seed{seed}", "eval_embeddings.npy"))
    ref_l = np.load(os.path.join(C.OUT, "embeddings", f"tobacco_intra_seed{seed}", "eval_logits.npy"))
    ref_y = np.load(os.path.join(C.OUT, "embeddings", f"tobacco_intra_seed{seed}", "eval_labels.npy"))
    acc = float((f["logits"].argmax(1) == f["labels"]).mean())
    acc_orig = float((ref_l.argmax(1) == ref_y).mean())
    res = {"n_train": len(tr), "n_eval": len(ev), "load": {k: v for k, v in info.items() if k != "label2id"},
           "labels_identical": bool(np.array_equal(f["labels"], ref_y)),
           "cls": cos_stats(f["cls_last"], ref_e), "logits": cos_stats(f["logits"], ref_l),
           "pred_agree": float((f["logits"].argmax(1) == ref_l.argmax(1)).mean()),
           "acc_rerun": acc, "acc_original_fp32_extraction": acc_orig,
           "acc_logged_by_trainer_fp16": meta.get("eval_accuracy"),
           "throughput_per_s": len(ev) / dt}
    # strict: same split (labels identical in order), same vectors, same predictions as the original fp32 run
    res["PASS"] = bool(len(ev) == 651 and res["labels_identical"] and res["cls"]["cos_min"] >= 0.9999
                       and res["pred_agree"] == 1.0 and acc == acc_orig)
    return res


# ---------------------------------------------------------------- RoBERTa
def gate_roberta():
    from transformers import RobertaTokenizer
    from run_roberta_experiments import load_sst2_data
    seed = 42
    tok = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=os.path.join(C.PROJECT_DIR, "data", "hf_cache"))
    _, eval_ds = load_sst2_data(seed, tok)
    model, info = C.build_roberta(os.path.join(C.OUT, "roberta_checkpoints", f"sst2_seed{seed}", "best_model"))
    dl = DataLoader(Subset(eval_ds, range(N_CMP)), batch_size=64, shuffle=False)
    t0 = time.time()
    f = C.run_features(model, dl, ("input_ids", "attention_mask"), want=("cls_layers",), output_attentions=True)
    C.device_sync(); dt = time.time() - t0
    ref = np.load(os.path.join(C.OUT, "roberta_results", f"sst2_20news_seed{seed}", "layer_embeddings",
                               "eval_layer_embeddings.npy"), mmap_mode="r")[:, :N_CMP]
    mine = f["cls_layers"]                     # (13, N, D) incl. embedding output
    # identify which 12 hidden states the original file holds
    best = max(((s, cos_stats(mine[s:s + 12], ref)["cos_min"]) for s in (0, 1)), key=lambda t: t[1])
    st = cos_stats(mine[best[0]:best[0] + 12], ref)
    res = {"load": info, "n_eval_total": len(eval_ds), "ref_layers_offset": best[0], "cls_layers": st,
           "throughput_per_s": N_CMP / dt}
    res["PASS"] = bool(len(eval_ds) == 6735 and st["cos_min"] >= 0.9999)
    return res


# ---------------------------------------------------------------- ViT
def gate_vit():
    from run_vit_experiments import load_cifar10_splits
    seed = 42
    train_ds, eval_ds, ood_ds, _, _ = load_cifar10_splits(seed, ood_class=0)
    model, info = C.build_vit(os.path.join(C.OUT, "vit_checkpoints", f"cifar10_intra_seed{seed}", "best_model"), 9)
    dl = DataLoader(Subset(eval_ds, range(N_CMP)), batch_size=64, shuffle=False)
    t0 = time.time()
    f = C.run_features(model, dl, ("pixel_values",), want=("cls_layers",), output_attentions=True)
    C.device_sync(); dt = time.time() - t0
    ref = np.load(os.path.join(C.OUT, "xmaha_results", f"vit_cifar10_intra_seed{seed}", "layer_embeddings",
                               "eval_layer_embeddings.npy"), mmap_mode="r")[:, :N_CMP]
    mine = f["cls_layers"]
    best = max(((s, cos_stats(mine[s:s + 12], ref)["cos_min"]) for s in (0, 1)), key=lambda t: t[1])
    st = cos_stats(mine[best[0]:best[0] + 12], ref)
    res = {"load": info, "sizes": [len(train_ds), len(eval_ds), len(ood_ds)], "ref_layers_offset": best[0],
           "cls_layers": st, "throughput_per_s": N_CMP / dt}
    res["PASS"] = bool([len(train_ds), len(eval_ds), len(ood_ds)] == [40500, 4500, 1000] and st["cos_min"] >= 0.9999)
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["layoutlmv3", "roberta", "vit"]
    for name in which:
        C.log(f"GATE {name} ...")
        try:
            r = {"layoutlmv3": gate_layoutlmv3, "roberta": gate_roberta, "vit": gate_vit}[name]()
        except Exception as e:
            import traceback; r = {"PASS": False, "error": repr(e), "trace": traceback.format_exc()}
        report["checks"][name] = r
        C.log(f"GATE {name}: PASS={r.get('PASS')}  {json.dumps({k: v for k, v in r.items() if k not in ('load', 'trace')})[:900]}")
        C.save_json(report, os.path.join(C.RESULTS, "gate0_parity.json"))
    report["ALL_PASS"] = all(v.get("PASS") for v in report["checks"].values())
    C.save_json(report, os.path.join(C.RESULTS, "gate0_parity.json"))
    C.log(f"ALL_PASS={report['ALL_PASS']}")
