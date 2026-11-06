"""train_leave_class_out.py — fine-tune LayoutLMv3 on Tobacco3482 with one class held out of training.

The held-out class is never seen during fine-tuning and is used as near-OOD at evaluation time (document near-OOD
results; the resulting splits are listed in split_configs/tobacco3482_leave_class_out.json).

Recipe = run_finetune.py with configs/config.py: learning rate 5e-5, 65 epochs, batch 32 x gradient accumulation 8,
warm-up ratio 0.1, linear schedule, fp16, evaluation every epoch from epoch 5, best checkpoint by eval accuracy;
the 9 remaining classes are split 80/20, stratified, with random_state = seed.
The released leave-class-out checkpoints were trained with transformers 4.28.0 (default optimizer adamw_hf);
use that version to reproduce them exactly.

  python train_leave_class_out.py --holdout_class Memo --seed 42 --data <processed Tobacco3482 folder> \
      [--out_dir outputs/checkpoints] [--dry_run]

<processed Tobacco3482 folder> is the colated_tensors folder written by run_prepare_data.py.
Writes <out_dir>/leaveout_<class>_seed<seed>/best_model and training_metadata.json.
--dry_run builds the split, prints its sizes and exits without training.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_processed_tobacco(folder):
    """Load the processed Tobacco3482 dataset (shards in name order), mapping the feature type 'List' written by
    older `datasets` versions to 'Sequence' so that current versions can read it."""
    import pyarrow as pa
    from datasets import Dataset
    shards = sorted(glob.glob(os.path.join(folder, "data-*-of-*.arrow")))
    if not shards:
        raise FileNotFoundError(f"no dataset shards in {folder}")

    def _open(p):
        try:
            return pa.ipc.open_stream(pa.memory_map(p, "r")).read_all()
        except Exception:
            return pa.ipc.open_file(pa.memory_map(p, "r")).read_all()

    t = pa.concat_tables([_open(s) for s in shards])
    md = dict(t.schema.metadata or {})
    if b"huggingface" in md:
        md[b"huggingface"] = md[b"huggingface"].decode().replace('"List"', '"Sequence"').encode()
        t = t.replace_schema_metadata(md)
    return Dataset(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_class", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", required=True, help="processed Tobacco3482 folder (colated_tensors)")
    ap.add_argument("--out_dir", default=os.path.join("outputs", "checkpoints"))
    ap.add_argument("--epochs", type=int, default=None, help="override (debug only)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Trainer imports apex when it detects it; disable detection so that fp16 uses native torch AMP.
    import transformers.utils.import_utils as _iu
    _iu.is_apex_available = lambda: False
    try:
        import transformers.utils as _u
        _u.is_apex_available = lambda: False
    except Exception:
        pass

    from transformers import LayoutLMv3ForSequenceClassification, TrainingArguments, Trainer
    from configs.config import get_default_config, get_label2id, get_id2label
    from utils.data import encode_labels, filter_dataset_by_classes, data_collator

    conf = get_default_config()
    all_classes = conf.tobacco.all_classes
    holdout = args.holdout_class
    assert holdout in all_classes, f"{holdout} not in {all_classes}"
    train_classes = [c for c in all_classes if c != holdout]   # 9 classes
    label2id = get_label2id(train_classes)
    id2label = get_id2label(train_classes)
    num_labels = len(train_classes)

    out_root = Path(args.out_dir) / f"leaveout_{holdout}_seed{args.seed}"
    best_dir = out_root / "best_model"
    if (best_dir / "config.json").exists() and not args.dry_run:
        print(f"[skip] checkpoint already exists: {best_dir}")
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # data: all classes except the held-out one; encode; stratified 80/20
    ds = load_processed_tobacco(args.data)
    ds = filter_dataset_by_classes(ds, train_classes)
    ds = encode_labels(ds, label2id)
    cols = [c for c in ["input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels"]
            if c in ds.column_names]
    ds = ds.with_format("torch", columns=cols)          # data_collator stacks these columns as tensors
    from sklearn.model_selection import train_test_split
    labels = np.array(ds["labels"])
    tr_idx, ev_idx = train_test_split(np.arange(len(ds)), train_size=conf.train_eval_split,
                                      random_state=args.seed, stratify=labels)
    train_ds, eval_ds = ds.select(tr_idx), ds.select(ev_idx)
    print(f"[leaveout {holdout} seed {args.seed}] train={len(train_ds)} eval={len(eval_ds)} classes={num_labels}")
    if args.dry_run:
        return
    out_root.mkdir(parents=True, exist_ok=True)

    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        conf.model.model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)

    def compute_metrics(p):
        return {"accuracy": float((np.argmax(p[0], -1) == p[1]).mean())}

    t = conf.training
    # transformers 4.28 names the evaluation schedule `evaluation_strategy`; later versions call it `eval_strategy`
    eval_key = ("evaluation_strategy" if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                else "eval_strategy")
    targs = TrainingArguments(
        output_dir=str(out_root),
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate, weight_decay=t.weight_decay,
        adam_beta1=t.adam_beta1, adam_beta2=t.adam_beta2, adam_epsilon=t.adam_epsilon,
        max_grad_norm=t.max_grad_norm,
        num_train_epochs=args.epochs or t.num_train_epochs,
        warmup_ratio=t.warmup_ratio, lr_scheduler_type=t.lr_scheduler_type,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy, save_total_limit=t.save_total_limit,
        eval_delay=t.eval_delay,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model, greater_is_better=t.greater_is_better,
        fp16=t.fp16, dataloader_num_workers=t.dataloader_num_workers,
        seed=args.seed, remove_unused_columns=False, report_to="none",
        **{eval_key: t.evaluation_strategy})

    def _collate(features):                     # pixel_values as fp32, as in feature extraction
        b = data_collator(features)
        b["pixel_values"] = b["pixel_values"].float()
        return b

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      compute_metrics=compute_metrics, data_collator=_collate)
    trainer.train()
    ev = trainer.evaluate()
    trainer.save_model(str(best_dir))
    (out_root / "training_metadata.json").write_text(json.dumps(
        {"holdout": holdout, "seed": args.seed, "train_classes": train_classes,
         "label2id": label2id, "eval_accuracy": ev.get("eval_accuracy")}, indent=2))
    print(f"[done] {holdout} seed {args.seed} eval_acc={ev.get('eval_accuracy')} -> {best_dir}")


if __name__ == "__main__":
    main()
