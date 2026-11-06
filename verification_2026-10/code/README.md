# Re-verification (October 2026)

Re-run of the experiments behind the corrected values reported in the paper
(single GPU, PyTorch, fp32), from the original code and checkpoints.
Read **PROTOCOL.md** first: it was written before any experiment ran and fixes every definition, gate and
comparison in advance (Addenda A–C are dated and precede the runs they affect).

| File | What it is |
|---|---|
| `PROTOCOL.md` | Pre-registered protocol (experiments E1–E6, gates, tolerances, addenda). |
| `common.py` | Model construction, **strict** weight loading, head-mask hook, feature extraction. Imports data loaders, scorers and metrics unchanged from the main code. |
| `gate0_parity.py` | Gate 0: reproduces the stored original embeddings / predictions (LayoutLMv3, RoBERTa, ViT). |
| `bench_speed.py` | Shows that not returning attention maps leaves hidden states bit-identical (PROTOCOL §1). |
| `e1_doc_nearood.py` | E1: document near-OOD — reproduction of submitted Table 4 (E1-R), corrected leave-class-out (E1-C), AHSS (E1-A). |
| `e2_roberta.py` | E2 (RoBERTa-base) and E3 (RoBERTa-large): padding/length and fit-on-masked vs fit-on-unmasked features. |
| `masked_sym.py` | Shared masked-feature logic for E2/E3/E5/E6. |
| `e4_mcdropout_layoutlmv3.py` | E4: LayoutLMv3 MC-Dropout with the fine-tuned weights. |
| `diag_e4_memory.py`, `validate_e4_fix.py` | Addendum C: GPU-memory diagnosis and the bit-identity validation of the cache-release fix. |
| `e5_vit.py` | E5 (ViT) and E6 (CIFAR-100 class-count). |
| `verify_blas_warnings.py` | Checks that floating-point warnings during covariance fitting did not change any score. |
| `aggregate.py` | Builds `summary/SUMMARY.md` + `summary/summary.json` from the per-seed results, re-checking every gate (the final copies are included here under `results/`). |

## Re-running (order used)
By default the scripts use this repository (override with `PROJECT_DIR`): datasets under `data/`, checkpoints under `outputs/` (leave-class-out models under `outputs/checkpoints/leaveout_<class>_seed<s>`, or set `LEAVE_CLASS_OUT_DIR` to the folder that holds `checkpoints/`). Then, from this folder:
```bash
python gate0_parity.py                                   # must print ALL_PASS=True
python e1_doc_nearood.py lco Memo Form Resume Scientific
python e2_roberta.py base 42 123 456 789 1024
python e2_roberta.py large 42 123 456 789 1024
python e4_mcdropout_layoutlmv3.py 42 123 456 789 1024
python e1_doc_nearood.py ahss
python e5_vit.py c10_intra,c100_svhn,c10_svhn,c100_dtd 42 123 456 789 1024
python e5_vit.py c100_c10_svhn,c100_c20_svhn,c100_c50_svhn 42 123 456
python verify_blas_warnings.py
python aggregate.py --final
```
Software: Python 3.9, torch 2.5.1, transformers 4.57.6, numpy 2.0.2, scikit-learn 1.6.1, scipy 1.13.1 (as PROTOCOL.md).

## Why these numbers can be trusted
1. Same software versions as the original runs; same data loaders and scorers (imported, not rewritten).
2. Strict weight loading (no silent partial loads).
3. Gate 0: re-computed embeddings match the stored original ones (cosine >= 0.99997 on all three backbones).
4. Every experiment first re-computes numbers the paper reported before any corrected number is interpreted
   (E1-R reproduced all five per-seed Table-4 values within 1e-4).
