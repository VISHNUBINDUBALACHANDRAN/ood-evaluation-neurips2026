# Verification experiments — pre-registered protocol

Written **before** any experiment below was run (2026-10-05, 17:30 UTC). Every definition,
threshold and comparison is fixed here; results are reported whatever they show.

Paper: *NeurIPS 2026 E&D submission #2623* (AHM OOD audit). Hardware: a single
24 GB GPU, PyTorch, fp32. Software pinned to the versions of the original original runs
(Python 3.9, torch 2.5.1, transformers 4.57.6, numpy 2.0.2, scikit-learn 1.6.1, scipy 1.13.1,
datasets 4.5.0, tokenizers 0.22.2).

## 0. Ground rules (apply to every experiment)
1. **Same code.** Data loading, splits, OOD scorers (`MahalanobisScorer` with ShrunkCovariance
   0.01, `KNNScorer`, `AHSSScorer`) and metrics (`compute_auroc`, `compute_fpr_at_tpr`) are
   imported unchanged from the original project. Only model construction (offline, from local
   config files) and the head-mask hook are re-implemented, verbatim to the original.
2. **Strict weight loading.** Every checkpoint is loaded with key-by-key checking; the only
   tolerated difference is a saved `position_ids` buffer that must be *identical* to the
   model's own. Any other missing/unexpected key aborts the run.
3. **Parity gate (done, `results/gate0_parity.json`).** The re-run reproduces the originally saved
   embeddings: LayoutLMv3 cos ≥ 0.9999999 with identical predictions and accuracy; RoBERTa
   cos ≥ 0.99999999999; ViT cos ≥ 0.99997. Returning attention maps does not change hidden
   states (bit-identical, `results/bench_speed.json`), so production runs do not return them.
4. **Reproduction gate inside each experiment.** Each experiment first re-computes, in the re-run,
   at least one previously reported number, using the original (uncorrected)
   definition. A corrected result is only interpreted if its reproduction gate passes
   (tolerance stated per experiment).
5. **Seeds** {42, 123, 456, 789, 1024} unless stated. Subsamples are drawn with
   `numpy.random.RandomState(seed + 7)` and their indices are saved.
6. **Reporting.** Per-seed JSON for every run; means ± population std (ddof = 0, as in the
   paper); paired one-sided exact Wilcoxon only where a direction is pre-stated below.

## E1 — Document near-OOD (Tobacco-3482), corrected construction
*Question:* what is near-OOD performance when the "OOD" class was **never trained on**?
- **E1-R (reproduction of the original Table 4 construction).** Original
  `run_near_ood.py` design: model `tobacco_intra_seed{s}` (trained on 9 classes incl. Memo
  and Letter); ID = 20 % of Letter (`train_test_split(train_size=0.8, random_state=seed)` on
  the Letter subset); OOD = all 620 Memo docs; Mahalanobis and kNN (k = 50) fit on the original
  `train_embeddings.npy`. Gate: per-seed AUROC within ±0.002 of the stored
  `near_ood_results.json`.
- **E1-C (corrected, leave-class-out).** Models `checkpoints/leaveout_{h}_seed{s}` (released on Zenodo),
  h ∈ {Memo, Form, Resume, Scientific}: trained without class h (training metadata). The
  ID train/eval split is rebuilt with the original `run_finetune.py` procedure on the 9
  training classes; **split gate:** Re-run eval accuracy must equal the logged `eval_accuracy`
  (fp16-trainer vs fp32 tolerance: ≤ 1 document). OOD = all documents of h.
  Scores: Mahalanobis, kNN (k = 50, document protocol; k = 10 also reported), MSP, Energy.
  ID-eval sets: (i) all held-out ID eval docs; (ii) for h = Memo also Letter-only eval docs
  (the submitted "letter vs memo" pair).
- **E1-A (AHSS, h = Memo).** 25 per-layer random masks (original `create_mask` scheme,
  p ∈ {0.1, 0.2}, seeded), AHSS = −mean‖φ̂(x;M) − φ̂(x;1)‖² on L2-normalised CLS.
  (Submitted Table 4 used 5 AHM + 20 random masks of the intra model; LCO models have no AHM
  masks, so 25 random masks are used — stated in the paper.)
- FinanceDocs near-OOD is **not** re-run (no leave-class-out models exist; retraining is
  infeasible on this hardware) and is removed from the paper.

## E2 — RoBERTa-base: padding/length and fit symmetry (SST-2 → 20NG, AG News, MNLI)
*Questions:* (a) does the mahAvgAvg(_AHM) advantage on SST-2 → 20NG survive when padding
positions are excluded from the sequence mean? (b) do AHM conclusions change when the
Gaussian is fit on masked training features (symmetric) instead of unmasked ones?
- **Full-data unmasked pass** (all 60,614 train, 6,735 eval, 5,000 per OOD set):
  Mahalanobis (last-layer CLS), kNN (k = 10), X-Mahalanobis-uniform (uniform mean of layer
  1–12 CLS, original `XMahalanobisScorer` math), mahAvgAvg with padding **included**
  (original crossarch pooling) and **excluded** (mean over attention_mask = 1).
  Reproduction gate: Mahalanobis, kNN, X-Maha-uniform within ±0.002 of stored per-seed
  `full_results.json`.
- **Extra OOD set 20NG-noheader:** `fetch_20newsgroups(subset="test",
  remove=("headers","footers","quotes"))` (sklearn default shuffle, random_state = 42),
  texts with ≥ 50 non-space characters, first 5,000, truncated to 512 characters.
- **Length baseline:** AUROC of −(number of non-pad tokens) — a method-free reference.
- **Masked part (subsample):** train_sub = 3,000, eval_sub = 1,000, OOD_sub = 1,000 per set.
  Masks = the original cross-architecture generator `generate_random_masks(25, seed=seed)`
  (global, int(L·H·p) heads, p alternating 0.1/0.2). Features averaged over the 25 masks.
  - *asym* (original definition): fit on **unmasked** train_sub, score masked eval/OOD.
  - *sym* (corrected): fit on **masked** train_sub (same 25 masks), score masked eval/OOD.
  Reported for mahAHM, knnAHM (CLS), mahAvgAvg_AHM (pad-incl and pad-excl) and the unmasked
  counterparts on the same subsample. AHSS from the same passes.
  Reproduction gate (approximate, subsample): asym values within ±0.03 of the stored
  full-data values.

## E3 — RoBERTa-large: is the "fourth failure mode" an artefact of asymmetric fitting?
Same as the E2 masked part (subsample sizes identical; masks from the original generator with
L = 24, H = 16 → 38 / 76 heads), three OOD sets, five seeds. Pre-stated direction: if the
collapse is a fitting artefact, mahAHM_sym > mahAHM_asym (one-sided Wilcoxon, n = 15 pooled
seed × OOD). Reproduction gate: mahAHM_asym pooled mean within ±0.05 of the reported 0.3448.

## E4 — LayoutLMv3 MC-Dropout with correct weight loading
Models `tobacco_intra_seed{s}` loaded strictly (run_mcdropout_mechanism.py loaded no
fine-tuned weights). ID = the original 651-doc eval split, OOD = 230 Advertisement docs,
25 stochastic passes at the checkpoint's native dropout (0.1), score = −Σ_d Var(L2-normalised
CLS). Reproduction gate: per-seed AUROC within ±0.02 of the paper's per-seed values
(stochastic estimator). Also reports the mean ID variance. Seeds 42, 123, 456 first; 789, 1024 if time allows.

## E5 — ViT-B/16: fit symmetry for the AHM methods
Configs C10-intra, C10/SVHN, C100/SVHN, C100/DTD; seeds 42, 123, 456 (789, 1024 if time
allows). Same subsample design and mask generator as E2 (ViT has no padding). Reports
Mahalanobis, kNN (k = 10), mahAHM / knnAHM / mahAvgAvg_AHM in *asym* and *sym* form, AHSS.
Reproduction gate (approximate): asym values within ±0.03 of the stored full-data values.

## E6 (only if time remains) — class-count ablation with symmetric fitting
CIFAR-100 subsets with 10/20/50/100 classes vs SVHN, seeds 42/123/456, E5 design.

## What will be claimed
Only what the gated numbers support. If a reproduction gate fails, the corresponding
corrected result is reported as *not interpretable* and the paper text falls back to a
written caveat.

## Addendum A — 2026-10-05 17:48 UTC, written BEFORE E2/E3/E5/E6 started
In addition to *asym* and *sym* (mean over all 25 masks), E2/E3/E5/E6 also report a
**top-5 variant** that applies the original AHM selection criterion (Algorithm 1 as implemented
in `models/attention_masking.AHMOptimizer._compute_similarity_score`: mean top-K=10 cosine
similarity of eval to train CLS under each mask) to the same 25 masks, keeps the best F=5, and
averages features over them (*top5_sym*: fit on masked train; *top5_asym*: fit on unmasked
train). This needs no extra forward passes. As in the original LayoutLMv3 pipeline, selection
uses the ID eval split. The only remaining difference from the paper's described method is the
mask *sampling* scheme (global int(L·H·p) heads, i.e. the generator the original ViT/RoBERTa
code used, instead of the per-layer scheme).

## Addendum B — 2026-10-06 08:40 UTC, written BEFORE E5/E6 started (no E5/E6 result exists)
A CPU dry run of the E5/E6 data loaders showed that the E6 configuration `c100_c10_svhn` has only
**500** ID-eval images (CIFAR-100 restricted to 10 classes), fewer than eval_sub = 1,000; drawing
1,000 without replacement would abort E6. As was already done for OOD sets, every subsample size
is now capped at the number available: n = min(target, available). Effect: `c100_c10_svhn` uses
all 500 eval images (train 3,000 of 4,500 and OOD 1,000 of 10,000 unchanged). For every other
E5/E6 configuration min() returns the target, so the random draws (RandomState(seed + 7)) and
subsamples are identical to the original specification.

## Addendum C — 2026-10-07 08:57 UTC, written BEFORE E4 seeds 123/456/789/1024 and E1-A were run
**What happened.** The first E4 run (step E4_mc_s3, started 6 Oct 23:09 UTC) completed seed 42 (7 Oct 02:20 UTC; weights gate and
reproduction gate passed: AUROC 0.9308 vs 0.9286) but ran ≈ 3× slower than planned because the process held ≈ 23 GB of
the machine's 24 GB and swapped. With the author's approval it was stopped at 08:44 UTC during seed 123; no partial result
of seed 123 was saved or used (queue log: "FAIL E4_mc_s3 rc=143" = stopped on request).
**Diagnosis** (`code/diag_e4_memory.py`, `logs/E4_diag.log`): one LayoutLMv3 batch (16 documents, hidden states
returned) makes PyTorch's GPU caching allocator keep ≈ 11 GB; each further batch size (E4 uses 16, 11 and 6) keeps its
own cached pool (≈ 12 + 8 + 4 GB). It is cached-but-unused memory, not a leak: releasing the cache after every batch
keeps memory at 1.6 GB in both eval and dropout mode.
**Change.** E4 (accuracy pass and all MC-Dropout passes) and E1-A (all feature passes) release the GPU cache after every
batch (synchronize, `gc.collect()`, empty the cache: `common.release_device_cache`). Implemented
as an opt-in argument (`run_features(..., release_cache=True)`); the default leaves the code path of every result
produced before this addendum unchanged. E4 also logs progress every 5 passes. Nothing else changes (same model mode,
same seeds, same random-number stream, same 25 passes, same documents).
**Verification before use** (`code/validate_e4_fix.py` → `results/_validation/E4_fix/validation.json`, PASS):
(1) the original loop (verbatim copy of the code that produced seed 42) and the changed loop, same seed and documents
(batches 16/16/8), give **bit-identical** embeddings (max |diff| = 0.0) on both passes, with dropout active (the two
passes differ); hence the saved seed-42 result is computationally identical to what the changed code produces;
(2) full E4-size passes (651 ID and 230 OOD documents) stay at 1.41–1.43 GB with no growth.
**Queue.** E4 seeds 123 and 456 run as step E4_mc_s3_fix (seed 42 is kept, not recomputed); seeds 789/1024 (E4_mc_s2)
and E1-A use the same changed code.
