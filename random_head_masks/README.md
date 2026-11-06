# Random attention-head masks (ViT-B/16, RoBERTa-base, RoBERTa-large)

The ViT-B/16 and RoBERTa runs use 25 random binary head masks per seed. Each `random_masks_{L}x{H}_seed{S}.npy` holds one seed's masks as a float32 array of shape `(25, layers, heads)`. In each mask, 1 means the head is kept and 0 means it is masked.

**How the masks are drawn.**
- They come from `generate_random_masks(25, num_layers, num_heads, seed)`.
- Mask *i* zeroes `int(L·H·p)` heads, with *p* alternating between 0.1 and 0.2.
- The heads are drawn without replacement using NumPy `RandomState(seed)`.
- The masks therefore depend only on the backbone shape and the seed. One file per (shape, seed) covers every configuration run with that seed.

| Files | Shape | Seeds | Used for |
|---|---|---|---|
| `random_masks_12x12_seed{42,123,456,789,1024}.npy` | 25 × 12 × 12 | 5 | ViT-B/16 (CIFAR-10, CIFAR-100, class-count subsets); RoBERTa-base (SST-2 vs MNLI / AG News / 20 Newsgroups); near-OOD vision and text |
| `random_masks_12x12_seed{7,13,31,77,101,202,303,404,505,2026}.npy` | 25 × 12 × 12 | 10 | RoBERTa-base SST-2 vs 20 Newsgroups, the additional seeds of the n = 15 comparison |
| `random_masks_24x16_seed{42,123,456,789,1024}.npy` | 25 × 24 × 16 | 5 | RoBERTa-large (SST-2 vs MNLI / AG News / 20 Newsgroups) |

`index.json` lists for each file:
- its shape and seed;
- the zeroed heads per mask (14/28 or 38/76);
- the mean masked fraction (0.1439 or 0.1465);
- its uses;
- its SHA-256.

`SHA256SUMS.txt` lists the same hashes in `shasum -c` format.

**Verification.** `export_random_head_masks.py` regenerates the files. Before writing, it checks two things:
- the 12 × 12 masks for seeds 42–1024 are identical, bit for bit, to the per-mask files saved by the ViT runs (`outputs/vit_results/*/masks/`);
- the 24 × 16 masks have the masked fractions recorded for RoBERTa-large.

The AHM-selected masks of the LayoutLMv3 runs are in `outputs/ahm_masks/` (hashes in `manifests/AHM_MASKS_SHA256SUMS.txt`).
