"""export_random_head_masks.py — write the random attention-head masks used for ViT-B/16, RoBERTa-base and RoBERTa-large.

These runs draw 25 random binary head masks per seed with generate_random_masks(num_masks, num_layers, num_heads, seed)
(run_vit_experiments.py, run_cifar100_benchmark.py, run_roberta_experiments.py, run_nearood_polarity.py,
run_mahavgavg_ahm_crossarch.py; all share this generator). The masks depend only on the backbone shape and the seed,
so one file per (shape, seed) covers every configuration.

Checks before writing:
  * 12x12 masks equal, bit for bit, the per-mask files stored by the ViT runs (outputs/vit_results/*_seed*/masks);
  * 24x16 masks give 38 / 76 zeroed heads and a mean masked fraction of 0.1465, as recorded for RoBERTa-large.

  python random_head_masks/export_random_head_masks.py [--outputs outputs] [--out random_head_masks]
"""
import argparse, hashlib, json, os
import numpy as np

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS_EXTRA = [7, 13, 31, 77, 101, 202, 303, 404, 505, 2026]     # RoBERTa-base SST-2 -> 20 Newsgroups, n = 15


def generate_random_masks(num_masks, num_layers=12, num_heads=12, masking_pcts=(0.1, 0.2), seed=42):
    """Same procedure as generate_random_masks in the run scripts (numpy part; they wrap each mask in a torch tensor)."""
    rng = np.random.RandomState(seed)
    masks = []
    for i in range(num_masks):
        pct = masking_pcts[i % len(masking_pcts)]
        mask = np.ones((num_layers, num_heads), dtype=np.float32)
        n_to_mask = int(num_layers * num_heads * pct)
        flat = mask.flatten()
        indices = rng.choice(len(flat), size=n_to_mask, replace=False)
        flat[indices] = 0.0
        masks.append(flat.reshape(num_layers, num_heads))
    return np.stack(masks)


USES = {
    (12, 12, "main"): ["ViT-B/16: CIFAR-10 intra, CIFAR-10 vs SVHN, CIFAR-100 vs SVHN / DTD, CIFAR-100 class-count subsets",
                       "RoBERTa-base: SST-2 vs MNLI / AG News / 20 Newsgroups",
                       "near-OOD: ViT-B/16 CIFAR-100 small mammals, RoBERTa-base 20 Newsgroups comp.*"],
    (12, 12, "extra"): ["RoBERTa-base: SST-2 vs 20 Newsgroups, additional seeds of the n = 15 comparison"],
    (24, 16, "main"): ["RoBERTa-large: SST-2 vs MNLI / AG News / 20 Newsgroups"],
}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outputs", default=os.path.join(here, "..", "outputs"))
    ap.add_argument("--out", default=here)
    a = ap.parse_args()
    from safetensors.numpy import load_file

    files = {}
    for (L, H, group), uses in USES.items():
        for s in (SEEDS_EXTRA if group == "extra" else SEEDS5):
            m = generate_random_masks(25, L, H, seed=s)
            zeroed = sorted({int((x == 0).sum()) for x in m})
            frac = float((m == 0).mean())
            if (L, H) == (12, 12):
                assert zeroed == [14, 28] and abs(frac - 0.143889) < 1e-5, (s, zeroed, frac)
                if group == "main":                       # bit-exact check against the masks stored by the ViT runs
                    for cfg in ("cifar10_intra", "cifar10_svhn"):
                        d = os.path.join(a.outputs, "vit_results", f"{cfg}_seed{s}", "masks")
                        stored = np.stack([load_file(os.path.join(d, f"mask_{i}.safetensors"))["tensor"] for i in range(25)])
                        assert np.array_equal(stored, m), f"mask mismatch {cfg} seed {s}"
            else:
                assert zeroed == [38, 76] and abs(frac - 0.146458) < 1e-5, (s, zeroed, frac)
            name = f"random_masks_{L}x{H}_seed{s}.npy"
            files[name] = (m, {"shape": [25, L, H], "seed": s, "zeroed_heads_per_mask": zeroed,
                               "mean_masked_fraction": round(frac, 6), "used_by": uses})
    os.makedirs(a.out, exist_ok=True)
    index, sums = {}, []
    for name, (m, meta) in files.items():
        p = os.path.join(a.out, name)
        np.save(p, m)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        index[name] = {**meta, "sha256": h}
        sums.append(f"{h}  {name}")
    json.dump({"generator": "generate_random_masks(25, num_layers, num_heads, masking_pcts=(0.1, 0.2), seed): mask i masks "
                            "int(L*H*p) heads drawn without replacement with numpy RandomState(seed), p alternating 0.1 / 0.2; "
                            "1 = head kept, 0 = head masked",
               "files": index}, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    open(os.path.join(a.out, "SHA256SUMS.txt"), "w").write("\n".join(sums) + "\n")
    print(f"wrote {len(files)} mask files; 12x12 seeds {SEEDS5} verified bit-exact against the stored ViT masks")


if __name__ == "__main__":
    main()
