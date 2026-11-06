"""
run_mask_analysis.py — Analyze AHM mask patterns across experiments.

Generates:
- Per-layer/per-head masking frequency heatmap data
- Mask overlap analysis across seeds and experiments
- Correlation between mask patterns and AHSS performance

For paper Section 4.3 (Analysis).

Usage:
    python run_mask_analysis.py
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from configs.config import get_default_config

config = get_default_config()
NUM_LAYERS = config.model.num_layers
NUM_HEADS = config.model.num_heads
SEEDS = config.seeds
EXPERIMENTS = [
    ("financedocs", "intra"),
    ("financedocs", "cross"),
    ("tobacco", "intra"),
    ("tobacco", "cross"),
]


def load_ahm_masks(exp_name):
    """Load the AHM masks for an experiment."""
    ahm_dir = os.path.join(BASE_DIR, "outputs", "ahm_masks", exp_name)
    masks = []
    if not os.path.exists(ahm_dir):
        return None

    for fname in sorted(os.listdir(ahm_dir)):
        if fname.endswith(".npy") and (fname.startswith("mask_") or fname.startswith("best_mask_")):
            mask = np.load(os.path.join(ahm_dir, fname))
            masks.append(mask)

    # Also check for combined mask file
    combined_path = os.path.join(ahm_dir, "ahm_masks.npy")
    if os.path.exists(combined_path) and not masks:
        masks_arr = np.load(combined_path)
        masks = [masks_arr[i] for i in range(masks_arr.shape[0])]

    return masks if masks else None


def compute_head_masking_frequency(all_masks):
    """Compute how often each head is masked (0) across all masks."""
    freq = np.zeros((NUM_LAYERS, NUM_HEADS))
    count = 0
    for masks in all_masks:
        if masks is None:
            continue
        for mask in masks:
            if mask.shape == (NUM_LAYERS, NUM_HEADS):
                freq += (mask == 0).astype(float)
                count += 1
            elif mask.shape == (NUM_LAYERS * NUM_HEADS,):
                reshaped = mask.reshape(NUM_LAYERS, NUM_HEADS)
                freq += (reshaped == 0).astype(float)
                count += 1
    if count > 0:
        freq /= count
    return freq


def compute_mask_overlap(masks1, masks2):
    """Compute Jaccard similarity between two sets of masks."""
    if masks1 is None or masks2 is None:
        return None

    overlaps = []
    for m1 in masks1:
        m1_flat = m1.flatten()
        for m2 in masks2:
            m2_flat = m2.flatten()
            masked1 = set(np.where(m1_flat == 0)[0])
            masked2 = set(np.where(m2_flat == 0)[0])
            if len(masked1) == 0 and len(masked2) == 0:
                overlaps.append(1.0)
            elif len(masked1 | masked2) == 0:
                overlaps.append(0.0)
            else:
                overlaps.append(len(masked1 & masked2) / len(masked1 | masked2))

    return np.mean(overlaps) if overlaps else None


def main():
    print("=" * 70)
    print("ATTENTION HEAD MASK ANALYSIS")
    print("=" * 70)

    # Load all AHM masks
    all_exp_masks = {}
    for ds, et in EXPERIMENTS:
        for seed in SEEDS:
            exp_name = f"{ds}_{et}_seed{seed}"
            masks = load_ahm_masks(exp_name)
            if masks:
                all_exp_masks[exp_name] = masks
                print(f"  Loaded {len(masks)} masks for {exp_name}")

    if not all_exp_masks:
        print("No AHM masks found!")
        return

    print(f"\nTotal: {len(all_exp_masks)} experiments with masks")

    # Per-experiment-type analysis
    print("\n" + "=" * 70)
    print("HEAD MASKING FREQUENCY BY EXPERIMENT TYPE")
    print("(fraction of AHM masks that mask each head)")
    print("=" * 70)

    exp_type_masks = defaultdict(list)
    for exp_name, masks in all_exp_masks.items():
        key = exp_name.rsplit("_seed", 1)[0]
        exp_type_masks[key].append(masks)

    all_freqs = {}
    for key in sorted(exp_type_masks.keys()):
        freq = compute_head_masking_frequency(exp_type_masks[key])
        all_freqs[key] = freq
        print(f"\n{key}:")
        print(f"  Most masked heads (top 10):")
        flat_indices = np.argsort(freq.flatten())[::-1]
        for idx in flat_indices[:10]:
            layer = idx // NUM_HEADS
            head = idx % NUM_HEADS
            print(f"    L{layer:2d}-H{head:2d}: {freq[layer, head]:.2%}")

        # Layer-level statistics
        layer_freq = freq.mean(axis=1)
        print(f"  Layer masking rate (mean per layer):")
        for l in range(NUM_LAYERS):
            bar = "█" * int(layer_freq[l] * 40)
            print(f"    L{l:2d}: {layer_freq[l]:.3f} {bar}")

    # Cross-seed mask overlap
    print("\n" + "=" * 70)
    print("CROSS-SEED MASK OVERLAP (Jaccard similarity)")
    print("=" * 70)

    for key in sorted(exp_type_masks.keys()):
        seed_masks = {}
        for exp_name, masks in all_exp_masks.items():
            if exp_name.startswith(key + "_seed"):
                seed = int(exp_name.split("_seed")[1])
                seed_masks[seed] = masks

        if len(seed_masks) < 2:
            continue

        print(f"\n{key}:")
        seeds_sorted = sorted(seed_masks.keys())
        for i, s1 in enumerate(seeds_sorted):
            for s2 in seeds_sorted[i + 1:]:
                overlap = compute_mask_overlap(seed_masks[s1], seed_masks[s2])
                if overlap is not None:
                    print(f"  seed {s1} vs {s2}: Jaccard = {overlap:.4f}")

    # Cross-experiment mask overlap
    print("\n" + "=" * 70)
    print("CROSS-EXPERIMENT MASK OVERLAP")
    print("=" * 70)

    exp_keys = sorted(exp_type_masks.keys())
    for i, k1 in enumerate(exp_keys):
        for k2 in exp_keys[i + 1:]:
            overlaps = []
            for m1 in exp_type_masks[k1]:
                for m2 in exp_type_masks[k2]:
                    ov = compute_mask_overlap(m1, m2)
                    if ov is not None:
                        overlaps.append(ov)
            if overlaps:
                print(f"  {k1} vs {k2}: Jaccard = {np.mean(overlaps):.4f}±{np.std(overlaps):.4f}")

    # Global head importance ranking
    print("\n" + "=" * 70)
    print("GLOBAL HEAD IMPORTANCE RANKING")
    print("(aggregated masking frequency across all experiments)")
    print("=" * 70)

    global_freq = compute_head_masking_frequency(list(all_exp_masks.values()))
    flat_indices = np.argsort(global_freq.flatten())[::-1]
    print("\nTop 20 most frequently masked heads:")
    for rank, idx in enumerate(flat_indices[:20]):
        layer = idx // NUM_HEADS
        head = idx % NUM_HEADS
        print(f"  #{rank + 1:2d}: Layer {layer:2d}, Head {head:2d} — masked {global_freq[layer, head]:.1%} of the time")

    print("\nBottom 10 (least masked, most critical for representations):")
    for rank, idx in enumerate(flat_indices[-10:]):
        layer = idx // NUM_HEADS
        head = idx % NUM_HEADS
        print(f"  #{NUM_LAYERS * NUM_HEADS - 9 + rank}: Layer {layer:2d}, Head {head:2d} — masked {global_freq[layer, head]:.1%}")

    # Save results
    out_dir = os.path.join(BASE_DIR, "outputs", "mask_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Save frequency heatmap data
    for key, freq in all_freqs.items():
        np.save(os.path.join(out_dir, f"{key}_head_freq.npy"), freq)
    np.save(os.path.join(out_dir, "global_head_freq.npy"), global_freq)

    # Save summary
    summary = {
        "num_experiments": len(all_exp_masks),
        "experiments": list(all_exp_masks.keys()),
        "top_10_global": [
            {"layer": int(idx // NUM_HEADS), "head": int(idx % NUM_HEADS),
             "freq": float(global_freq.flatten()[idx])}
            for idx in flat_indices[:10]
        ],
    }
    with open(os.path.join(out_dir, "mask_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
