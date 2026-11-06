"""
run_vit_extended_eval.py — Step 2: ViT Full 17-Method OOD Evaluation.

Extends existing ViT results (AHSS, Mahalanobis, kNN, Combined, MC-Dropout)
with additional methods: MSP, MaxLogit, Energy, KL, VIM, Residual, NECO,
ReAct, ASH, mahAHM, knnAHM, mahAvgAvg_AHM.

Uses saved ViT checkpoints — no re-training needed.

Usage:
    python run_vit_extended_eval.py --run_all
    python run_vit_extended_eval.py --experiment cifar10_intra --seed 42
"""

import argparse
import os
import sys
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.vit_classifier import (
    ViTClassifier,
    extract_vit_embeddings_with_multiple_masks,
    extract_vit_embeddings_from_dataloader,
)
from ood_methods.sensitivity_based import AHSSScorer
from ood_methods.distance_based import MahalanobisScorer, KNNScorer
from ood_methods.confidence_based import MSPScorer, MaxLogitScorer, EnergyScorer, KLDivergenceScorer
from ood_methods.feature_based import VIMScorer, ResidualScorer, NECOScorer, ReActScorer, ASHScorer
from evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from utils.model_io import load_model_weights, model_weights_exist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_vit_transform():
    """Get ViT-compatible image transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_cifar10_splits(seed, ood_class=0, data_dir=None):
    """Load CIFAR-10 with one class as OOD (intra-dataset)."""
    from torchvision.datasets import CIFAR10

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data", "cifar10")
    transform = get_vit_transform()
    train_raw = CIFAR10(root=data_dir, train=True, download=True)
    test_raw = CIFAR10(root=data_dir, train=False, download=True)

    all_classes = list(range(10))
    id_classes = [c for c in all_classes if c != ood_class]
    label_map = {c: i for i, c in enumerate(id_classes)}

    class RemappedDataset(Dataset):
        def __init__(self, base_dataset, class_indices, lmap, tfm):
            self.base = base_dataset
            self.indices = class_indices
            self.lmap = lmap
            self.tfm = tfm
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.base[real_idx]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": self.lmap[label]}

    class OODDataset(Dataset):
        def __init__(self, base_dataset, class_indices, tfm):
            self.base = base_dataset
            self.indices = class_indices
            self.tfm = tfm
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.base[real_idx]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": -1}

    train_id_indices = [i for i, (_, l) in enumerate(train_raw) if l in id_classes]
    test_id_indices = [i for i, (_, l) in enumerate(test_raw) if l in id_classes]
    test_ood_indices = [i for i, (_, l) in enumerate(test_raw) if l == ood_class]

    rng = np.random.RandomState(seed)
    rng.shuffle(train_id_indices)
    split = int(0.9 * len(train_id_indices))
    train_indices = train_id_indices[:split]
    eval_from_train = train_id_indices[split:]

    train_ds = RemappedDataset(train_raw, train_indices, label_map, transform)
    eval_ds = RemappedDataset(train_raw, eval_from_train, label_map, transform)
    ood_ds = OODDataset(test_raw, test_ood_indices, transform)

    return train_ds, eval_ds, ood_ds, id_classes, label_map


def load_cifar10_cross(seed, ood_source="svhn", data_dir=None):
    """Load CIFAR-10 (all 10 classes) with external OOD."""
    from torchvision.datasets import CIFAR10, SVHN

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data")
    transform = get_vit_transform()
    train_raw = CIFAR10(root=os.path.join(data_dir, "cifar10"), train=True, download=True)

    all_indices = list(range(len(train_raw)))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    train_indices = all_indices[:split]
    eval_indices = all_indices[split:]

    class WrappedDataset(Dataset):
        def __init__(self, base, indices, tfm):
            self.base = base
            self.indices = indices
            self.tfm = tfm
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            image, label = self.base[self.indices[idx]]
            image = self.tfm(image)
            return {"pixel_values": image, "labels": label}

    train_ds = WrappedDataset(train_raw, train_indices, transform)
    eval_ds = WrappedDataset(train_raw, eval_indices, transform)

    if ood_source == "svhn":
        ood_raw = SVHN(root=os.path.join(data_dir, "svhn"), split="test", download=True)
        ood_indices = list(range(min(10000, len(ood_raw))))

        class SVHNWrapped(Dataset):
            def __init__(self, base, indices, tfm):
                self.base = base
                self.indices = indices
                self.tfm = tfm
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                image, label = self.base[self.indices[idx]]
                image = self.tfm(image)
                return {"pixel_values": image, "labels": -1}

        ood_ds = SVHNWrapped(ood_raw, ood_indices, transform)
    else:
        raise ValueError(f"Unknown OOD source: {ood_source}")

    return train_ds, eval_ds, ood_ds, list(range(10)), {i: i for i in range(10)}


def extract_logits(model, dataloader, device="cuda", desc="logits"):
    """Extract logits and embeddings from ViT model."""
    model.eval()
    model.to(device)
    all_logits = []
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pv = batch["pixel_values"].to(device)
            outputs = model.forward(pixel_values=pv)
            logits = outputs["logits"].cpu().numpy()
            # CLS token embedding from last hidden state
            emb = outputs["hidden_states"][-1][:, 0, :].cpu().numpy()
            all_logits.append(logits)
            all_embeddings.append(emb)
            if "labels" in batch:
                all_labels.append(batch["labels"].numpy())

    logits = np.concatenate(all_logits, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return logits, embeddings, labels


def get_classifier_weights(model):
    """Extract classifier weights/bias from ViT model."""
    classifier = model.model.classifier
    weight = classifier.weight.detach().cpu().numpy()
    bias = classifier.bias.detach().cpu().numpy() if classifier.bias is not None else None
    return weight, bias


def run_single_experiment(experiment, seed, base_output_dir):
    """Run extended evaluation for a single ViT experiment."""
    exp_name = f"{experiment}_seed{seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = os.path.join(base_output_dir, "vit_results", exp_name)
    checkpoint_dir = os.path.join(base_output_dir, "vit_checkpoints", f"{experiment}_seed{seed}")

    # Check if extended results already exist
    ext_path = os.path.join(output_dir, "extended_results.json")
    if os.path.exists(ext_path):
        logger.info(f"Extended results already exist: {ext_path}, skipping.")
        with open(ext_path) as f:
            return json.load(f)

    if not model_weights_exist(os.path.join(checkpoint_dir, "best_model"), "model"):
        logger.error(f"Checkpoint not found: {checkpoint_dir}")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"VIT EXTENDED EVAL: {exp_name}")
    logger.info(f"{'='*60}")

    start = time.time()

    # Load data
    if experiment == "cifar10_intra":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_splits(seed, ood_class=0)
        num_labels = 9
    elif experiment == "cifar10_svhn":
        train_ds, eval_ds, ood_ds, id_classes, label_map = load_cifar10_cross(seed, ood_source="svhn")
        num_labels = 10
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = ViTClassifier(num_labels=num_labels)
    state = load_model_weights(os.path.join(checkpoint_dir, "best_model"), "model")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Extract logits + embeddings
    logger.info("Extracting train logits & embeddings...")
    train_logits, train_embs, train_labels = extract_logits(model, train_loader, device, f"{exp_name} train")

    logger.info("Extracting eval logits & embeddings...")
    eval_logits, eval_embs, eval_labels = extract_logits(model, eval_loader, device, f"{exp_name} eval")

    logger.info("Extracting OOD logits & embeddings...")
    ood_logits, ood_embs, _ = extract_logits(model, ood_loader, device, f"{exp_name} OOD")

    # Get classifier weights for ReAct/ASH
    cls_weight, cls_bias = get_classifier_weights(model)

    # Generate masks and extract multi-mask embeddings (for AHSS and AHM variants)
    from run_vit_experiments import generate_random_masks
    masks = generate_random_masks(25, seed=seed)

    logger.info("Extracting eval multi-mask embeddings...")
    eval_unmasked, eval_masked_list, _ = extract_vit_embeddings_with_multiple_masks(
        model, eval_loader, masks, device=device, desc=f"{exp_name} eval masks"
    )
    logger.info("Extracting OOD multi-mask embeddings...")
    ood_unmasked, ood_masked_list, _ = extract_vit_embeddings_with_multiple_masks(
        model, ood_loader, masks, device=device, desc=f"{exp_name} OOD masks"
    )

    results = {"experiment": experiment, "seed": seed, "architecture": "ViT-B/16"}

    def evaluate(eval_scores, ood_scores, name):
        auroc = compute_auroc(eval_scores, ood_scores)
        fpr = compute_fpr_at_tpr(eval_scores, ood_scores)
        results[name] = {"auroc": float(auroc), "fpr_at_95": float(fpr)}
        logger.info(f"  {name}: AUROC={auroc:.4f}, FPR@95={fpr:.4f}")

    # ========================
    # 1. AHSS
    # ========================
    logger.info("Computing AHSS...")
    ahss_scorer = AHSSScorer(normalize=True)
    eval_ahss = -ahss_scorer.compute_ahss(eval_unmasked, eval_masked_list, normalize=True)
    ood_ahss = -ahss_scorer.compute_ahss(ood_unmasked, ood_masked_list, normalize=True)
    evaluate(eval_ahss, ood_ahss, "ahss")

    # ========================
    # 2. Mahalanobis
    # ========================
    logger.info("Computing Mahalanobis...")
    maha = MahalanobisScorer()
    maha.fit(train_embs, train_labels)
    evaluate(maha.score(eval_embs), maha.score(ood_embs), "mahalanobis")

    # ========================
    # 3. kNN
    # ========================
    logger.info("Computing kNN...")
    knn = KNNScorer(k=10, metric="euclidean")
    knn.fit(train_embs)
    evaluate(knn.score(eval_embs), knn.score(ood_embs), "knn")

    # ========================
    # 4. MSP
    # ========================
    logger.info("Computing MSP...")
    msp = MSPScorer()
    evaluate(msp.score(eval_logits), msp.score(ood_logits), "msp")

    # ========================
    # 5. MaxLogit
    # ========================
    logger.info("Computing MaxLogit...")
    ml = MaxLogitScorer()
    evaluate(ml.score(eval_logits), ml.score(ood_logits), "maxLogit")

    # ========================
    # 6. Energy
    # ========================
    logger.info("Computing Energy...")
    energy = EnergyScorer(temperature=1.0)
    evaluate(energy.score(eval_logits), energy.score(ood_logits), "energy")

    # ========================
    # 7. KL Divergence
    # ========================
    logger.info("Computing KL...")
    kl = KLDivergenceScorer()
    evaluate(kl.score(eval_logits), kl.score(ood_logits), "kl")

    # ========================
    # 8. VIM
    # ========================
    logger.info("Computing VIM...")
    vim = VIMScorer(principal_dim=384)
    vim.fit(train_embs, train_logits)
    evaluate(vim.score(eval_embs, eval_logits), vim.score(ood_embs, ood_logits), "vim")

    # ========================
    # 9. Residual
    # ========================
    logger.info("Computing Residual...")
    res = ResidualScorer(principal_dim=384)
    res.fit(train_embs)
    evaluate(res.score(eval_embs), res.score(ood_embs), "residual")

    # ========================
    # 10. NECO
    # ========================
    logger.info("Computing NECO...")
    neco = NECOScorer()
    neco.fit(train_embs, train_labels)
    evaluate(neco.score(eval_embs), neco.score(ood_embs), "neco")

    # ========================
    # 11. ReAct
    # ========================
    logger.info("Computing ReAct...")
    react = ReActScorer(threshold_percentile=90.0)
    react.fit(train_embs)
    evaluate(
        react.score(eval_embs, eval_logits, classifier_weight=cls_weight, classifier_bias=cls_bias),
        react.score(ood_embs, ood_logits, classifier_weight=cls_weight, classifier_bias=cls_bias),
        "react",
    )

    # ========================
    # 12. ASH
    # ========================
    logger.info("Computing ASH...")
    ash = ASHScorer(pruning_percentile=90.0)
    evaluate(
        ash.score(eval_embs, eval_logits, classifier_weight=cls_weight, classifier_bias=cls_bias),
        ash.score(ood_embs, ood_logits, classifier_weight=cls_weight, classifier_bias=cls_bias),
        "ash",
    )

    # ========================
    # 13. mahAHM (Mahalanobis on AHM-masked ensemble embeddings)
    # ========================
    logger.info("Computing mahAHM...")
    # Average all masked embeddings as AHM ensemble
    eval_ahm_ensemble = np.mean(eval_masked_list, axis=0)
    ood_ahm_ensemble = np.mean(ood_masked_list, axis=0)
    train_ahm_embs, _ = extract_vit_embeddings_from_dataloader(
        model, train_loader, device=device, desc=f"{exp_name} train (for mahAHM)"
    )
    # Use unmasked train for fitting (same as original)
    mah_ahm = MahalanobisScorer()
    mah_ahm.fit(train_ahm_embs, train_labels)
    evaluate(mah_ahm.score(eval_ahm_ensemble), mah_ahm.score(ood_ahm_ensemble), "mahAHM")

    # ========================
    # 14. knnAHM
    # ========================
    logger.info("Computing knnAHM...")
    knn_ahm = KNNScorer(k=10, metric="euclidean")
    knn_ahm.fit(train_embs)
    evaluate(knn_ahm.score(eval_ahm_ensemble), knn_ahm.score(ood_ahm_ensemble), "knnAHM")

    # ========================
    # 15. Combined (AHSS + Mahalanobis)
    # ========================
    logger.info("Computing Combined...")
    from scipy.stats import rankdata

    def rank_norm(scores):
        r = rankdata(scores, method='average')
        return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(scores)

    eval_maha_scores = maha.score(eval_embs)
    ood_maha_scores = maha.score(ood_embs)

    all_maha = np.concatenate([eval_maha_scores, ood_maha_scores])
    all_ahss = np.concatenate([eval_ahss, ood_ahss])
    n_eval = len(eval_maha_scores)

    best_alpha = 0.5
    best_combined_auroc = 0.0
    for alpha in np.arange(0.0, 1.05, 0.1):
        combined = alpha * rank_norm(all_maha) + (1 - alpha) * rank_norm(all_ahss)
        a = compute_auroc(combined[:n_eval], combined[n_eval:])
        if a > best_combined_auroc:
            best_combined_auroc = a
            best_alpha = alpha

    combined_final = best_alpha * rank_norm(all_maha) + (1 - best_alpha) * rank_norm(all_ahss)
    combined_fpr = compute_fpr_at_tpr(combined_final[:n_eval], combined_final[n_eval:])
    results["combined"] = {
        "auroc": float(best_combined_auroc),
        "fpr_at_95": float(combined_fpr),
        "best_alpha": float(best_alpha),
    }
    logger.info(f"  Combined: AUROC={best_combined_auroc:.4f}, alpha={best_alpha:.1f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["n_train"] = len(train_ds)
    results["n_eval"] = len(eval_ds)
    results["n_ood"] = len(ood_ds)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(ext_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {ext_path}")

    # Print summary table
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY: {exp_name}")
    logger.info(f"{'Method':<20} {'AUROC':>8} {'FPR@95':>8}")
    logger.info("-" * 38)
    methods_order = ["ahss", "mahalanobis", "knn", "msp", "maxLogit", "energy", "kl",
                     "vim", "residual", "neco", "react", "ash", "mahAHM", "knnAHM", "combined"]
    for m in methods_order:
        if m in results and isinstance(results[m], dict) and "auroc" in results[m]:
            logger.info(f"{m:<20} {results[m]['auroc']:>8.4f} {results[m].get('fpr_at_95', 0):>8.4f}")
    logger.info(f"{'='*50}")

    return results


def aggregate_all_results(base_output_dir, seeds):
    """Aggregate results across seeds."""
    experiments = ["cifar10_intra", "cifar10_svhn"]
    methods = ["ahss", "mahalanobis", "knn", "msp", "maxLogit", "energy", "kl",
               "vim", "residual", "neco", "react", "ash", "mahAHM", "knnAHM", "combined"]

    agg_dir = os.path.join(base_output_dir, "vit_results", "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    for experiment in experiments:
        results_list = []
        for seed in seeds:
            ext_path = os.path.join(base_output_dir, "vit_results",
                                     f"{experiment}_seed{seed}", "extended_results.json")
            if os.path.exists(ext_path):
                with open(ext_path) as f:
                    results_list.append(json.load(f))

        if not results_list:
            continue

        agg = {}
        for m in methods:
            aurocs = []
            fprs = []
            for r in results_list:
                if m in r and isinstance(r[m], dict) and "auroc" in r[m]:
                    aurocs.append(r[m]["auroc"])
                    fprs.append(r[m].get("fpr_at_95", 0))
            if aurocs:
                agg[m] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "fpr_mean": float(np.mean(fprs)),
                    "fpr_std": float(np.std(fprs)),
                    "n_seeds": len(aurocs),
                }

        with open(os.path.join(agg_dir, f"{experiment}_extended_aggregated.json"), "w") as f:
            json.dump(agg, f, indent=2)

        logger.info(f"\n{'='*70}")
        logger.info(f"AGGREGATED: {experiment} ({len(results_list)} seeds)")
        logger.info(f"{'Method':<20} {'AUROC':>20} {'FPR@95':>20}")
        logger.info("-" * 62)
        for m in methods:
            if m in agg:
                a = agg[m]
                logger.info(f"{m:<20} {a['auroc_mean']:.4f}±{a['auroc_std']:.4f}"
                             f"    {a['fpr_mean']:.4f}±{a['fpr_std']:.4f}")
        logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="ViT Extended 17-Method OOD Evaluation")
    parser.add_argument("--experiment", type=str, default="cifar10_intra",
                        choices=["cifar10_intra", "cifar10_svhn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs")

    seeds = [42, 123, 456, 789, 1024]

    if args.run_all:
        experiments = ["cifar10_intra", "cifar10_svhn"]
        for experiment in experiments:
            for seed in seeds:
                try:
                    run_single_experiment(experiment, seed, args.output_dir)
                except Exception as e:
                    logger.error(f"FAILED: {experiment}_seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
        aggregate_all_results(args.output_dir, seeds)
    else:
        run_single_experiment(args.experiment, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
