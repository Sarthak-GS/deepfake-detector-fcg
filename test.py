"""
test.py
-------
Evaluation script for the EnhancedDeepfakeDetector (plan.txt Section 7).

Evaluates on:
  1. FF++ test split (in-domain)
  2. CDF (cross-dataset generalization)

Outputs metrics: AUC, Accuracy, F1, AP per dataset.

Usage:
  python test.py \
      --checkpoint checkpoints/run1/best.pt \
      --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
      --ffpp-root datasets/ffpp \
      --cdf-root datasets/cdf \
      --output-dir results/
"""

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model import EnhancedDeepfakeDetector
from src.dataset.dataset import (
    build_ffpp_loaders, build_cdf_loader
)


def parse_args():
    p = argparse.ArgumentParser(description='Test Deepfake Detector')
    p.add_argument('--checkpoint',    required=True,
                   help='Path to best.pt checkpoint')
    p.add_argument('--face-features', required=True,
                   help='Path to L14_real_semantic_patches_v4_2000.pickle')
    p.add_argument('--ffpp-root',     default=None,
                   help='Path to datasets/ffpp (for in-domain test)')
    p.add_argument('--cdf-root',      default=None,
                   help='Path to datasets/cdf (for cross-dataset test)')
    p.add_argument('--output-dir',    default='results',
                   help='Directory to save results JSON')
    p.add_argument('--batch-size',    type=int, default=4)
    p.add_argument('--num-frames',    type=int, default=8)
    p.add_argument('--num-workers',   type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    desc: str = 'Evaluating',
) -> dict:
    """
    Run full inference on a dataloader.
    Returns per-video averaged probabilities and labels.
    """
    model.eval()
    video_probs  = {}   # name → running sum of P(fake)
    video_counts = {}
    video_labels = {}

    for batch in tqdm(loader, desc=desc):
        x      = batch['x'].to(device)
        x_raw  = batch['x_raw'].to(device)
        labels = batch['labels'].to(device)
        names  = batch['names']

        logits_final, *_ = model(x, x_raw)
        probs = F.softmax(logits_final, dim=-1)[:, 1].cpu().numpy()

        for name, prob, lbl in zip(names, probs, labels.cpu().numpy()):
            if name not in video_probs:
                video_probs[name]  = 0.0
                video_counts[name] = 0
                video_labels[name] = int(lbl)
            video_probs[name]  += float(prob)
            video_counts[name] += 1

    # Average per video
    names_list  = list(video_probs.keys())
    probs_final = np.array([video_probs[n] / video_counts[n] for n in names_list])
    labels_arr  = np.array([video_labels[n] for n in names_list])

    return {'probs': probs_final, 'labels': labels_arr, 'names': names_list}


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute AUC, Accuracy, F1, and Average Precision.
    (plan.txt Section 7.1)
    """
    # AUC (primary metric)
    auc = roc_auc_score(labels, probs)

    # Accuracy at threshold=0.5
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)

    # F1 at threshold=0.5
    f1    = f1_score(labels, preds, zero_division=0)

    # Average Precision
    ap    = average_precision_score(labels, probs)

    return {
        'auc':      round(float(auc), 4),
        'accuracy': round(float(acc), 4),
        'f1':       round(float(f1),  4),
        'ap':       round(float(ap),  4),
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print("[INFO] Loading model...")
    model = EnhancedDeepfakeDetector(
        face_feature_path=args.face_features,
        num_frames=args.num_frames,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"  Best val AUC: {ckpt.get('val_auc', '?')}")

    all_results = {}

    # ── FF++ in-domain evaluation ───────────────────────────────────────────
    if args.ffpp_root:
        print("\n[INFO] Evaluating on FF++ test set...")
        _, _, test_loader = build_ffpp_loaders(
            ffpp_root=args.ffpp_root,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_workers=args.num_workers,
        )

        output = run_inference(model, test_loader, device, desc='FF++ test')
        metrics = compute_metrics(output['probs'], output['labels'])

        print(f"  FF++ AUC:      {metrics['auc']:.4f}")
        print(f"  FF++ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  FF++ F1:       {metrics['f1']:.4f}")
        print(f"  FF++ AP:       {metrics['ap']:.4f}")

        all_results['ffpp_test'] = metrics

    # ── CDF cross-dataset evaluation ────────────────────────────────────────
    if args.cdf_root:
        print("\n[INFO] Evaluating on Celeb-DF v2 (cross-dataset)...")
        cdf_loader = build_cdf_loader(
            cdf_root=args.cdf_root,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_workers=args.num_workers,
        )

        output = run_inference(model, cdf_loader, device, desc='CDF test')
        metrics = compute_metrics(output['probs'], output['labels'])

        print(f"  CDF AUC:      {metrics['auc']:.4f}")
        print(f"  CDF Accuracy: {metrics['accuracy']:.4f}")
        print(f"  CDF F1:       {metrics['f1']:.4f}")
        print(f"  CDF AP:       {metrics['ap']:.4f}")

        all_results['cdf_test'] = metrics

    # ── Save results ────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] Results saved to {out_path}")

    # Print summary table
    print("\n" + "="*52)
    print(f"  {'Dataset':<15} {'AUC':>8} {'Acc':>8} {'F1':>8}")
    print("="*52)
    for ds, m in all_results.items():
        print(f"  {ds:<15} {m['auc']:>8.4f} {m['accuracy']:>8.4f} {m['f1']:>8.4f}")
    print("="*52)


if __name__ == '__main__':
    main()
