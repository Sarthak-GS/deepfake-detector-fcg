"""
train.py
--------
Training script for the EnhancedDeepfakeDetector (plan.txt Section 6).

Key changes vs original:
  - best.pt decided by COMPOSITE score = auc_weight*AUC + (1-auc_weight)*(1-norm_loss)
  - Smoothed scoring over last --smooth-window epochs (avoids epoch-1 fluke)
  - Warmup: early stopping disabled for first --min-epochs epochs
  - Patience raised to 15 by default
  - history saved into checkpoint for clean resume
"""

import os
import sys
import json
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import AUROC, Accuracy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model import EnhancedDeepfakeDetector
from src.dataset.dataset import build_ffpp_loaders, build_simple_loaders
from src.preprocess.preprocess import run_fetch_landmarks, run_crop_faces, run_simple_preprocess


def seed_everything(seed: int = 1019):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser(description='Train Enhanced Deepfake Detector')
    p.add_argument('--ffpp-root',       required=True)
    p.add_argument('--face-features',   required=True)
    p.add_argument('--output-dir',      default='checkpoints')
    p.add_argument('--batch-size',      type=int, default=3)
    p.add_argument('--epochs',          type=int, default=30)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight-decay',    type=float, default=1e-3)
    p.add_argument('--num-frames',      type=int, default=8)
    p.add_argument('--num-workers',     type=int, default=4)
    p.add_argument('--accum-steps',     type=int, default=2)
    p.add_argument('--patience',        type=int, default=15,
                   help='Early stopping patience (default raised from 10 → 15)')
    # ── NEW ──────────────────────────────────────────────────────────────────
    p.add_argument('--min-epochs',      type=int, default=5,
                   help='Warmup: early stopping disabled for first N epochs')
    p.add_argument('--smooth-window',   type=int, default=3,
                   help='Average composite score over last N epochs before comparing')
    p.add_argument('--auc-weight',      type=float, default=0.7,
                   help='Weight of AUC in composite score (rest = 1-norm_loss)')
    # ─────────────────────────────────────────────────────────────────────────
    p.add_argument('--seed',            type=int, default=1019)
    p.add_argument('--resume',          default=None)
    p.add_argument('--no-amp',          action='store_true')
    p.add_argument('--limit-samples',   type=int, default=None)
    p.add_argument('--dataset-type',    type=str, default='ffpp',
                   choices=['ffpp', 'simple'])
    p.add_argument('--config',          default='configs/config.yaml')
    p.add_argument('--skip-prep',       action='store_true')
    p.add_argument('--cdf-root',        default=None)
    return p.parse_args(), p


def load_config_into_args(args, parser=None):
    try:
        import yaml
    except ImportError:
        return args
    if not os.path.isfile(args.config):
        return args
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cli_explicit = set()
    for arg_str in sys.argv[1:]:
        if arg_str.startswith('--'):
            cli_explicit.add(arg_str.lstrip('-').replace('-', '_'))
    key_map = {
        'num_frames': 'num_frames', 'num_workers': 'num_workers',
        'batch_size': 'batch_size', 'epochs': 'epochs',
        'lr': 'lr', 'weight_decay': 'weight_decay',
        'accum_steps': 'accum_steps', 'patience': 'patience',
        'seed': 'seed', 'min_epochs': 'min_epochs',
        'smooth_window': 'smooth_window', 'auc_weight': 'auc_weight',
    }
    for yaml_key, attr in key_map.items():
        if yaml_key in cfg and attr not in cli_explicit:
            setattr(args, attr, cfg[yaml_key])
    print(f"[INFO] Config loaded from {args.config}")
    return args


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    auc_calc = AUROC(task='binary').to(device)
    acc_calc = Accuracy(task='binary').to(device)
    all_probs, all_labels = [], []

    for batch in loader:
        x      = batch.get('x')
        x_raw  = batch.get('x_raw')
        labels = batch.get('labels')
        if x is None or x_raw is None or labels is None:
            continue
        x, x_raw, labels = x.to(device), x_raw.to(device), labels.to(device)
        logits_final, *_ = model(x, x_raw)
        probs = torch.softmax(logits_final, dim=-1)[:, 1]
        all_probs.append(probs)
        all_labels.append(labels)

    if not all_probs:
        print("  [WARN] evaluate() got zero valid batches — returning AUC=0.5")
        return {'auc': 0.5, 'accuracy': 0.5}

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    if len(all_labels.unique()) < 2:
        print(f"  [WARN] Only one class in val set — AUC undefined, returning 0.5")
        return {'auc': 0.5, 'accuracy': round(acc_calc(all_probs, all_labels).item(), 4)}

    return {
        'auc':      round(auc_calc(all_probs, all_labels).item(), 4),
        'accuracy': round(acc_calc(all_probs, all_labels).item(), 4),
    }


def compute_composite_score(auc: float, train_loss: float,
                             history: list, auc_weight: float) -> float:
    """
    composite = auc_weight * AUC + (1 - auc_weight) * (1 - norm_loss)

    WHY COMPOSITE SCORE FOR best.pt?
    ─────────────────────────────────
    • Pure AUC  → can spike on epoch 1 if val set is small (only 24 batches
                  here). A lucky batch ordering gives fake 0.9981 before the
                  model has actually learned anything useful.
    • Pure loss → ignores generalisation; a model can memorise train set.
    • Composite → rewards a checkpoint that is BOTH generalising well (high
                  AUC) AND still actively learning (low train loss relative
                  to its own history). The ratio 70/30 keeps AUC dominant
                  while loss acts as a tie-breaker / sanity check.

    norm_loss is relative to the worst (highest) loss seen so far, so it
    always lives in [0, 1] regardless of the absolute loss scale.
    """
    loss_weight = 1.0 - auc_weight
    if history:
        max_loss  = max(h['train_loss'] for h in history)
        norm_loss = train_loss / max_loss if max_loss > 0 else 0.0
    else:
        norm_loss = 0.0   # first epoch: neutral
    return round(auc_weight * auc + loss_weight * (1.0 - norm_loss), 6)


def smoothed_score(history: list, window: int, auc_weight: float) -> float:
    """
    Average composite score over the last `window` epochs.
    Smoothing prevents a single lucky val batch from saving best.pt or
    resetting the patience counter.
    """
    recent = history[-window:]
    scores = [
        compute_composite_score(h['val_auc'], h['train_loss'], history, auc_weight)
        for h in recent
    ]
    return sum(scores) / len(scores)


def main():
    args, parser = parse_args()
    args = load_config_into_args(args, parser)
    seed_everything(args.seed)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and torch.cuda.is_available()

    print(f"[INFO] Device: {device} | AMP: {use_amp}")
    print(f"[INFO] Early-stop patience : {args.patience} epochs  (was 10)")
    print(f"[INFO] Warmup (no ES)      : first {args.min_epochs} epochs")
    print(f"[INFO] Score smoothing     : last {args.smooth_window} epochs")
    print(f"[INFO] best.pt score       : {args.auc_weight:.0%} AUC + "
          f"{1-args.auc_weight:.0%} (1-norm_loss)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 0. Preprocessing ────────────────────────────────────────────────────
    if not args.skip_prep:
        print("[INFO] Starting automatic data preprocessing...")
        try:
            if args.dataset_type == 'simple':
                out_dir = args.ffpp_root.rstrip('/') + '_cropped'
                run_simple_preprocess(in_dir=args.ffpp_root, out_dir=out_dir,
                                      mean_face_path='misc/20words_mean_face.npy',
                                      crop_w=224, crop_h=224)
                args.ffpp_root = out_dir
                print(f"  [Prep] Data root updated to: {args.ffpp_root}")
            else:
                run_fetch_landmarks(root_dir=args.ffpp_root, video_subdir='videos',
                                    fdata_subdir='frame_data', glob_exp='*/*', max_res=1920)
                run_crop_faces(root_dir=args.ffpp_root, video_subdir='videos',
                               fdata_subdir='frame_data', crop_dir='cropped',
                               glob_exp='*/*', crop_w=150, crop_h=150,
                               mean_face_path='misc/20words_mean_face.npy')
                if args.cdf_root and os.path.isdir(args.cdf_root):
                    run_fetch_landmarks(root_dir=args.cdf_root, video_subdir='videos',
                                        fdata_subdir='frame_data', glob_exp='*', max_res=1920)
                    run_crop_faces(root_dir=args.cdf_root, video_subdir='videos',
                                   fdata_subdir='frame_data', crop_dir='cropped',
                                   glob_exp='*', crop_w=150, crop_h=150,
                                   mean_face_path='misc/20words_mean_face.npy')
        except ImportError:
            print("[WARNING] face-alignment not installed — skipping preprocessing.")
        except Exception as e:
            print(f"[WARNING] Preprocessing failed: {e}")

    # ── 1. DataLoaders ──────────────────────────────────────────────────────
    print(f"[INFO] Building dataloaders (type: {args.dataset_type})...")
    if args.dataset_type == 'simple':
        train_loader, val_loader, test_loader = build_simple_loaders(
            root_dir=args.ffpp_root, batch_size=args.batch_size,
            num_frames=args.num_frames, num_workers=args.num_workers,
            val_frac=0.15, test_frac=0.10, seed=args.seed,
            limit_samples=args.limit_samples,
        )
    else:
        train_loader, val_loader, test_loader = build_ffpp_loaders(
            ffpp_root=args.ffpp_root, batch_size=args.batch_size,
            num_frames=args.num_frames, num_workers=args.num_workers,
            limit_samples=args.limit_samples,
        )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")

    # ── 2. Model ────────────────────────────────────────────────────────────
    print("[INFO] Building model (loading OpenCLIP ViT-L/14)...")
    model = EnhancedDeepfakeDetector(
        face_feature_path=args.face_features, num_frames=args.num_frames,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen    params: {frozen:,}")

    # ── 3. Optimizer ────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # ── 4. Resume ───────────────────────────────────────────────────────────
    start_epoch      = 0
    best_score       = 0.0
    best_auc_ever    = 0.0
    history          = []
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_score    = ckpt.get('best_score', 0.0)
        best_auc_ever = ckpt.get('best_auc', 0.0)
        history       = ckpt.get('history', [])
        print(f"  Epoch {start_epoch} | best_score={best_score:.4f} | best_auc={best_auc_ever:.4f}")

    # ── 5. Training Loop ────────────────────────────────────────────────────
    print("[INFO] Clearing CUDA cache and starting training...")
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.visual.eval()   # encoder always frozen

        epoch_loss = epoch_cls_loss = epoch_fcg_loss = 0.0
        n_batches  = 0
        _has_unflushed_grads = False
        _optimizer_stepped   = False
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            try:
                x      = batch['x'].to(device)
                x_raw  = batch['x_raw'].to(device)
                labels = batch['labels'].to(device)

                with autocast(enabled=use_amp):
                    logits_final, logits_s, logits_t, logits_f, logits_a, sq = model(x, x_raw)
                    total_loss, cls_l, fcg_l = model.compute_loss(
                        logits_s, logits_t, logits_f, logits_a,
                        sq, labels, cls_weight=10.0, ffg_weight=1.5,
                    )
                    total_loss = total_loss / args.accum_steps

                scaler.scale(total_loss).backward()
                _has_unflushed_grads = True

                if (step + 1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    _has_unflushed_grads = False
                    _optimizer_stepped   = True

                epoch_loss     += total_loss.item() * args.accum_steps
                epoch_cls_loss += cls_l.item()
                epoch_fcg_loss += fcg_l.item() if isinstance(fcg_l, torch.Tensor) else fcg_l
                n_batches += 1

                if step % 20 == 0:
                    print(
                        f"  Epoch [{epoch+1}/{args.epochs}] "
                        f"Step [{step+1}/{len(train_loader)}] "
                        f"Loss: {total_loss.item()*args.accum_steps:.4f} "
                        f"(cls: {cls_l.item():.4f}, "
                        f"fcg: {fcg_l.item() if isinstance(fcg_l, torch.Tensor) else fcg_l:.4f})"
                    )
            except Exception as e:
                print(f"  [ERROR] Step {step} crashed: {e}")
                import traceback; traceback.print_exc()
                optimizer.zero_grad()
                _has_unflushed_grads = False
                continue

        if _has_unflushed_grads:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            _optimizer_stepped = True

        if _optimizer_stepped:
            scheduler.step()
        else:
            print("  [WARN] No optimizer step this epoch — scheduler not stepped")

        avg_loss = epoch_loss     / max(n_batches, 1)
        avg_cls  = epoch_cls_loss / max(n_batches, 1)
        avg_fcg  = epoch_fcg_loss / max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────────
        print(f"[INFO] Validating epoch {epoch+1}...")
        val_metrics = evaluate(model, val_loader, device)
        val_auc, val_acc = val_metrics['auc'], val_metrics['accuracy']

        epoch_score = compute_composite_score(val_auc, avg_loss, history, args.auc_weight)

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Composite: {epoch_score:.4f}"
        )

        record = {
            'epoch': epoch + 1, 'train_loss': avg_loss,
            'train_cls_loss': avg_cls, 'train_fcg_loss': avg_fcg,
            'val_auc': val_auc, 'val_accuracy': val_acc, 'composite': epoch_score,
        }
        history.append(record)
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # ── Smoothed score ──────────────────────────────────────────────────
        smooth    = smoothed_score(history, args.smooth_window, args.auc_weight)
        in_warmup = (epoch + 1) < args.min_epochs

        # ── Checkpoint & Early Stopping ─────────────────────────────────────
        try:
            ckpt = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc':              val_auc,
                'best_auc':             best_auc_ever,
                'best_score':           best_score,
                'history':              history,
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'last.pt'))

            if smooth > best_score:
                best_score    = smooth
                best_auc_ever = max(best_auc_ever, val_auc)
                patience_counter = 0
                torch.save(ckpt, os.path.join(args.output_dir, 'best.pt'))
                print(f"  ✓ New best  smooth={smooth:.4f}  AUC={val_auc:.4f} — saved best.pt")
            else:
                if in_warmup:
                    print(
                        f"  [Warmup {epoch+1}/{args.min_epochs}] "
                        f"smooth={smooth:.4f} — early stopping paused"
                    )
                else:
                    patience_counter += 1
                    print(
                        f"  No improvement  smooth={smooth:.4f} ≤ best={best_score:.4f} "
                        f"[{patience_counter}/{args.patience}]"
                    )
                    if patience_counter >= args.patience:
                        print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                        break

        except Exception as e:
            print(f"  [ERROR] Checkpoint save failed: {e}")
            print(f"  [ERROR] Check disk: df -h {args.output_dir}")
            import traceback; traceback.print_exc()

    # ── Final test evaluation ───────────────────────────────────────────────
    best_path = os.path.join(args.output_dir, 'best.pt')
    if os.path.isfile(best_path):
        print("\n[INFO] Running final test evaluation on best.pt...")
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
        test_metrics = evaluate(model, test_loader, device)
        print(f"[FINAL] Test AUC: {test_metrics['auc']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    else:
        print("\n[WARN] No best.pt found — skipping final test evaluation.")

    print(f"\n[DONE] Best composite score : {best_score:.4f}")
    print(f"       Best val AUC ever    : {best_auc_ever:.4f}")
    print(f"       Checkpoint           : {best_path}")


if __name__ == '__main__':
    main()