"""
train.py — Training Script for Hybrid Face Recognition Model
Optimized for CPU (no GPU required).

Usage:
  python train.py --data data/train --model hybrid_concat --epochs 20
  python train.py --data data/train --model resnet50 --epochs 20
  python train.py --data data/train --model hybrid_attention --epochs 10 --max_id 200

  # New options for improving the hybrid model:
  python train.py --model hybrid_concat --max_id 30 --epochs 20 \
                   --vit_input_size 160 --freeze_vit_epochs 5 \
                   --arc_scale 30 --arc_margin 0.3

Arguments:
  --data       Path to training folder (identity subfolders)
  --model      Model type: hybrid_concat | hybrid_weighted | hybrid_attention | resnet50 | vit
  --epochs     Number of epochs (default: 20)
  --batch      Batch size (default: 16, keep low on CPU)
  --lr         Learning rate (default: 0.001)
  --embed      Embedding dimension (default: 512)
  --max_id     Limit number of identities (useful for quick testing)
  --resume     Path to checkpoint to resume from
  --save_dir   Where to save checkpoints (default: checkpoints/)

  --vit_lr_mult       ViT backbone LR = lr * this (default 0.02, lower than CNN
                      since ViT is more sensitive to learning rate)
  --cnn_lr_mult       CNN backbone LR = lr * this (default 0.1)
  --arc_scale         ArcFace scale s (default 30.0, lower than the paper's 64.0
                      since that default assumes millions of training images)
  --arc_margin        ArcFace margin m (default 0.3, lower than the paper's 0.5
                      for the same small-dataset reason)
  --vit_input_size    Image size fed to the ViT branch only (default 112 = original
                      behavior; try 160 to reduce position-embedding distortion —
                      ResNet always still sees the image at its original size)
  --freeze_vit_epochs Number of initial epochs to keep the ViT branch frozen
                      (default 0 = no freezing, same as before). Freezing lets
                      ResNet + fusion stabilize first, and saves CPU time during
                      the frozen epochs since ViT skips its backward pass.
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_train_loader
from model import build_model


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',     default='data/train')
    p.add_argument('--model',    default='hybrid_concat',
                   choices=['hybrid_concat', 'hybrid_weighted', 'hybrid_attention',
                            'resnet50', 'vit'])
    p.add_argument('--epochs',   type=int, default=20)
    p.add_argument('--batch',    type=int, default=16)
    p.add_argument('--lr',       type=float, default=0.001)
    p.add_argument('--embed',    type=int, default=512)
    p.add_argument('--max_id',   type=int, default=None,
                   help='Max identities to use (None = all)')
    p.add_argument('--resume',   default=None)
    p.add_argument('--save_dir', default='checkpoints')

    # ── NEW arguments for improving the hybrid model ──
    p.add_argument('--vit_lr_mult', type=float, default=0.02,
                   help='ViT backbone LR = lr * this (lower than CNN since ViT is more sensitive)')
    p.add_argument('--cnn_lr_mult', type=float, default=0.1,
                   help='CNN backbone LR = lr * this')
    p.add_argument('--arc_scale',  type=float, default=30.0,
                   help='ArcFace scale (lower than the default 64 for small identity counts)')
    p.add_argument('--arc_margin', type=float, default=0.3,
                   help='ArcFace margin (lower than the default 0.5 for small identity counts)')
    p.add_argument('--vit_input_size', type=int, default=112,
                   help='Image size fed to the ViT branch (try 160 for less position-embedding distortion)')
    p.add_argument('--freeze_vit_epochs', type=int, default=0,
                   help='Number of initial epochs to freeze the ViT branch (0 = no freezing)')
    return p.parse_args()


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()

        # Gradient clipping (helps stability on CPU)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_samples += imgs.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            avg = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {avg:.4f}")

    return total_loss / len(loader)


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss': loss,
        'args': vars(args),
    }
    path = os.path.join(save_dir, f"{args.model}_epoch{epoch:03d}.pth")
    torch.save(ckpt, path)

    # Also save as "latest"
    latest = os.path.join(save_dir, f"{args.model}_latest.pth")
    torch.save(ckpt, latest)
    print(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    print(f"  Resumed from epoch {ckpt['epoch']} | Loss: {ckpt['loss']:.4f}")
    return ckpt['epoch']


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cpu')
    print(f"\n{'='*55}")
    print(f"  Phase II Training — {args.model.upper()}")
    print(f"  Device  : CPU")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"  ViT input size: {args.vit_input_size}  |  Freeze ViT epochs: {args.freeze_vit_epochs}")
    print(f"  ArcFace scale: {args.arc_scale}  |  ArcFace margin: {args.arc_margin}")
    print(f"{'='*55}\n")

    # ── Dataset ──
    loader, num_classes = get_train_loader(
        args.data,
        batch_size=args.batch,
        num_workers=0,               # 0 for CPU compatibility
        max_identities=args.max_id
    )

    # ── Model ──
    model = build_model(args.model, num_classes=num_classes,
                        embed_dim=args.embed, pretrained=True,
                        arcface_scale=args.arc_scale, arcface_margin=args.arc_margin,
                        vit_input_size=args.vit_input_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {total_params:.1f}M\n")

    # ── ViT freeze schedule ──
    # Freezing ViT for the first few epochs lets ResNet + fusion adjust first,
    # avoiding noisy early gradients disturbing ViT's pretrained features.
    # It also saves CPU time, since frozen params skip the backward pass.
    if hasattr(model, 'vit') and args.freeze_vit_epochs > 0:
        for p in model.vit.parameters():
            p.requires_grad = False
        print(f"  ViT branch FROZEN for the first {args.freeze_vit_epochs} epoch(s)\n")

    # ── Optimizer & Scheduler ──
    # CNN and ViT now get DIFFERENT learning rates (ViT is more sensitive, needs to move slower)
    if hasattr(model, 'cnn'):
        cnn_params = list(model.cnn.parameters())
        vit_params = list(model.vit.parameters())
        backbone_params = cnn_params + vit_params
    else:
        # Single-backbone baselines (resnet50 or vit alone)
        cnn_params = list(model.backbone.parameters())
        vit_params = []
        backbone_params = cnn_params

    head_params = [p for p in model.parameters()
                   if not any(p is bp for bp in backbone_params)]

    param_groups = [{'params': cnn_params, 'lr': args.lr * args.cnn_lr_mult, 'name': 'cnn'}]
    if vit_params:
        param_groups.append({'params': vit_params, 'lr': args.lr * args.vit_lr_mult, 'name': 'vit'})
    param_groups.append({'params': head_params, 'lr': args.lr, 'name': 'head'})

    optimizer = Adam(param_groups, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ──
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler) + 1

    # ── Training Log ──
    log = {'model': args.model, 'epochs': [], 'losses': []}
    log_path = os.path.join(args.save_dir, f"{args.model}_log.json")

    # ── Training Loop ──
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Unfreeze ViT once we pass the freeze window
        if hasattr(model, 'vit') and args.freeze_vit_epochs > 0 and epoch == args.freeze_vit_epochs + 1:
            for p in model.vit.parameters():
                p.requires_grad = True
            print(f"\n  ── Unfreezing ViT branch at epoch {epoch} ──")

        print(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────")

        loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[-1]['lr']   # head LR — always last group regardless of count
        print(f"  Loss: {loss:.4f} | LR: {lr_now:.6f} | Time: {elapsed:.1f}s")

        # Save best
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'loss': loss}, best_path)
            print(f"  ★ New best saved → {best_path}")

        # Save every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss, args, args.save_dir)

        log['epochs'].append(epoch)
        log['losses'].append(round(loss, 4))

    # Always save final checkpoint
    save_checkpoint(model, optimizer, scheduler, args.epochs, loss, args, args.save_dir)

    # Save log
    os.makedirs(args.save_dir, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n  Training log saved → {log_path}")

    print(f"\n{'='*55}")
    print(f"  Training complete! Best loss: {best_loss:.4f}")
    print(f"  Best model: checkpoints/{args.model}_best.pth")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()