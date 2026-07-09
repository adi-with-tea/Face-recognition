"""
robustness_analysis.py — Robustness Analysis for Phase II
Tests the hybrid model under challenging conditions by applying
artificial degradations to LFW test pairs.

Conditions tested:
  1. Clean (baseline)
  2. Low resolution (downscale + upscale)
  3. Low illumination (darken)
  4. Gaussian blur (simulates motion/defocus)
  5. Grayscale (color removed)
  6. Noise (simulate low-quality camera)

Usage:
  python robustness_analysis.py --checkpoint checkpoints/hybrid_weighted_best.pth
                                --model hybrid_weighted
                                --lfw_dir data/lfw
                                --lfw_pairs data/matchpairsDevTest.csv
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default='checkpoints/hybrid_weighted_best.pth')
    p.add_argument('--model',       default='hybrid_weighted')
    p.add_argument('--num_classes', type=int, default=30)
    p.add_argument('--embed',       type=int, default=512)
    p.add_argument('--lfw_dir',     default='data/lfw')
    p.add_argument('--lfw_pairs',   default='data/matchpairsDevTest.csv')
    p.add_argument('--out_dir',     default='robustness_outputs')
    p.add_argument('--max_pairs',   type=int, default=200,
                   help='Limit pairs for speed (200 is enough)')
    return p.parse_args()


# ─── Load pairs ──────────────────────────────────────────────────────────────

def load_pairs(pairs_csv, lfw_dir, max_pairs=200):
    pairs = []
    mismatch_csv = pairs_csv.replace('matchpairs', 'mismatchpairs')

    with open(pairs_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            name, n1, n2 = row[0].strip(), row[1].strip(), row[2].strip()
            p1 = os.path.join(lfw_dir, name, f"{name}_{int(n1):04d}.jpg")
            p2 = os.path.join(lfw_dir, name, f"{name}_{int(n2):04d}.jpg")
            if os.path.exists(p1) and os.path.exists(p2):
                pairs.append((p1, p2, 1))

    if os.path.exists(mismatch_csv):
        with open(mismatch_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                n1, i1, n2, i2 = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                p1 = os.path.join(lfw_dir, n1, f"{n1}_{int(i1):04d}.jpg")
                p2 = os.path.join(lfw_dir, n2, f"{n2}_{int(i2):04d}.jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 0))

    # Balance and limit
    pos = [p for p in pairs if p[2] == 1][:max_pairs//2]
    neg = [p for p in pairs if p[2] == 0][:max_pairs//2]
    pairs = pos + neg
    print(f"  Loaded {len(pairs)} pairs ({len(pos)} pos, {len(neg)} neg)")
    return pairs


# ─── Degradation functions ───────────────────────────────────────────────────

def apply_degradation(img_pil, mode):
    """Apply a degradation to a PIL image and return PIL image."""
    if mode == 'clean':
        return img_pil

    elif mode == 'low_res':
        # Downscale to 28x28 then upscale back
        small = img_pil.resize((28, 28), Image.BILINEAR)
        return small.resize((112, 112), Image.BILINEAR)

    elif mode == 'low_illumination':
        enhancer = ImageEnhance.Brightness(img_pil)
        return enhancer.enhance(0.3)  # 30% brightness

    elif mode == 'blur':
        return img_pil.filter(ImageFilter.GaussianBlur(radius=3))

    elif mode == 'grayscale':
        gray = img_pil.convert('L')
        return gray.convert('RGB')  # back to 3-channel

    elif mode == 'noise':
        arr = np.array(img_pil).astype(np.float32)
        noise = np.random.normal(0, 30, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    return img_pil


# ─── Evaluation ──────────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


@torch.no_grad()
def evaluate_under_condition(model, pairs, mode, transform, device):
    model.eval()
    sims, labels = [], []

    for p1, p2, label in pairs:
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')

        img1 = apply_degradation(img1, mode)
        img2 = apply_degradation(img2, mode)

        t1 = transform(img1).unsqueeze(0).to(device)
        t2 = transform(img2).unsqueeze(0).to(device)

        e1 = model.extract(t1)
        e2 = model.extract(t2)
        sim = F.cosine_similarity(e1, e2).item()

        sims.append(sim)
        labels.append(label)

    sims   = np.array(sims)
    labels = np.array(labels)

    # Best threshold
    best_acc, best_t = 0, 0
    for t in np.arange(-1, 1, 0.02):
        preds = (sims >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    preds = (sims >= best_t).astype(int)
    f1  = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, sims)
    except:
        auc = float('nan')

    return {'accuracy': best_acc, 'f1': f1, 'auc': auc}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_robustness(results, out_path):
    conditions = list(results.keys())
    accs = [results[c]['accuracy'] for c in conditions]
    f1s  = [results[c]['f1']       for c in conditions]
    aucs = [results[c]['auc']      for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    axes[0].bar(x - width, accs, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    axes[0].bar(x,         f1s,  width, label='F1-Score', color='#DD8452', alpha=0.85)
    axes[0].bar(x + width, aucs, width, label='ROC-AUC',  color='#55A868', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Robustness Under Different Conditions')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=accs[0], color='red', linestyle='--', alpha=0.5, label='Clean baseline')

    # Line chart (degradation impact)
    axes[1].plot(conditions, accs, 'o-', color='#4C72B0', label='Accuracy', linewidth=2)
    axes[1].plot(conditions, f1s,  's-', color='#DD8452', label='F1-Score', linewidth=2)
    axes[1].plot(conditions, aucs, '^-', color='#55A868', label='ROC-AUC',  linewidth=2)
    axes[1].set_xticks(range(len(conditions)))
    axes[1].set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Performance Degradation Trend')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Robustness Analysis — Hybrid Weighted Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Chart saved → {out_path}")


def save_sample_degradations(pairs, out_dir):
    """Save visual examples of each degradation for the report."""
    sample_path = pairs[0][0]
    img = Image.open(sample_path).convert('RGB')
    modes = ['clean', 'low_res', 'low_illumination', 'blur', 'grayscale', 'noise']

    fig, axes = plt.subplots(1, len(modes), figsize=(18, 3))
    titles = ['Clean', 'Low Res\n(28px)', 'Low\nIllumination', 'Blur\n(Gaussian)', 'Grayscale', 'Noise\n(σ=30)']

    for ax, mode, title in zip(axes, modes, titles):
        degraded = apply_degradation(img.resize((112,112)), mode)
        ax.imshow(degraded)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.suptitle('Degradation Examples Used in Robustness Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    sample_out = os.path.join(out_dir, 'degradation_examples.png')
    plt.savefig(sample_out, dpi=150)
    print(f"  Sample degradations saved → {sample_out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')
    transform = get_transform()

    # Load model
    print(f"\nLoading {args.model}...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
    model = build_model(args.model, num_classes=num_cls,
                        embed_dim=args.embed, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Model loaded ✓")

    # Load pairs
    print(f"\nLoading pairs...")
    pairs = load_pairs(args.lfw_pairs, args.lfw_dir, args.max_pairs)

    # Save visual examples
    save_sample_degradations(pairs, args.out_dir)

    # Evaluate under each condition
    conditions = ['clean', 'low_res', 'low_illumination', 'blur', 'grayscale', 'noise']
    results = {}

    print(f"\nEvaluating under {len(conditions)} conditions...")
    print(f"{'Condition':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 55)

    for mode in conditions:
        r = evaluate_under_condition(model, pairs, mode, transform, device)
        results[mode] = r
        print(f"  {mode:<18} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")

    # Plot
    plot_robustness(results, os.path.join(args.out_dir, 'robustness_chart.png'))

    # Print summary
    print(f"\n{'='*55}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'='*55}")
    baseline_acc = results['clean']['accuracy']
    for mode, r in results.items():
        drop = baseline_acc - r['accuracy']
        print(f"  {mode:<20} Acc: {r['accuracy']:.4f}  Drop: {drop:+.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
