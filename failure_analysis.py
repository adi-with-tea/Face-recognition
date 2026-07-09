"""
failure_analysis.py — Failure Case Analysis for Phase II
Identifies and visualizes pairs where the model fails most severely.

Failure types:
  - False Positives: Different people predicted as same (sim > threshold but label=0)
  - False Negatives: Same person predicted as different (sim < threshold but label=1)

Usage:
  python failure_analysis.py --checkpoint checkpoints/hybrid_weighted_best.pth
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
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default='checkpoints/hybrid_weighted_best.pth')
    p.add_argument('--model',       default='hybrid_weighted')
    p.add_argument('--num_classes', type=int, default=30)
    p.add_argument('--embed',       type=int, default=512)
    p.add_argument('--lfw_dir',     default='data/lfw')
    p.add_argument('--lfw_pairs',   default='data/matchpairsDevTest.csv')
    p.add_argument('--out_dir',     default='failure_outputs')
    p.add_argument('--top_n',       type=int, default=5,
                   help='Number of worst failures to visualize per type')
    return p.parse_args()


def load_pairs(pairs_csv, lfw_dir):
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

    print(f"  Loaded {len(pairs)} pairs")
    return pairs


def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


@torch.no_grad()
def run_inference(model, pairs, transform, device):
    model.eval()
    results = []

    for p1, p2, label in pairs:
        img1 = transform(Image.open(p1).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(p2).convert('RGB')).unsqueeze(0).to(device)
        e1 = model.extract(img1)
        e2 = model.extract(img2)
        sim = F.cosine_similarity(e1, e2).item()
        results.append({'p1': p1, 'p2': p2, 'label': label, 'sim': sim})

    return results


def find_best_threshold(results):
    sims   = np.array([r['sim']   for r in results])
    labels = np.array([r['label'] for r in results])
    best_acc, best_t = 0, 0
    for t in np.arange(-1, 1, 0.02):
        preds = (sims >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def visualize_failures(failures, title, out_path, top_n=5):
    """Plot pairs of images with their similarity scores."""
    failures = failures[:top_n]
    if not failures:
        print(f"  No failures of type: {title}")
        return

    n = len(failures)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    if n == 1:
        axes = [axes]

    for i, f in enumerate(failures):
        img1 = Image.open(f['p1']).convert('RGB').resize((112, 112))
        img2 = Image.open(f['p2']).convert('RGB').resize((112, 112))

        name1 = os.path.basename(os.path.dirname(f['p1']))
        name2 = os.path.basename(os.path.dirname(f['p2']))

        axes[i][0].imshow(img1)
        axes[i][0].set_title(f"{name1}", fontsize=8)
        axes[i][0].axis('off')

        axes[i][1].imshow(img2)
        label_str = "Same Person" if f['label'] == 1 else "Different People"
        pred_str  = "MATCH" if f['pred'] == 1 else "NO MATCH"
        color = 'red'
        axes[i][1].set_title(
            f"{name2}\nTrue: {label_str}\nPred: {pred_str} (sim={f['sim']:.3f})",
            fontsize=8, color=color
        )
        axes[i][1].axis('off')

    plt.suptitle(title, fontsize=12, fontweight='bold', color='red')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  Saved → {out_path}")


def plot_similarity_distribution(results, threshold, out_path):
    """Plot similarity score distributions for same vs different pairs."""
    same_sims = [r['sim'] for r in results if r['label'] == 1]
    diff_sims = [r['sim'] for r in results if r['label'] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(same_sims, bins=30, alpha=0.7, color='#55A868', label='Same Person')
    axes[0].hist(diff_sims, bins=30, alpha=0.7, color='#C44E52', label='Different People')
    axes[0].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                    label=f'Threshold={threshold:.2f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Similarity Score Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Error analysis
    sims   = np.array([r['sim']   for r in results])
    labels = np.array([r['label'] for r in results])
    preds  = (sims >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    confusion = np.array([[tn, fp], [fn, tp]])
    im = axes[1].imshow(confusion, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Predicted\nDifferent', 'Predicted\nSame'])
    axes[1].set_yticklabels(['Actually\nDifferent', 'Actually\nSame'])
    axes[1].set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(confusion[i, j]),
                        ha='center', va='center', fontsize=16, fontweight='bold')

    plt.suptitle('Failure Analysis — Hybrid Weighted Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Distribution chart saved → {out_path}")


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

    # Load pairs
    print("\nLoading pairs...")
    pairs = load_pairs(args.lfw_pairs, args.lfw_dir)

    # Run inference
    print("\nRunning inference on all pairs...")
    results = run_inference(model, pairs, transform, device)

    # Find threshold
    threshold, acc = find_best_threshold(results)
    print(f"  Best threshold: {threshold:.2f} | Accuracy: {acc:.4f}")

    # Add predictions
    for r in results:
        r['pred'] = 1 if r['sim'] >= threshold else 0

    # Find failures
    false_positives = [r for r in results if r['pred'] == 1 and r['label'] == 0]
    false_negatives = [r for r in results if r['pred'] == 0 and r['label'] == 1]

    # Sort by severity
    false_positives.sort(key=lambda x: x['sim'], reverse=True)   # highest sim = most wrong
    false_negatives.sort(key=lambda x: x['sim'])                  # lowest sim = most wrong

    print(f"\n  False Positives (different people matched): {len(false_positives)}")
    print(f"  False Negatives (same person not matched):  {len(false_negatives)}")

    # Visualize worst failures
    visualize_failures(
        false_positives, 
        f"False Positives — Different People Incorrectly Matched (Top {args.top_n})",
        os.path.join(args.out_dir, 'false_positives.png'),
        args.top_n
    )

    visualize_failures(
        false_negatives,
        f"False Negatives — Same Person Not Recognized (Top {args.top_n})",
        os.path.join(args.out_dir, 'false_negatives.png'),
        args.top_n
    )

    # Similarity distribution + confusion matrix
    plot_similarity_distribution(
        results, threshold,
        os.path.join(args.out_dir, 'similarity_distribution.png')
    )

    # Summary
    total = len(results)
    print(f"\n{'='*55}")
    print(f"  FAILURE ANALYSIS SUMMARY")
    print(f"{'='*55}")
    print(f"  Total pairs evaluated : {total}")
    print(f"  Correct predictions   : {sum(1 for r in results if r['pred']==r['label'])}")
    print(f"  False Positives       : {len(false_positives)} ({100*len(false_positives)/total:.1f}%)")
    print(f"  False Negatives       : {len(false_negatives)} ({100*len(false_negatives)/total:.1f}%)")
    print(f"\n  Outputs saved to: {args.out_dir}/")
    print(f"    - false_positives.png")
    print(f"    - false_negatives.png")
    print(f"    - similarity_distribution.png")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
