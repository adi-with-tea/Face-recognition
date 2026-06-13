"""
baselines.py — Baseline Model Evaluation
Evaluates FaceNet (via facenet-pytorch) on the same benchmarks
so you can compare against your hybrid model.

Usage:
  python baselines.py --lfw_dir data/lfw --lfw_pairs data/lfw_pairs.txt
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_OK = True
except ImportError:
    FACENET_OK = False
    print("[Warning] facenet-pytorch not installed: pip install facenet-pytorch")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lfw_dir',    default='data/lfw')
    p.add_argument('--lfw_pairs',  default='data/lfw_pairs.txt')
    p.add_argument('--cfp_dir',    default=None)
    p.add_argument('--cfp_pairs',  default=None)
    p.add_argument('--agedb_dir',  default=None)
    p.add_argument('--agedb_pairs',default=None)
    return p.parse_args()


def get_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),   # FaceNet expects 160x160
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_lfw_pairs(lfw_dir, pairs_file):
    pairs = []
    transform = get_transform()
    with open(pairs_file) as f:
        lines = f.read().strip().split('\n')[1:]   # skip header
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            name, n1, n2 = parts
            p1 = os.path.join(lfw_dir, name, f"{name}_{int(n1):04d}.jpg")
            p2 = os.path.join(lfw_dir, name, f"{name}_{int(n2):04d}.jpg")
            label = 1
        elif len(parts) == 4:
            name1, n1, name2, n2 = parts
            p1 = os.path.join(lfw_dir, name1, f"{name1}_{int(n1):04d}.jpg")
            p2 = os.path.join(lfw_dir, name2, f"{name2}_{int(n2):04d}.jpg")
            label = 0
        else:
            continue
        if os.path.exists(p1) and os.path.exists(p2):
            pairs.append((p1, p2, label))
    return pairs, transform


@torch.no_grad()
def evaluate_facenet(model, pairs, transform, device):
    model.eval()
    sims, labels_all, times = [], [], []

    for p1, p2, label in pairs:
        img1 = transform(Image.open(p1).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(p2).convert('RGB')).unsqueeze(0).to(device)

        t0 = time.time()
        e1 = model(img1)
        e2 = model(img2)
        times.append(time.time() - t0)

        sim = F.cosine_similarity(e1, e2).item()
        sims.append(sim)
        labels_all.append(label)

    sims   = np.array(sims)
    labels = np.array(labels_all)

    # Find best threshold
    best_acc, best_t = 0, 0
    for t in np.arange(-1, 1, 0.01):
        preds = (sims >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    preds = (sims >= best_t).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, sims)
    except Exception:
        auc = float('nan')

    avg_time = np.mean(times)
    return {
        'accuracy':     best_acc,
        'precision':    prec,
        'recall':       rec,
        'f1':           f1,
        'auc':          auc,
        'fps':          1.0 / avg_time,
        'inference_ms': avg_time * 1000,
    }


def main():
    args = parse_args()
    device = torch.device('cpu')

    print("\n" + "="*55)
    print("  FaceNet Baseline Evaluation")
    print("="*55)

    if not FACENET_OK:
        print("ERROR: Install facenet-pytorch first:\n  pip install facenet-pytorch")
        return

    # Load FaceNet pretrained on VGGFace2
    print("Loading FaceNet (VGGFace2 pretrained)...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # FaceNet model size
    tmp = '/tmp/_facenet.pth'
    torch.save(facenet.state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    print(f"Model size: {size_mb:.1f} MB")

    results = {}

    # ── LFW ──
    if os.path.exists(args.lfw_dir) and os.path.exists(args.lfw_pairs):
        print("\nEvaluating on LFW...")
        pairs, transform = load_lfw_pairs(args.lfw_dir, args.lfw_pairs)
        print(f"  {len(pairs)} pairs loaded")
        m = evaluate_facenet(facenet, pairs, transform, device)
        results['lfw'] = m
        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  ROC-AUC   : {m['auc']:.4f}")
        print(f"  FPS       : {m['fps']:.2f}")
        print(f"  Inf. Time : {m['inference_ms']:.2f} ms")

    print("\n" + "="*55)
    print("  Copy these numbers into your comparison table!")
    print("="*55)
    import json
    with open('checkpoints/facenet_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved → checkpoints/facenet_baseline.json")


if __name__ == "__main__":
    main()
