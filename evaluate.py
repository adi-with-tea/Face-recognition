"""
evaluate.py — Phase II Evaluation Script
Evaluates any trained model on LFW, CFP-FP, AgeDB-30.

Metrics computed:
  - Verification Accuracy (threshold-optimized)
  - Precision, Recall, F1-Score
  - ROC-AUC
  - Inference Time & FPS
  - Model size (MB)

Usage:
  python evaluate.py --checkpoint checkpoints/hybrid_concat_best.pth \
                     --model hybrid_concat \
                     --lfw_dir data/lfw \
                     --lfw_pairs data/lfw_pairs.txt

  # Evaluate all checkpoints and compare:
  python evaluate.py --compare --lfw_dir data/lfw --lfw_pairs data/lfw_pairs.txt
"""

import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, precision_score,
                              recall_score, f1_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless-safe

from dataset import get_lfw_loader, get_cfp_loader, get_agedb_loader
from model import build_model


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--model',      default='hybrid_concat')
    p.add_argument('--num_classes',type=int, default=100,
                   help='Must match training (only needed to build model skeleton)')
    p.add_argument('--embed',      type=int, default=512)
    p.add_argument('--lfw_dir',    default='data/lfw')
    p.add_argument('--lfw_pairs',  default='data/lfw_pairs.txt')
    p.add_argument('--cfp_dir',    default=None)
    p.add_argument('--cfp_pairs',  default=None)
    p.add_argument('--agedb_dir',  default=None)
    p.add_argument('--agedb_pairs',default=None)
    p.add_argument('--batch',      type=int, default=16)
    p.add_argument('--compare',    action='store_true',
                   help='Load all checkpoints from checkpoints/ and compare')
    p.add_argument('--save_dir',   default='checkpoints')
    return p.parse_args()


# ─── Core Evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_pair_similarities(model, loader, device):
    """
    Pass all pairs through model, compute cosine similarity.
    Returns: similarities (np array), labels (np array), avg_time (s)
    """
    model.eval()
    sims, labels_all, times = [], [], []

    for img1, img2, label in loader:
        img1, img2 = img1.to(device), img2.to(device)

        t0 = time.time()
        e1 = model.extract(img1)
        e2 = model.extract(img2)
        times.append((time.time() - t0) / img1.size(0))

        sim = F.cosine_similarity(e1, e2).cpu().numpy()
        sims.extend(sim.tolist())
        labels_all.extend(label.numpy().tolist())

    return np.array(sims), np.array(labels_all), np.mean(times)


def find_best_threshold(sims, labels):
    """Grid-search threshold on [−1, 1] to maximize accuracy."""
    best_acc, best_thresh = 0.0, 0.0
    for t in np.arange(-1.0, 1.0, 0.01):
        preds = (sims >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_thresh, best_acc


def compute_all_metrics(sims, labels, threshold):
    preds = (sims >= threshold).astype(int)
    acc  = (preds == labels).mean()
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, sims)
    except Exception:
        auc = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def model_size_mb(model):
    tmp = '/tmp/_tmp_model.pth'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return size


# ─── Evaluate One Model ───────────────────────────────────────────────────────

def evaluate_model(model, args, device, tag=''):
    results = {}

    # ── LFW ──
    if os.path.exists(args.lfw_dir) and os.path.exists(args.lfw_pairs):
        print(f"\n  [{tag}] LFW evaluation...")
        loader = get_lfw_loader(args.lfw_dir, args.lfw_pairs, batch_size=args.batch)
        sims, labels, avg_time = extract_pair_similarities(model, loader, device)
        thresh, _ = find_best_threshold(sims, labels)
        metrics = compute_all_metrics(sims, labels, thresh)
        metrics['inference_ms'] = avg_time * 1000
        metrics['fps'] = 1.0 / avg_time
        results['lfw'] = metrics
        print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} "
              f"| AUC: {metrics['auc']:.4f} | {metrics['fps']:.1f} FPS")

    # ── CFP-FP ──
    if args.cfp_dir and args.cfp_pairs and os.path.exists(args.cfp_dir):
        print(f"  [{tag}] CFP-FP evaluation...")
        loader = get_cfp_loader(args.cfp_dir, args.cfp_pairs, batch_size=args.batch)
        sims, labels, avg_time = extract_pair_similarities(model, loader, device)
        thresh, _ = find_best_threshold(sims, labels)
        metrics = compute_all_metrics(sims, labels, thresh)
        metrics['inference_ms'] = avg_time * 1000
        results['cfp_fp'] = metrics
        print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

    # ── AgeDB-30 ──
    if args.agedb_dir and args.agedb_pairs and os.path.exists(args.agedb_dir):
        print(f"  [{tag}] AgeDB-30 evaluation...")
        loader = get_agedb_loader(args.agedb_dir, args.agedb_pairs, batch_size=args.batch)
        sims, labels, avg_time = extract_pair_similarities(model, loader, device)
        thresh, _ = find_best_threshold(sims, labels)
        metrics = compute_all_metrics(sims, labels, thresh)
        metrics['inference_ms'] = avg_time * 1000
        results['agedb_30'] = metrics
        print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

    results['model_mb'] = model_size_mb(model)
    return results


# ─── Comparison Plot ──────────────────────────────────────────────────────────

def plot_comparison(all_results, save_path='evaluation_phase2.png'):
    models = list(all_results.keys())
    benchmarks = ['lfw', 'cfp_fp', 'agedb_30']
    metrics = ['accuracy', 'f1', 'auc']
    labels = ['Accuracy', 'F1-Score', 'ROC-AUC']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    available = [b for b in benchmarks
                 if any(b in all_results[m] for m in models)]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, bench in zip(axes, available):
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        for i, (mname, color) in enumerate(zip(models, colors)):
            if bench not in all_results[mname]:
                continue
            vals = [all_results[mname][bench].get(m, 0) for m in metrics]
            ax.bar(x + i * width, vals, width, label=mname, color=color, alpha=0.85)

        ax.set_title(bench.upper().replace('_', '-'))
        ax.set_xticks(x + width * len(models) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Phase II: Model Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Comparison chart saved → {save_path}")


def print_summary_table(all_results, benchmark='lfw'):
    print(f"\n{'='*70}")
    print(f"  PHASE II EVALUATION SUMMARY — {benchmark.upper()}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'FPS':>8} {'MB':>8}")
    print(f"{'-'*70}")
    for mname, res in all_results.items():
        if benchmark not in res:
            continue
        m = res[benchmark]
        mb = res.get('model_mb', 0)
        fps = m.get('fps', 0)
        print(f"{mname:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {fps:>8.1f} {mb:>8.1f}")
    print(f"{'='*70}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cpu')

    if args.compare:
        # Load all checkpoints from save_dir and compare
        ckpt_dir = args.save_dir
        all_results = {}

        for fname in sorted(os.listdir(ckpt_dir)):
            if not fname.endswith('_best.pth'):
                continue
            model_type = fname.replace('_best.pth', '')
            ckpt_path = os.path.join(ckpt_dir, fname)
            print(f"\nLoading {model_type} from {ckpt_path}...")

            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
                model = build_model(model_type, num_classes=num_cls,
                                    embed_dim=args.embed, pretrained=False)
                model.load_state_dict(ckpt['model_state'])
                model = model.to(device)
                all_results[model_type] = evaluate_model(model, args, device, tag=model_type)
            except Exception as e:
                print(f"  [Skip] {model_type}: {e}")

        if all_results:
            print_summary_table(all_results, 'lfw')
            plot_comparison(all_results)
            # Save JSON
            with open(os.path.join(ckpt_dir, 'comparison_results.json'), 'w') as f:
                json.dump({k: {bk: {mk: float(mv) for mk, mv in bv.items()}
                               for bk, bv in v.items()}
                           for k, v in all_results.items()}, f, indent=2)
        return

    # Single model evaluation
    if not args.checkpoint:
        print("ERROR: Provide --checkpoint path or use --compare")
        return

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
    model = build_model(args.model, num_classes=num_cls,
                        embed_dim=args.embed, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)

    print(f"\nEvaluating {args.model} | {model_size_mb(model):.1f} MB")
    results = evaluate_model(model, args, device, tag=args.model)

    # Save
    out = os.path.join(args.save_dir, f"{args.model}_eval.json")
    with open(out, 'w') as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()
