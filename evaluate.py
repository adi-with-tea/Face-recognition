"""
evaluate.py — Phase II Evaluation (Kaggle LFW CSV format)
"""

import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import get_lfw_loader
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--model',      default='hybrid_concat')
    p.add_argument('--num_classes',type=int, default=30)
    p.add_argument('--embed',      type=int, default=512)
    p.add_argument('--lfw_dir',    default='data/lfw')
    p.add_argument('--lfw_pairs',  default='data/matchpairsDevTest.csv')
    p.add_argument('--lfw_mismatch', default='data/mismatchpairsDevTest.csv')
    p.add_argument('--batch',      type=int, default=16)
    p.add_argument('--compare',    action='store_true')
    p.add_argument('--save_dir',   default='checkpoints')
    return p.parse_args()


@torch.no_grad()
def extract_pair_similarities(model, loader, device):
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
    best_acc, best_thresh = 0.0, 0.0
    for t in np.arange(-1.0, 1.0, 0.01):
        preds = (sims >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_thresh, best_acc


def compute_metrics(sims, labels, threshold):
    preds = (sims >= threshold).astype(int)
    acc  = (preds == labels).mean()
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, sims)
    except:
        auc = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def model_size_mb(model):
    tmp = '/tmp/_tmp_model.pth'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return size


def plot_comparison(all_results, save_path='evaluation_phase2.png'):
    models  = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Metrics bar chart
    for i, (mname, color) in enumerate(zip(models, colors)):
        if 'lfw' not in all_results[mname]:
            continue
        vals = [all_results[mname]['lfw'].get(m, 0) for m in metrics]
        axes[0].bar(x + i * width, vals, width, label=mname, color=color, alpha=0.85)

    axes[0].set_title('Detection Metrics Comparison')
    axes[0].set_xticks(x + width * len(models) / 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=7)
    axes[0].grid(axis='y', alpha=0.3)

    # FPS comparison
    model_names = []
    fps_vals = []
    for mname in models:
        if 'lfw' in all_results[mname]:
            model_names.append(mname)
            fps_vals.append(all_results[mname]['lfw'].get('fps', 0))

    axes[1].bar(model_names, fps_vals, color=colors[:len(model_names)], alpha=0.85)
    axes[1].set_title('Inference Speed (FPS)')
    axes[1].set_ylabel('FPS')
    axes[1].grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)

    plt.suptitle('Phase II: Model Comparison on LFW', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Chart saved → {save_path}")


def print_table(all_results):
    print(f"\n{'='*75}")
    print(f"  PHASE II EVALUATION — LFW BENCHMARK")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'FPS':>7} {'MB':>7}")
    print(f"{'-'*75}")
    for mname, res in all_results.items():
        if 'lfw' not in res:
            continue
        m  = res['lfw']
        mb = res.get('model_mb', 0)
        print(f"{mname:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} "
              f"{m.get('fps',0):>7.1f} {mb:>7.1f}")
    print(f"{'='*75}\n")


def main():
    args = parse_args()
    device = torch.device('cpu')

    if args.compare:
        all_results = {}
        for fname in sorted(os.listdir(args.save_dir)):
            if not fname.endswith('_best.pth'):
                continue
            model_type = fname.replace('_best.pth', '')
            ckpt_path  = os.path.join(args.save_dir, fname)
            print(f"\nEvaluating {model_type}...")
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
                model = build_model(model_type, num_classes=num_cls,
                                    embed_dim=args.embed, pretrained=False)
                model.load_state_dict(ckpt['model_state'])
                model.eval()

                loader = get_lfw_loader(
                    args.lfw_dir,
                    args.lfw_pairs,
                    args.lfw_mismatch,
                    batch_size=args.batch
                )
                sims, labels, avg_time = extract_pair_similarities(model, loader, device)
                thresh, _ = find_best_threshold(sims, labels)
                metrics = compute_metrics(sims, labels, thresh)
                metrics['fps'] = 1.0 / avg_time
                metrics['inference_ms'] = avg_time * 1000

                all_results[model_type] = {
                    'lfw': metrics,
                    'model_mb': model_size_mb(model)
                }
                print(f"  Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f} | FPS: {metrics['fps']:.2f}")

            except Exception as e:
                print(f"  [Skip] {model_type}: {e}")

        if all_results:
            print_table(all_results)
            plot_comparison(all_results)
            out = os.path.join(args.save_dir, 'comparison_results.json')
            with open(out, 'w') as f:
                json.dump({k: {bk: {mk: float(mv) for mk, mv in bv.items()}
                               for bk, bv in v.items()}
                           for k, v in all_results.items()}, f, indent=2)
            print(f"  Results saved → {out}")
        return


if __name__ == "__main__":
    main()