"""
explainability.py — Grad-CAM + ViT Attention Maps
Visualizes which facial regions the model focuses on.

Usage:
  python explainability.py --image path/to/face.jpg \
                           --checkpoint checkpoints/hybrid_concat_best.pth \
                           --model hybrid_concat
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image',      required=True, help='Path to face image')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model',      default='hybrid_concat')
    p.add_argument('--num_classes',type=int, default=100)
    p.add_argument('--embed',      type=int, default=512)
    p.add_argument('--out_dir',    default='explainability_outputs')
    return p.parse_args()


def load_image(path):
    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img).unsqueeze(0), img


# ─── Grad-CAM (CNN branch) ────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM on the last conv layer of ResNet50.
    Reference: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_layer()

    def _hook_layer(self):
        # Target: last ResNet layer (layer4)
        target = self.model.cnn.features[-1]

        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, x):
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward through CNN branch only
        cnn_feat = self.model.cnn(x)
        score = cnn_feat.sum()   # proxy: sum of features
        self.model.zero_grad()
        score.backward()

        # Grad-CAM computation
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(112, 112), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()

        # Normalize
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def apply_heatmap(cam, original_img):
    """Overlay Grad-CAM heatmap on original PIL image."""
    img_np = np.array(original_img.resize((112, 112)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.uint8(overlay)


# ─── ViT Attention Map ────────────────────────────────────────────────────────

class ViTAttentionRollout:
    """
    Attention Rollout for ViT — visualizes which patches the model attends to.
    Reference: https://arxiv.org/abs/2005.00928
    """

    def __init__(self, model):
        self.model = model
        self.attentions = []
        self._hook_attention()

    def _hook_attention(self):
        def hook(module, input, output):
            # output of attention is (context, attn_weights) in some versions
            # timm ViT returns only context; we hook the softmax instead
            self.attentions.append(output.detach())

        # Hook every attention block's softmax output
        for block in self.model.vit.vit.blocks:
            block.attn.attn_drop.register_forward_hook(hook)

    def generate(self, x):
        self.attentions = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model.vit(x)

        if not self.attentions:
            print("[Warning] No attention maps captured. ViT version may differ.")
            return None

        # Rollout: chain attention matrices through layers
        result = torch.eye(self.attentions[0].size(-1))
        for attn in self.attentions:
            # attn shape: (B, heads, tokens, tokens)
            attn_avg = attn.mean(dim=1)[0]   # average over heads: (tokens, tokens)
            # Add residual and normalize
            attn_aug = attn_avg + torch.eye(attn_avg.size(-1))
            attn_aug /= attn_aug.sum(dim=-1, keepdim=True)
            result = torch.mm(attn_aug, result)

        # CLS token attends to all patches (row 0, cols 1:)
        mask = result[0, 1:]

        # Reshape: 112/16 = 7 → 7x7 grid
        grid_size = int(mask.size(0) ** 0.5)
        mask = mask.reshape(grid_size, grid_size).numpy()
        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        # Upsample to 112x112
        mask = cv2.resize(mask, (112, 112))
        return mask


# ─── Main Visualization ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')

    # Load image
    tensor, orig_img = load_image(args.image)
    tensor = tensor.to(device)
    img_name = os.path.splitext(os.path.basename(args.image))[0]

    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
    model = build_model(args.model, num_classes=num_cls,
                        embed_dim=args.embed, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    print(f"\nGenerating explanations for: {args.image}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    orig_np = np.array(orig_img.resize((112, 112)))

    # ── Original ──
    axes[0].imshow(orig_np)
    axes[0].set_title('Original Face')
    axes[0].axis('off')

    # ── Grad-CAM ──
    print("  Computing Grad-CAM (CNN branch)...")
    try:
        gradcam = GradCAM(model)
        cam = gradcam.generate(tensor)
        overlay = apply_heatmap(cam, orig_img)
        axes[1].imshow(overlay)
        axes[1].set_title('Grad-CAM\n(CNN / ResNet50 branch)')
        axes[1].axis('off')

        # Save raw cam
        cam_path = os.path.join(args.out_dir, f'{img_name}_gradcam.png')
        plt.imsave(cam_path, overlay)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'GradCAM error:\n{e}',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
        print(f"  [Warning] Grad-CAM failed: {e}")

    # ── ViT Attention ──
    print("  Computing ViT attention rollout...")
    try:
        rollout = ViTAttentionRollout(model)
        mask = rollout.generate(tensor)
        if mask is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_VIRIDIS)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            attn_overlay = np.uint8(0.5 * orig_np + 0.5 * heatmap)
            axes[2].imshow(attn_overlay)
        else:
            axes[2].imshow(orig_np)
        axes[2].set_title('Attention Rollout\n(Transformer / ViT branch)')
        axes[2].axis('off')
    except Exception as e:
        axes[2].text(0.5, 0.5, f'Attention error:\n{e}',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
        print(f"  [Warning] ViT attention failed: {e}")

    plt.suptitle(f'Explainability Analysis — {args.model}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f'{img_name}_explain.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Visualization saved → {out_path}")


if __name__ == "__main__":
    main()
