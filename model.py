"""
model.py — Hybrid ResNet50 + ViT-B/16 Face Recognition Model

Architecture:
  Input (112x112 RGB)
       ├── ResNet50 → 2048-d CNN features
       └── ViT-B/16 → 768-d Transformer features
            │
       Feature Fusion (3 modes: concat | weighted | attention)
            │
       Embedding Layer → 512-d L2-normalized embedding
            │
       ArcFace Loss (during training)

Usage:
  model = HybridFaceNet(num_classes=8631, fusion='concat')
  embeddings = model(x)                  # inference
  loss = model(x, labels)               # training (returns loss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[Warning] timm not installed. Run: pip install timm")


# ─── ArcFace Loss ─────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Face Recognition
    Paper: https://arxiv.org/abs/1801.07698

    Adds an angular margin m to the target class angle to
    make the decision boundary more discriminative.
    """

    def __init__(self, in_features, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-6)

        # cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Prevent theta + margin > pi (numerical safety)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encode labels and apply margin only to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale

        return F.cross_entropy(output, labels)


# ─── Fusion Modules ───────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """Simple concatenation of CNN and Transformer features."""

    def __init__(self, cnn_dim, vit_dim, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, cnn_feat, vit_feat):
        x = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.fc(x)


class WeightedFusion(nn.Module):
    """
    Learnable scalar weights for CNN and ViT branches.
    alpha * cnn + (1 - alpha) * vit, where alpha is learned.
    Both branches projected to same dimension first.
    """

    def __init__(self, cnn_dim, vit_dim, out_dim=512):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, out_dim)
        self.vit_proj = nn.Linear(vit_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))   # learnable weight
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, cnn_feat, vit_feat):
        alpha = torch.sigmoid(self.alpha)   # keep in [0, 1]
        c = self.cnn_proj(cnn_feat)
        v = self.vit_proj(vit_feat)
        return self.bn(alpha * c + (1 - alpha) * v)


class AttentionFusion(nn.Module):
    """
    Cross-attention fusion: ViT features attend over CNN features.
    Projects both to a shared space, computes attention weights,
    then fuses with a residual connection.
    """

    def __init__(self, cnn_dim, vit_dim, out_dim=512, num_heads=8):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, out_dim)
        self.vit_proj = nn.Linear(vit_dim, out_dim)

        # Multi-head attention: query from ViT, key/value from CNN
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, cnn_feat, vit_feat):
        c = self.cnn_proj(cnn_feat).unsqueeze(1)   # (B, 1, D)
        v = self.vit_proj(vit_feat).unsqueeze(1)   # (B, 1, D)

        # ViT queries CNN context
        attended, _ = self.attn(query=v, key=c, value=c)
        fused = self.norm(attended + v).squeeze(1)  # residual
        return self.bn(self.drop(fused))


# ─── Backbone Wrappers ────────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """ResNet50 pretrained on ImageNet, strips the classifier head."""

    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # Remove avgpool and fc; keep feature maps
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)   # (B, 2048)


class ViTBackbone(nn.Module):
    """ViT-B/16 pretrained on ImageNet21k via timm."""

    def __init__(self, pretrained=True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required: pip install timm")
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,       # remove classification head → returns [CLS] token
            img_size=112,        # our face images are 112x112
        )
        self.out_dim = self.vit.num_features   # 768

    def forward(self, x):
        return self.vit(x)   # (B, 768)


# ─── Main Hybrid Model ────────────────────────────────────────────────────────

class HybridFaceNet(nn.Module):
    """
    Hybrid ResNet50 + ViT-B/16 Face Recognition Network.

    Args:
        num_classes  : Number of training identities (for ArcFace)
        fusion       : 'concat' | 'weighted' | 'attention'
        embed_dim    : Output embedding dimension (default 512)
        pretrained   : Load ImageNet pretrained weights for backbones
        arcface_scale: ArcFace scale parameter s
        arcface_margin: ArcFace angular margin m (radians)
    """

    def __init__(self, num_classes, fusion='concat', embed_dim=512,
                 pretrained=True, arcface_scale=64.0, arcface_margin=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion

        # Backbones
        self.cnn = ResNet50Backbone(pretrained=pretrained)
        self.vit = ViTBackbone(pretrained=pretrained)

        # Fusion module
        cnn_dim = self.cnn.out_dim   # 2048
        vit_dim = self.vit.out_dim   # 768

        if fusion == 'concat':
            self.fusion = ConcatFusion(cnn_dim, vit_dim, embed_dim)
        elif fusion == 'weighted':
            self.fusion = WeightedFusion(cnn_dim, vit_dim, embed_dim)
        elif fusion == 'attention':
            self.fusion = AttentionFusion(cnn_dim, vit_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion: {fusion}. Choose: concat | weighted | attention")

        # ArcFace loss head
        self.arcface = ArcFaceLoss(embed_dim, num_classes, arcface_scale, arcface_margin)

        print(f"[HybridFaceNet] fusion={fusion} | embed={embed_dim} | classes={num_classes}")

    def extract(self, x):
        """Returns L2-normalized embeddings (use for inference/evaluation)."""
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        emb = self.fusion(cnn_feat, vit_feat)
        return F.normalize(emb, dim=1)   # unit-norm embedding

    def forward(self, x, labels=None):
        """
        Training: forward(x, labels) → ArcFace loss (scalar)
        Inference: forward(x)        → L2-normalized embeddings
        """
        emb = self.extract(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return emb


# ─── Baseline Models ──────────────────────────────────────────────────────────

class ResNet50ArcFace(nn.Module):
    """ArcFace with ResNet50 only (baseline)."""

    def __init__(self, num_classes, embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.arcface = ArcFaceLoss(embed_dim, num_classes)

    def extract(self, x):
        return F.normalize(self.fc(self.backbone(x)), dim=1)

    def forward(self, x, labels=None):
        emb = self.extract(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return emb


class ViTArcFace(nn.Module):
    """ArcFace with ViT-B/16 only (baseline)."""

    def __init__(self, num_classes, embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = ViTBackbone(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.arcface = ArcFaceLoss(embed_dim, num_classes)

    def extract(self, x):
        return F.normalize(self.fc(self.backbone(x)), dim=1)

    def forward(self, x, labels=None):
        emb = self.extract(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return emb


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(model_type, num_classes, embed_dim=512, pretrained=True):
    """
    model_type: 'hybrid_concat' | 'hybrid_weighted' | 'hybrid_attention'
                'resnet50'      | 'vit'
    """
    if model_type.startswith('hybrid'):
        fusion = model_type.split('_')[1]   # concat / weighted / attention
        return HybridFaceNet(num_classes, fusion=fusion,
                             embed_dim=embed_dim, pretrained=pretrained)
    elif model_type == 'resnet50':
        return ResNet50ArcFace(num_classes, embed_dim=embed_dim, pretrained=pretrained)
    elif model_type == 'vit':
        return ViTArcFace(num_classes, embed_dim=embed_dim, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HybridFaceNet (concat)...")
    model = HybridFaceNet(num_classes=100, fusion='concat', pretrained=False)
    x = torch.randn(2, 3, 112, 112)
    labels = torch.tensor([0, 1])

    # Training mode
    loss = model(x, labels)
    print(f"  Training loss: {loss.item():.4f}")

    # Inference mode
    emb = model(x)
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Embedding norms: {emb.norm(dim=1)}")  # should be ~1.0

    print("\nAll 3 fusion variants:")
    for fusion in ['concat', 'weighted', 'attention']:
        m = HybridFaceNet(num_classes=100, fusion=fusion, pretrained=False)
        e = m(x)
        print(f"  {fusion}: {e.shape} ✓")
