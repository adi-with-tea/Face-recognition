"""
setup_phase2.py — One-click environment check and install for Phase II
Run this first to verify everything is installed correctly.

Usage:
  python setup_phase2.py
"""

import subprocess
import sys
import importlib


def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])


def check_or_install(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
        print(f"  ✅ {import_name}")
        return True
    except ImportError:
        print(f"  ⚙️  Installing {pip_name}...")
        try:
            install(pip_name)
            print(f"  ✅ {import_name} installed")
            return True
        except Exception as e:
            print(f"  ❌ Failed to install {pip_name}: {e}")
            return False


print("\n" + "="*55)
print("  Phase II — Environment Setup Check")
print("="*55 + "\n")

packages = [
    ('torch',               'torch'),
    ('torchvision',         'torchvision'),
    ('timm',                'timm'),
    ('PIL',                 'Pillow'),
    ('sklearn',             'scikit-learn'),
    ('cv2',                 'opencv-python'),
    ('numpy',               'numpy'),
    ('matplotlib',          'matplotlib'),
    ('facenet_pytorch',     'facenet-pytorch'),
]

all_ok = True
for import_name, pip_name in packages:
    ok = check_or_install(import_name, pip_name)
    if not ok:
        all_ok = False

print()

# Check torch
try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()} (running on CPU)")
except Exception:
    pass

# Test model imports
print("\nTesting model imports...")
try:
    from model import HybridFaceNet
    import torch
    m = HybridFaceNet(num_classes=10, fusion='concat', pretrained=False)
    x = torch.randn(1, 3, 112, 112)
    emb = m(x)
    print(f"  ✅ HybridFaceNet (concat) — embedding shape: {emb.shape}")

    m2 = HybridFaceNet(num_classes=10, fusion='weighted', pretrained=False)
    emb2 = m2(x)
    print(f"  ✅ HybridFaceNet (weighted) — embedding shape: {emb2.shape}")

    m3 = HybridFaceNet(num_classes=10, fusion='attention', pretrained=False)
    emb3 = m3(x)
    print(f"  ✅ HybridFaceNet (attention) — embedding shape: {emb3.shape}")
except Exception as e:
    print(f"  ❌ Model test failed: {e}")
    all_ok = False

print()
if all_ok:
    print("  🎉 All good! You're ready for Phase II.")
else:
    print("  ⚠️  Some packages failed. Fix errors above before proceeding.")

print("\n" + "="*55)
print("  Next step: Download LFW dataset, then run:")
print("    python train.py --data data/train --model hybrid_concat --epochs 5 --max_id 50")
print("="*55 + "\n")
