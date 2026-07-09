"""
realtime_recognition.py — Full Pipeline: Detection + Recognition
Integrates Phase I (RetinaFace detection) with Phase II (Hybrid model recognition).

Workflow:
  Webcam → RetinaFace detects faces → Crop & align →
  Hybrid model extracts embeddings → Compare to registered faces →
  Display name + similarity score

Usage:
  # First register known faces:
  python realtime_recognition.py --register --name "Alice" --images path/to/alice/

  # Then run live recognition:
  python realtime_recognition.py --checkpoint checkpoints/hybrid_concat_best.pth \
                                  --model hybrid_concat

Controls:
  q → quit
  r → re-register from webcam (press 'c' to capture)
  s → save screenshot
"""

import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from retinaface import RetinaFace
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default=None,
                   help='Path to trained model checkpoint')
    p.add_argument('--model',       default='hybrid_concat')
    p.add_argument('--num_classes', type=int, default=100)
    p.add_argument('--embed',       type=int, default=512)
    p.add_argument('--gallery',     default='gallery',
                   help='Folder where registered face embeddings are stored')
    p.add_argument('--threshold',   type=float, default=0.6,
                   help='Cosine similarity threshold for recognition (0-1)')
    p.add_argument('--register',    action='store_true',
                   help='Register a new person from image files')
    p.add_argument('--name',        default=None,
                   help='Name of person to register')
    p.add_argument('--images',      default=None,
                   help='Folder of images to register from')
    return p.parse_args()


# ─── Preprocessing ───────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def crop_face(frame, facial_area, margin=0.1):
    """Crop and marginally expand detected face region."""
    x1, y1, x2, y2 = facial_area
    h, w = frame.shape[:2]
    mw = int((x2 - x1) * margin)
    mh = int((y2 - y1) * margin)
    x1 = max(0, x1 - mw)
    y1 = max(0, y1 - mh)
    x2 = min(w, x2 + mw)
    y2 = min(h, y2 + mh)
    return frame[y1:y2, x1:x2]


# ─── Gallery (registered faces) ──────────────────────────────────────────────

class FaceGallery:
    """Stores average embeddings per identity."""

    def __init__(self, gallery_dir):
        self.gallery_dir = gallery_dir
        os.makedirs(gallery_dir, exist_ok=True)
        self.embeddings = {}   # name → mean embedding tensor
        self._load()

    def _load(self):
        for fname in os.listdir(self.gallery_dir):
            if fname.endswith('.npy'):
                name = fname[:-4]
                emb = np.load(os.path.join(self.gallery_dir, fname))
                self.embeddings[name] = torch.from_numpy(emb)
        print(f"[Gallery] Loaded {len(self.embeddings)} identities: {list(self.embeddings.keys())}")

    def register(self, name, embeddings):
        """Register a person with a list of embedding tensors (mean pooled)."""
        mean_emb = torch.stack(embeddings).mean(0)
        mean_emb = F.normalize(mean_emb, dim=0)
        self.embeddings[name] = mean_emb
        np.save(os.path.join(self.gallery_dir, f'{name}.npy'), mean_emb.numpy())
        print(f"[Gallery] Registered '{name}' with {len(embeddings)} images")

    def identify(self, query_emb, threshold=0.6):
        """Return (name, similarity) for best match, or ('Unknown', sim)."""
        if not self.embeddings:
            return 'No faces registered', 0.0

        best_name, best_sim = 'Unknown', -1.0
        for name, emb in self.embeddings.items():
            sim = F.cosine_similarity(query_emb.unsqueeze(0),
                                      emb.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim < threshold:
            return 'Unknown', best_sim
        return best_name, best_sim


# ─── Registration from images ────────────────────────────────────────────────

def register_from_images(model, gallery, name, images_dir, transform, device):
    embeddings = []
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in files:
        path = os.path.join(images_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Detect face
        results = RetinaFace.detect_faces(frame)
        if not isinstance(results, dict) or len(results) == 0:
            print(f"  No face in {fname}, skipping")
            continue

        # Use first detected face
        face_data = list(results.values())[0]
        crop = crop_face(frame, face_data['facial_area'])
        if crop.size == 0:
            continue

        # Extract embedding
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.extract(tensor).squeeze(0)
        embeddings.append(emb.cpu())
        print(f"  Processed {fname}")

    if embeddings:
        gallery.register(name, embeddings)
    else:
        print(f"[Error] No valid face images found in {images_dir}")


# ─── Live Recognition ─────────────────────────────────────────────────────────

def run_recognition(model, gallery, args, transform, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("\nLive recognition running...")
    print("Controls: q=quit | s=screenshot")

    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        t0 = time.time()

        # Detect faces
        results = RetinaFace.detect_faces(frame)

        if isinstance(results, dict):
            for face_id, face_data in results.items():
                x1, y1, x2, y2 = face_data['facial_area']
                score = face_data['score']

                # Crop and embed
                crop = crop_face(frame, (x1, y1, x2, y2))
                if crop.size == 0:
                    continue

                try:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.extract(tensor).squeeze(0).cpu()
                    name, sim = gallery.identify(emb, args.threshold)
                except Exception:
                    name, sim = 'Error', 0.0

                # Draw
                color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{name} ({sim:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Landmarks
                for pt in face_data['landmarks'].values():
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

        # FPS
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)
        avg_fps = np.mean(fps_list[-20:])

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Model: {args.model}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Registered: {len(gallery.embeddings)}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Recognition — Phase II", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('recognition_screenshot.png', frame)
            print("Screenshot saved")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAvg FPS: {np.mean(fps_list):.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cpu')
    transform = get_transform()

    # ── Load model ──
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        num_cls = ckpt.get('args', {}).get('num_classes', args.num_classes)
        model = build_model(args.model, num_classes=num_cls,
                            embed_dim=args.embed, pretrained=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"[Model] Loaded {args.model} from {args.checkpoint}")
    else:
        # No checkpoint: use pretrained ResNet50 backbone only for embeddings
        # (useful for demo before training is complete)
        print("[Model] No checkpoint provided — using pretrained ResNet50 embeddings (no ArcFace)")
        from model import ResNet50Backbone
        import torch.nn as nn

        class QuickModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = ResNet50Backbone(pretrained=True)
                self.proj = nn.Linear(2048, 512)
            def extract(self, x):
                return F.normalize(self.proj(self.backbone(x)), dim=1)
        model = QuickModel()

    model = model.to(device).eval()
    gallery = FaceGallery(args.gallery)

    # ── Register mode ──
    if args.register:
        if not args.name:
            print("ERROR: Provide --name when registering")
            return
        if args.images and os.path.isdir(args.images):
            register_from_images(model, gallery, args.name, args.images, transform, device)
        else:
            print(f"ERROR: --images folder not found: {args.images}")
        return

    # ── Live recognition ──
    run_recognition(model, gallery, args, transform, device)


if __name__ == "__main__":
    main()
