"""
dataset.py — Phase II Dataset Loader (Updated for Kaggle LFW CSV format)
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class FaceTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_identities=None):
        self.root_dir = root_dir
        self.transform = transform or get_train_transform()
        self.samples = []
        self.class_to_idx = {}

        identities = sorted(os.listdir(root_dir))
        if max_identities:
            identities = identities[:max_identities]

        for idx, identity in enumerate(identities):
            self.class_to_idx[identity] = idx
            identity_dir = os.path.join(root_dir, identity)
            if not os.path.isdir(identity_dir):
                continue
            for fname in os.listdir(identity_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(identity_dir, fname), idx))

        self.num_classes = len(self.class_to_idx)
        print(f"[Dataset] {self.num_classes} identities | {len(self.samples)} images loaded from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


class LFWPairsDataset(Dataset):
    """
    Handles Kaggle LFW CSV format:
      Positive pairs CSV: name,imagenum1,imagenum2
      Mismatch pairs CSV: name1,imagenum1,name2,imagenum2
    """

    def __init__(self, lfw_dir, match_csv, mismatch_csv=None, transform=None):
        self.lfw_dir = lfw_dir
        self.transform = transform or get_eval_transform()
        self.pairs = []
        self._load(match_csv, mismatch_csv)

    def _img_path(self, name, num):
        fname = f"{name}_{int(num):04d}.jpg"
        return os.path.join(self.lfw_dir, name, fname)

    def _load(self, match_csv, mismatch_csv):
        # Positive pairs
        with open(match_csv) as f:
            lines = f.read().strip().split('\n')
        for line in lines[1:]:  # skip header
            parts = line.strip().rstrip(',').split(',')
            if len(parts) >= 3:
                name, n1, n2 = parts[0], parts[1], parts[2]
                p1 = self._img_path(name, n1)
                p2 = self._img_path(name, n2)
                if os.path.exists(p1) and os.path.exists(p2):
                    self.pairs.append((p1, p2, 1))

        # Negative pairs
        if mismatch_csv and os.path.exists(mismatch_csv):
            with open(mismatch_csv) as f:
                lines = f.read().strip().split('\n')
            for line in lines[1:]:
                parts = line.strip().rstrip(',').split(',')
                if len(parts) >= 4:
                    name1, n1, name2, n2 = parts[0], parts[1], parts[2], parts[3]
                    p1 = self._img_path(name1, n1)
                    p2 = self._img_path(name2, n2)
                    if os.path.exists(p1) and os.path.exists(p2):
                        self.pairs.append((p1, p2, 0))

        print(f"[LFW] Loaded {len(self.pairs)} pairs "
              f"({sum(1 for _,_,l in self.pairs if l==1)} pos, "
              f"{sum(1 for _,_,l in self.pairs if l==0)} neg)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        return self.transform(img1), self.transform(img2), label


def get_train_loader(root_dir, batch_size=16, num_workers=0, max_identities=None):
    dataset = FaceTrainDataset(root_dir, max_identities=max_identities)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=False)
    return loader, dataset.num_classes


def get_lfw_loader(lfw_dir, match_csv, mismatch_csv=None, batch_size=16, num_workers=0):
    dataset = LFWPairsDataset(lfw_dir, match_csv, mismatch_csv)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    # Test training loader
    loader, n_cls = get_train_loader('data/train', batch_size=4, max_identities=10)
    imgs, labels = next(iter(loader))
    print(f"Train batch: {imgs.shape} | Classes: {n_cls}")

    # Test eval loader
    eval_loader = get_lfw_loader(
        'data/lfw',
        'data/matchpairsDevTest.csv',
        'data/mismatchpairsDevTest.csv',
        batch_size=4
    )
    img1, img2, label = next(iter(eval_loader))
    print(f"Eval batch: {img1.shape} | Labels: {label}")