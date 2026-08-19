import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from landmarks.normalize import FEATURE_DIM, augment_landmarks
from utils import TRAIN_SPLIT


class LandmarkDataset(Dataset):
    def __init__(self, landmarks, labels, augment=False):
        self.landmarks = landmarks
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.landmarks[index].copy()
        if self.augment:
            features = augment_landmarks(features)

        return torch.from_numpy(features), int(self.labels[index])


def load_landmark_cache(cache_path):
    landmarks_path = os.path.join(cache_path, "landmarks.npy")
    labels_path = os.path.join(cache_path, "labels.npy")
    metadata_path = os.path.join(cache_path, "metadata.json")

    if not os.path.exists(landmarks_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Landmark cache not found in {cache_path}. "
            "Run: python extract_landmarks.py"
        )

    landmarks = np.load(landmarks_path)
    labels = np.load(labels_path)

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        class_names = metadata["class_names"]
    else:
        class_names = [str(i) for i in range(int(labels.max()) + 1)]

    return landmarks, labels, class_names


def create_landmark_dataloaders(
    cache_path,
    batch_size,
    num_workers,
    train_split=TRAIN_SPLIT,
):
    landmarks, labels, class_names = load_landmark_cache(cache_path)
    num_classes = len(class_names)

    full_dataset = LandmarkDataset(landmarks, labels, augment=False)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
    )

    train_dataset = LandmarkDataset(
        landmarks,
        labels,
        augment=True,
    )
    train_subset = Subset(train_dataset, train_subset.indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, class_names, num_classes, len(full_dataset)
