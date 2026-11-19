import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("two_hand_landmark_dataset")
OUTPUT_DIR = Path("validation_export")

SEED = 42
TRAIN_SPLIT = 0.8

# Load the original dataset.
features = np.load(DATA_DIR / "landmarks.npy")
labels = np.load(DATA_DIR / "labels.npy")
groups = np.load(DATA_DIR / "groups.npy")

with open(DATA_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

class_names = metadata["class_names"]

# Reproduce the exact group-based split from landmark_dataset.py.
unique_groups = np.unique(groups)
np.random.default_rng(SEED).shuffle(unique_groups)

train_groups = set(
    unique_groups[:int(TRAIN_SPLIT * len(unique_groups))].tolist()
)

is_training = np.isin(groups, list(train_groups))
validation_indices = np.flatnonzero(~is_training)

# Extract validation samples in their original dataset order.
validation_features = features[validation_indices].astype("<f4")
validation_labels = labels[validation_indices].astype("<i4")

# Basic integrity checks.
assert validation_features.shape == (4624, 131), validation_features.shape
assert validation_labels.shape == (4624,), validation_labels.shape
assert np.isfinite(validation_features).all(), "Non-finite feature values found"
assert validation_labels.min() >= 0
assert validation_labels.max() < len(class_names)

OUTPUT_DIR.mkdir(exist_ok=True)

# Raw binary files for the Luckfox C tester.
validation_features.tofile(OUTPUT_DIR / "validation_features.bin")
validation_labels.tofile(OUTPUT_DIR / "validation_labels.bin")

# Keep the indices and class names for result interpretation.
np.save(OUTPUT_DIR / "validation_indices.npy", validation_indices)

with open(OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

manifest = {
    "samples": len(validation_labels),
    "features_per_sample": validation_features.shape[1],
    "classes": len(class_names),
    "feature_dtype": "float32 little-endian",
    "label_dtype": "int32 little-endian",
    "seed": SEED,
    "train_split": TRAIN_SPLIT,
    "unique_groups": len(unique_groups),
    "training_samples": int(is_training.sum()),
    "validation_samples": int((~is_training).sum()),
}

with open(OUTPUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Validation export completed.")
print(f"Training samples:   {is_training.sum()}")
print(f"Validation samples: {len(validation_labels)}")
print(f"Feature shape:      {validation_features.shape}")
print(f"Feature file size:  {(OUTPUT_DIR / 'validation_features.bin').stat().st_size:,} bytes")
print(f"Label file size:    {(OUTPUT_DIR / 'validation_labels.bin').stat().st_size:,} bytes")

print("\nValidation samples per class:")
for i, name in enumerate(class_names):
    count = int(np.sum(validation_labels == i))
    print(f"{i:2} {name:6} {count}")

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")
