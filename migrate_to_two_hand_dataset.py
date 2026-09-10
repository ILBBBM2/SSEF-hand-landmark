"""Convert a legacy one-hand landmark cache into the two-hand feature format.

Each legacy sample is copied twice: once in the left slot and once in the
right slot. The source files are read only; output must be a different folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from landmarks.normalize import FEATURE_DIM, HAND_FEATURE_DIM


def migrate(source_path, output_path):
    source = Path(source_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Output must be a new folder; the legacy dataset is never overwritten.")
    landmarks = np.load(source / "landmarks.npy")
    labels = np.load(source / "labels.npy")
    if landmarks.ndim != 2 or landmarks.shape[1] != HAND_FEATURE_DIM:
        raise ValueError(f"Expected legacy landmarks with {HAND_FEATURE_DIM} features, got {landmarks.shape}.")
    if len(landmarks) != len(labels):
        raise ValueError("landmarks.npy and labels.npy have different numbers of samples.")
    with open(source / "metadata.json") as file:
        source_metadata = json.load(file)

    count = len(landmarks)
    converted = np.zeros((count * 2, FEATURE_DIM), dtype=np.float32)
    # [left hand, right hand, left present, right present, relative wrist xyz]
    converted[:count, :HAND_FEATURE_DIM] = landmarks
    converted[:count, HAND_FEATURE_DIM * 2] = 1.0
    converted[count:, HAND_FEATURE_DIM:HAND_FEATURE_DIM * 2] = landmarks
    converted[count:, HAND_FEATURE_DIM * 2 + 1] = 1.0
    converted_labels = np.concatenate([labels, labels]).astype(np.int64)
    groups = np.concatenate([np.arange(count), np.arange(count)]).astype(np.int64)

    output.mkdir(parents=True, exist_ok=True)
    for filename in ("landmarks.npy", "labels.npy", "groups.npy", "metadata.json"):
        if (output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output / filename}")
    np.save(output / "landmarks.npy", converted)
    np.save(output / "labels.npy", converted_labels)
    np.save(output / "groups.npy", groups)
    metadata = {
        "class_names": source_metadata["class_names"],
        "num_samples": int(len(converted_labels)),
        "feature_dim": FEATURE_DIM,
        "format": "two_hand_v1",
        "migrated_from": str(source.resolve()),
        "migration": "Each legacy sample was duplicated into left-only and right-only slots.",
    }
    with open(output / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)
    print(f"Migrated {count} legacy samples into {len(converted_labels)} two-hand samples at {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely migrate a legacy 63-feature dataset to two-hand features.")
    parser.add_argument("--input", default="./landmark_dataset")
    parser.add_argument("--output", default="./two_hand_landmark_dataset")
    args = parser.parse_args()
    migrate(args.input, args.output)
