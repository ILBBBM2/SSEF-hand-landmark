"""Create representative real landmark samples for RKNN INT8 calibration.

The input model expects the normalized 131-value vectors already stored in
two_hand_landmark_dataset/landmarks.npy. This script samples those vectors
evenly across labels and writes one (1, 131) float32 .npy file per sample,
plus the dataset list that RKNN-Toolkit2 consumes.
"""

import argparse
from pathlib import Path

import numpy as np

from landmarks.normalize import FEATURE_DIM


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_ROOT / "two_hand_landmark_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "rknn_calibration"


def select_indices(labels, max_samples, seed):
    """Select a balanced, reproducible set of indices from every class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_samples // len(classes))
    selected = []

    for label in classes:
        candidates = np.flatnonzero(labels == label)
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class].tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)].tolist())

    rng.shuffle(selected)
    return np.asarray(selected[:max_samples])


def make_dataset(cache_dir, output_dir, max_samples, seed):
    features_path = cache_dir / "landmarks.npy"
    labels_path = cache_dir / "labels.npy"
    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Expected landmarks.npy and labels.npy in {cache_dir}")

    features = np.load(features_path).astype(np.float32, copy=False)
    labels = np.load(labels_path)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected landmark features shaped (N, {FEATURE_DIM}), got {features.shape}.")
    if len(features) != len(labels):
        raise ValueError("landmarks.npy and labels.npy have different sample counts.")

    selected = select_indices(labels, min(max_samples, len(features)), seed)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    dataset_list = output_dir / "dataset.txt"

    with dataset_list.open("w", encoding="utf-8") as file:
        for number, index in enumerate(selected):
            # RKNN expects a batch dimension, matching the static ONNX input.
            sample_path = samples_dir / f"sample_{number:04d}_label_{labels[index]}.npy"
            np.save(sample_path, features[index : index + 1])
            # Keep paths portable: the same list works after the project (or
            # its RKNN bundle) is copied from Windows to WSL/Linux.
            file.write(f"{sample_path.relative_to(PROJECT_ROOT).as_posix()}\n")

    print(f"Wrote {len(selected)} real calibration samples to {dataset_list}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=300, help="Maximum number of representative samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    make_dataset(args.cache, args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
