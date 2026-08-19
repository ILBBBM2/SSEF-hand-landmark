"""Safely remove one label and all of its samples from a landmark dataset."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from utils import LANDMARKS_CACHE_PATH, normalize_label


def delete_label(label, dataset_path, assume_yes=False):
    """Remove ``label`` while keeping class indexes and metadata consistent."""
    requested_label = normalize_label(label)
    dataset = Path(dataset_path)
    landmarks_path = dataset / "landmarks.npy"
    labels_path = dataset / "labels.npy"
    metadata_path = dataset / "metadata.json"

    required_files = (landmarks_path, labels_path, metadata_path)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Dataset is missing: " + ", ".join(missing))

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    classes = metadata.get("class_names", [])
    matching_indexes = [
        index for index, class_name in enumerate(classes)
        if normalize_label(class_name) == requested_label
    ]
    if not matching_indexes:
        raise ValueError(f"Label {label!r} was not found. Available labels: {classes}")
    if len(matching_indexes) > 1:
        raise ValueError(
            f"More than one dataset label matches {label!r}: "
            f"{[classes[index] for index in matching_indexes]}. Resolve the duplicate first."
        )

    landmarks = np.load(landmarks_path)
    labels = np.load(labels_path)
    if len(landmarks) != len(labels):
        raise ValueError("landmarks.npy and labels.npy have different numbers of samples.")

    label_index = matching_indexes[0]
    stored_label = classes[label_index]
    remove_mask = labels == label_index
    removed_count = int(remove_mask.sum())
    print(
        f"Found {removed_count} sample(s) for {stored_label!r} in {dataset}. "
        f"{len(labels) - removed_count} sample(s) will remain."
    )

    if not assume_yes:
        confirmation = input(f"Type DELETE {stored_label} to remove these samples: ").strip()
        if confirmation.casefold() != f"DELETE {stored_label}".casefold():
            print("Cancelled. No files were changed.")
            return False

    # Keep a recoverable copy before overwriting the three interdependent files.
    backup = dataset / "backups" / datetime.now().strftime("before_delete_%Y%m%d_%H%M%S")
    backup.mkdir(parents=True)
    for path in required_files:
        shutil.copy2(path, backup / path.name)

    remaining_landmarks = landmarks[~remove_mask]
    remaining_labels = labels[~remove_mask].copy()
    # Class positions are the encoded label values, so values after the deleted
    # class must shift down by one to match the shortened class_names list.
    remaining_labels[remaining_labels > label_index] -= 1

    metadata["class_names"] = [
        name for index, name in enumerate(classes) if index != label_index
    ]
    metadata["num_samples"] = int(len(remaining_labels))
    metadata["feature_dim"] = int(landmarks.shape[1]) if landmarks.ndim == 2 else 0

    np.save(landmarks_path, remaining_landmarks)
    np.save(labels_path, remaining_labels)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Removed {removed_count} sample(s) for {stored_label!r}.")
    print(f"Backup saved to {backup}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove one label and its samples from a webcam landmark dataset."
    )
    parser.add_argument("--label", help="Label to delete. If omitted, you are prompted.")
    parser.add_argument("--dataset", default=LANDMARKS_CACHE_PATH)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without typing the interactive confirmation.",
    )
    args = parser.parse_args()
    label = args.label or input("Label to delete: ").strip()
    if not label:
        parser.error("A label is required.")
    delete_label(label, args.dataset, args.yes)


if __name__ == "__main__":
    main()
