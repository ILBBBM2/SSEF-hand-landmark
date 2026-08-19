import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets

from landmarks.extract import HandLandmarkExtractor
from utils import DATASET_PATH, LANDMARKS_CACHE_PATH


def extract_dataset(dataset_path, cache_path):
    print(f"Starting landmark extraction for dataset: {dataset_path}")
    print(f"Output directory: {cache_path}")
    dataset = datasets.ImageFolder(dataset_path)
    class_names = dataset.classes

    os.makedirs(cache_path, exist_ok=True)

    landmarks_list = []
    labels_list = []
    failed_images = []

    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for index in tqdm(range(len(dataset)), desc="Extracting landmarks"):
            image_path, label = dataset.samples[index]
            try:
                features = extractor.extract_from_path(image_path)
            except Exception as exc:
                failed_images.append({"path": image_path, "error": str(exc)})
                continue

            if features is None:
                failed_images.append({"path": image_path, "error": "no hand detected"})
                continue

            landmarks_list.append(features)
            labels_list.append(label)

    if failed_images:
        with open(os.path.join(cache_path, "failed.json"), "w") as f:
            json.dump(failed_images, f, indent=2)

    if not landmarks_list:
        initialization_errors = [
            failure["error"]
            for failure in failed_images
            if "could not be initialized" in failure["error"]
        ]
        if initialization_errors:
            raise RuntimeError(initialization_errors[0])
        raise RuntimeError(
            "No hand landmarks were extracted. "
            "Check that your dataset images contain visible hands, and make sure the hand is clearly visible "
            f"and centered. Processed {len(dataset)} images; saved {len(failed_images)} failures to {cache_path}."
        )

    landmarks = np.stack(landmarks_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    np.save(os.path.join(cache_path, "landmarks.npy"), landmarks)
    np.save(os.path.join(cache_path, "labels.npy"), labels)

    metadata = {
        "class_names": class_names,
        "num_samples": len(labels),
        "feature_dim": landmarks.shape[1],
        "failed_count": len(failed_images),
        "source_dataset": os.path.abspath(dataset_path),
    }

    with open(os.path.join(cache_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtracted {len(labels)} samples from {len(dataset)} images.")
    print(f"Skipped {len(failed_images)} images (no hand or error).")
    print(f"Saved cache to {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from an image dataset."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=LANDMARKS_CACHE_PATH)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset}. "
            "Place your ImageFolder dataset there first."
        )

    extract_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
