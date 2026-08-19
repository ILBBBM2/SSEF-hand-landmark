"""Record detected webcam hand landmarks for one sign class at a time."""

import argparse
import json
import os

import cv2
import numpy as np

from landmarks.extract import HandLandmarkExtractor, draw_hand_landmarks
from utils import LANDMARKS_CACHE_PATH, normalize_label


def load_cache(cache_path):
    landmarks_file = os.path.join(cache_path, "landmarks.npy")
    labels_file = os.path.join(cache_path, "labels.npy")
    metadata_file = os.path.join(cache_path, "metadata.json")
    if not os.path.exists(landmarks_file):
        return np.empty((0, 63), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    if not os.path.exists(metadata_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"Incomplete landmark dataset in {cache_path}. Expected landmarks.npy, "
            "labels.npy, and metadata.json. Choose an empty output folder or restore the files."
        )
    with open(metadata_file) as file:
        classes = json.load(file)["class_names"]
    return np.load(landmarks_file), np.load(labels_file), classes


def collect(label, samples, camera_index, cache_path, every_n_frames):
    os.makedirs(cache_path, exist_ok=True)
    landmarks, labels, classes = load_cache(cache_path)
    label = normalize_label(label)
    matching_indexes = [
        index for index, class_name in enumerate(classes)
        if normalize_label(class_name) == label
    ]
    if len(matching_indexes) > 1:
        raise ValueError(
            f"Dataset has duplicate labels that differ only by case: {classes}. "
            "Remove or merge one before collecting more samples."
        )
    if matching_indexes:
        label_index = matching_indexes[0]
        # Upgrade a legacy lowercase/mixed-case class name when it is reused.
        classes[label_index] = label
    else:
        classes.append(label)
        label_index = len(classes) - 1

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    captured, frame_number, recording = [], 0, False
    print("Press R to start/pause recording; Q to finish. Hold the sign steady and vary angle/distance.")
    with HandLandmarkExtractor(static_image_mode=False) as extractor:
        while len(captured) < samples:
            ok, frame = camera.read()
            if not ok:
                break
            frame_number += 1
            feature = extractor.extract_from_bgr(frame, timestamp_ms=frame_number * 33)
            display = frame.copy()
            draw_hand_landmarks(display, extractor.last_result)
            if recording and feature is not None and frame_number % every_n_frames == 0:
                captured.append(feature)
            status = "RECORDING" if recording else "PAUSED (press R)"
            cv2.putText(display, f"{label}: {len(captured)}/{samples}  {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(display, "Q quit | R record/pause", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Collect sign landmarks", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recording = not recording

    camera.release()
    cv2.destroyAllWindows()
    if not captured:
        print("No samples saved.")
        return
    landmarks = np.vstack([landmarks, np.asarray(captured, dtype=np.float32)])
    labels = np.concatenate([labels, np.full(len(captured), label_index, dtype=np.int64)])
    np.save(os.path.join(cache_path, "landmarks.npy"), landmarks)
    np.save(os.path.join(cache_path, "labels.npy"), labels)
    with open(os.path.join(cache_path, "metadata.json"), "w") as file:
        json.dump({"class_names": classes, "num_samples": len(labels), "feature_dim": 63}, file, indent=2)
    print(f"Saved {len(captured)} '{label}' samples. Dataset total: {len(labels)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect tracked MediaPipe landmarks for one sign.")
    parser.add_argument(
        "--label",
        help="Class name, e.g. A, HELLO, or THANK_YOU. If omitted, you are prompted.",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default=LANDMARKS_CACHE_PATH)
    parser.add_argument("--every", type=int, default=3, help="Save one detected frame every N camera frames")
    args = parser.parse_args()
    raw_label = args.label or input("Label for these samples (for example A or HELLO): ").strip()
    if not raw_label:
        parser.error("A label is required.")
    collect(raw_label, args.samples, args.camera, args.output, max(args.every, 1))
