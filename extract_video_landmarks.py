"""Build a landmark training dataset from labeled video files.

Each video's filename stem is its class label: ``A.mov`` becomes label ``A``.
Only frames where MediaPipe tracks a hand are included in the training cache.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from landmarks.extract import HandLandmarkExtractor, draw_hand_landmarks
from utils import LANDMARKS_CACHE_PATH, normalize_label


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
PREVIEW_MAX_WIDTH = 1000
PREVIEW_MAX_HEIGHT = 800


def video_files(input_path):
    return sorted(
        (path for path in Path(input_path).iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.stem.casefold(),
    )


def resize_for_preview(frame):
    """Fit a frame on screen without cropping or changing its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def extract_videos(input_path, output_path, sample_fps=6.0, visualize=False):
    files = video_files(input_path)
    if not files:
        raise FileNotFoundError(f"No supported video files found in {input_path}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")

    class_names = [normalize_label(path.stem) for path in files]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Video filenames must have unique stems (labels).")

    all_features, all_labels, report = [], [], []
    print(f"Extracting tracked landmarks from {len(files)} videos at up to {sample_fps:g} FPS")
    window_name = "MediaPipe hand landmark extraction"
    if visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    with HandLandmarkExtractor(static_image_mode=False) as extractor:
        for label_index, path in enumerate(tqdm(files, desc="Videos")):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                report.append({"video": str(path), "label": path.stem, "error": "could not open"})
                continue

            source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            frame_stride = max(1, round(source_fps / sample_fps))
            frame_index, accepted, processed = 0, 0, 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride == 0:
                    timestamp_ms = round(frame_index * 1000 / source_fps)
                    feature = extractor.extract_from_bgr(frame, timestamp_ms=timestamp_ms)
                    processed += 1
                    if feature is not None:
                        all_features.append(feature)
                        all_labels.append(label_index)
                        accepted += 1
                    if visualize:
                        # The source videos can be 4K portrait. Resize the
                        # preview first, then draw normalized landmarks on it.
                        # The full-size frame is still used for extraction.
                        display = resize_for_preview(frame)
                        draw_hand_landmarks(display, extractor.last_result)
                        status = "HAND DETECTED" if feature is not None else "NO HAND DETECTED"
                        cv2.putText(
                            display,
                            f"{path.name} | {status} | saved: {accepted}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0) if feature is not None else (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            display,
                            "Q: stop visualization",
                            (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            visualize = False
                            cv2.destroyWindow(window_name)
                frame_index += 1
            capture.release()
            report.append(
                {
                    "video": str(path),
                    "label": path.stem,
                    "source_fps": source_fps,
                    "processed_frames": processed,
                    "accepted_landmarks": accepted,
                }
            )

    cv2.destroyAllWindows()
    if not all_features:
        raise RuntimeError("No hand landmarks were found in any video.")

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    landmarks = np.asarray(all_features, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    np.save(output / "landmarks.npy", landmarks)
    np.save(output / "labels.npy", labels)
    with open(output / "metadata.json", "w") as file:
        json.dump(
            {
                "class_names": class_names,
                "num_samples": len(labels),
                "feature_dim": int(landmarks.shape[1]),
                "source_videos": str(Path(input_path).resolve()),
                "sample_fps": sample_fps,
            },
            file,
            indent=2,
        )
    with open(output / "video_report.json", "w") as file:
        json.dump(report, file, indent=2)

    accepted_by_label = {name: int((labels == index).sum()) for index, name in enumerate(class_names)}
    print(f"Saved {len(labels)} landmark frames across {len(class_names)} labels to {output}")
    print("Samples per label:", accepted_by_label)


def main():
    parser = argparse.ArgumentParser(description="Extract tracked hand landmarks from labeled videos.")
    parser.add_argument("--input", default="./vids", help="Folder containing one labeled video per filename")
    parser.add_argument("--output", default=LANDMARKS_CACHE_PATH)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the detected landmark skeleton while frames are extracted (press Q to hide it).",
    )
    args = parser.parse_args()
    extract_videos(args.input, args.output, args.sample_fps, args.visualize)


if __name__ == "__main__":
    main()
