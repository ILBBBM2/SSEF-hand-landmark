import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
TASK_FILE = ROOT / "landmarks" / "hand_landmarker.task"
OUT_DIR = ROOT / "models" / "mediapipe_rknn"


def main():
    if not TASK_FILE.is_file():
        sys.exit(f"Cannot find {TASK_FILE}. Make sure you are in your main project folder.")

    extract_dir = OUT_DIR / "extracted"
    calib_dir = OUT_DIR / "calib"
    extract_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unzip hand_landmarker.task to get the two TFLite models inside
    with ZipFile(TASK_FILE, "r") as z:
        z.extractall(extract_dir)

    detector_tflite = extract_dir / "hand_detector.tflite"
    landmarker_tflite = extract_dir / "hand_landmarks_detector.tflite"

    # 2. Create 32 calibration images for INT8 quantization
    dataset_txt = calib_dir / "dataset.txt"
    rng = np.random.default_rng(42)
    with dataset_txt.open("w", encoding="utf-8") as f:
        for i in range(32):
            img_path = calib_dir / f"calib_{i:02d}.jpg"
            arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), arr)
            f.write(f"{img_path.resolve()}\n")

    # 3. Convert both TFLite models into RV1106 .rknn models
    models = [
        (detector_tflite, OUT_DIR / "hand_detector.rknn"),
        (landmarker_tflite, OUT_DIR / "hand_landmarks_detector.rknn"),
    ]

    for tflite_path, rknn_path in models:
        print(f"\n=== Converting {tflite_path.name} -> {rknn_path.name} ===")
        rknn = RKNN(verbose=True)
        try:
            assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106") == 0
            assert rknn.load_tflite(model=str(tflite_path)) == 0
            assert rknn.build(do_quantization=True, dataset=str(dataset_txt)) == 0
            assert rknn.export_rknn(str(rknn_path)) == 0
            print(f"SUCCESS: Created {rknn_path}")
        finally:
            rknn.release()


if __name__ == "__main__":
    main()
