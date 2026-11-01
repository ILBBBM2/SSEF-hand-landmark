"""Create one ZIP file containing everything needed to convert on WSL/Linux."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "models" / "luckfox_rv1106_rknn_bundle.zip"
REQUIRED_FILES = (
    PROJECT_ROOT / "onnx_to_rknn.py",
    PROJECT_ROOT / "models" / "landmark_mlp.onnx",
    PROJECT_ROOT / "models" / "rknn_calibration" / "dataset.txt",
)
SAMPLES_DIR = PROJECT_ROOT / "models" / "rknn_calibration" / "samples"


def main():
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if not SAMPLES_DIR.is_dir() or not any(SAMPLES_DIR.glob("*.npy")):
        missing.append(SAMPLES_DIR)
    if missing:
        names = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required bundle files:\n{names}\nRun export_onnx.py and make_calibration_dataset.py first.")

    with ZipFile(BUNDLE_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for path in REQUIRED_FILES:
            archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        for sample_path in sorted(SAMPLES_DIR.glob("*.npy")):
            archive.write(sample_path, sample_path.relative_to(PROJECT_ROOT).as_posix())

    print(f"Created {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
