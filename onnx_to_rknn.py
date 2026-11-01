"""Convert this project's ONNX classifier to a Rockchip RKNN model.

Run from VS Code's Run Python File button, or use:
    python onnx_to_rknn.py --target rv1106
    python onnx_to_rknn.py --quantize --dataset calibration_dataset.txt
"""

import argparse
import sys
from pathlib import Path

try:
    from rknn.api import RKNN
except ImportError as error:
    raise SystemExit(
        "RKNN-Toolkit2 is not installed in the selected Python interpreter. "
        "Install Rockchip's RKNN-Toolkit2 package, then run this file again."
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX = PROJECT_ROOT / "models" / "landmark_mlp.onnx"
DEFAULT_RKNN = PROJECT_ROOT / "models" / "landmark_mlp.rknn"


def check_result(result, step):
    if result != 0:
        raise RuntimeError(f"{step} failed (RKNN return code {result}).")


def convert(onnx_path, rknn_path, target, quantize=False, dataset=None):
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Run export_onnx.py first."
        )
    if quantize and (dataset is None or not dataset.is_file()):
        raise FileNotFoundError(
            "INT8 conversion needs --dataset pointing to an existing calibration list."
        )

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        # These are already normalized 131-float landmark vectors. Do not set
        # mean_values/std_values, or RKNN would normalize the values again.
        check_result(rknn.config(target_platform=target), "RKNN configuration")
        check_result(rknn.load_onnx(model=str(onnx_path)), "Loading ONNX")
        check_result(
            rknn.build(
                do_quantization=quantize,
                dataset=str(dataset) if quantize else None,
            ),
            "Building RKNN model",
        )
        check_result(rknn.export_rknn(str(rknn_path)), "Exporting RKNN")
    finally:
        rknn.release()

    print(f"Exported RKNN model to {rknn_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="rv1106", help="Target, e.g. rv1106 or rk3588.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ONNX, help="Input ONNX path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RKNN, help="Output RKNN path.")
    parser.add_argument("--quantize", action="store_true", help="Build an INT8 model using --dataset.")
    parser.add_argument("--dataset", type=Path, help="Text file listing INT8 calibration inputs.")
    args = parser.parse_args()

    try:
        convert(args.input, args.output, args.target, args.quantize, args.dataset)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
