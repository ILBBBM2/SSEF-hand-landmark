import argparse
import json
from pathlib import Path

import torch

from models import LANDMARK_MODELS, build_model
from utils import (
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    LANDMARK_FEATURE_DIM,
    class_names_path_for,
    checkpoint_path_for,
    get_model_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def model_paths(model_name):
    """Return the checkpoint and class-name files independent of the launch cwd."""
    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_DIR
    return (
        Path(checkpoint_path_for(model_name, checkpoint_dir=str(checkpoint_dir))),
        Path(class_names_path_for(model_name, checkpoint_dir=str(checkpoint_dir))),
    )


def infer_num_classes(model_name, class_names_path):
    """Use the saved labels so this script can be run directly from VS Code."""
    if class_names_path.is_file():
        with class_names_path.open(encoding="utf-8") as file:
            return len(json.load(file))

    raise FileNotFoundError(
        f"Class-name file not found: {class_names_path}. "
        "Train the model first, or pass --num-classes explicitly."
    )


def export(model_name, num_classes, output_path):
    checkpoint_path, _ = model_paths(model_name)
    state_dict = get_model_state_dict(str(checkpoint_path))

    # Older checkpoints may have a different landmark feature dimension.
    input_dim = state_dict["0.weight"].shape[1] if model_name in LANDMARK_MODELS else None
    model, _ = build_model(model_name, num_classes, input_dim=input_dim)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    if model_name in LANDMARK_MODELS:
        dummy_input = torch.randn(1, LANDMARK_FEATURE_DIM, device=DEVICE)
        input_names = ["landmarks"]
    else:
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        input_names = ["input"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=["output"],
        opset_version=17,
        # Keep the exporter compatible with the installed PyTorch without
        # requiring the separate onnxscript package.
        dynamo=False,
    )

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="landmark_mlp")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override the class count saved in models/<model>_classes.json.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, class_names_path = model_paths(args.model)
    num_classes = args.num_classes or infer_num_classes(args.model, class_names_path)
    output_path = args.output or PROJECT_ROOT / CHECKPOINT_DIR / f"{args.model}.onnx"
    export(args.model, num_classes, output_path)


if __name__ == "__main__":
    main()
