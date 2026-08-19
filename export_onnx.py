import argparse
import json
import os

import torch

from models import LANDMARK_MODELS, build_model
from utils import (
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    LANDMARK_FEATURE_DIM,
    checkpoint_path_for,
    get_model_state_dict,
)


def export(model_name, num_classes, output_path):
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()

    if model_name in LANDMARK_MODELS:
        dummy_input = torch.randn(1, LANDMARK_FEATURE_DIM, device=DEVICE)
        input_names = ["landmarks"]
    else:
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        input_names = ["input"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes={input_names[0]: {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="landmark_mlp")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(CHECKPOINT_DIR, f"{args.model}.onnx")
    export(args.model, args.num_classes, output_path)


if __name__ == "__main__":
    main()
