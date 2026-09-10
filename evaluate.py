"""Evaluate a trained landmark model and save competition-ready metrics."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from landmark_dataset import create_landmark_dataloaders
from models import LANDMARK_MODELS, build_model
from utils import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    SPLIT_SEED,
    checkpoint_path_for,
    class_names_path_for,
    get_model_state_dict,
    set_random_seed,
)


def load_model(model_name, num_classes):
    state_dict = get_model_state_dict(checkpoint_path_for(model_name))
    input_dim = state_dict["0.weight"].shape[1]
    model, _ = build_model(model_name, num_classes, input_dim=input_dim)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


@torch.no_grad()
def collect_predictions(model, loader):
    actual, predicted = [], []
    for features, labels in loader:
        outputs = model(features.to(DEVICE, non_blocking=True))
        predicted.extend(outputs.argmax(dim=1).cpu().tolist())
        actual.extend(labels.tolist())
    return np.asarray(actual), np.asarray(predicted)


def metric_summary(actual, predicted, class_names):
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    true_positives = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    precision = np.divide(true_positives, predicted_count, out=np.zeros(class_count), where=predicted_count != 0)
    recall = np.divide(true_positives, support, out=np.zeros(class_count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count), where=(precision + recall) != 0)
    report = [
        {"class": class_names[i], "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in range(class_count)
    ]
    return matrix, report, {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "samples": int(len(actual)),
    }


def save_confusion_matrix(matrix, class_names, output_path):
    cell = max(26, min(54, 1600 // max(len(class_names), 1)))
    margin = 110
    image = np.full((margin + cell * len(class_names), margin + cell * len(class_names), 3), 255, dtype=np.uint8)
    maximum = max(int(matrix.max()), 1)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            intensity = int(255 * matrix[row, column] / maximum)
            color = (255 - intensity, 255 - intensity // 3, 255)
            x, y = margin + column * cell, margin + row * cell
            cv2.rectangle(image, (x, y), (x + cell, y + cell), color, thickness=-1)
            if cell >= 30:
                cv2.putText(image, str(matrix[row, column]), (x + 2, y + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, class_names[row], (8, margin + row * cell + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, class_names[row], (margin + row * cell + 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Rows: true class | Columns: predicted class", (margin, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained landmark classifier.")
    parser.add_argument("--model", default="landmark_mlp")
    parser.add_argument("--dataset", default="./two_hand_landmark_dataset")
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    if args.model not in LANDMARK_MODELS:
        parser.error("evaluate.py currently supports landmark models only.")

    set_random_seed(args.seed)
    _, validation_loader, dataset_class_names, num_classes, _, _ = create_landmark_dataloaders(
        args.dataset, BATCH_SIZE, NUM_WORKERS, seed=args.seed
    )
    class_names = dataset_class_names
    classes_file = Path(class_names_path_for(args.model))
    if classes_file.exists():
        with open(classes_file) as file:
            class_names = json.load(file)
    if class_names != dataset_class_names:
        raise ValueError("Checkpoint class list does not match the evaluation dataset.")

    actual, predicted = collect_predictions(load_model(args.model, num_classes), validation_loader)
    matrix, report, summary = metric_summary(actual, predicted, class_names)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "metrics.json", "w") as file:
        json.dump(summary, file, indent=2)
    with open(output / "per_class_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(report)
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    save_confusion_matrix(matrix, class_names, output / "confusion_matrix.png")
    print(f"Accuracy: {summary['accuracy'] * 100:.2f}%")
    print(f"Macro-F1: {summary['macro_f1'] * 100:.2f}%")
    print(f"Saved evaluation files to {output.resolve()}")


if __name__ == "__main__":
    main()
