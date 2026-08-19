import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from dataset import get_val_transform
from landmark_dataset import load_landmark_cache
from landmarks.extract import HandLandmarkExtractor
from models import LANDMARK_MODELS, build_model
from utils import (
    CHECKPOINT_DIR,
    DATASET_PATH,
    DEVICE,
    IMAGE_SIZE,
    LANDMARKS_CACHE_PATH,
    checkpoint_path_for,
    get_model_state_dict,
)


def is_landmark_model(model_name):
    return model_name in LANDMARK_MODELS


def load_class_names(dataset_path, cache_path=LANDMARKS_CACHE_PATH):
    classes_path = os.path.join(CHECKPOINT_DIR, "class_names.json")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            return json.load(f)

    metadata_path = os.path.join(cache_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)["class_names"]

    dataset = datasets.ImageFolder(dataset_path)
    return dataset.classes


def predict_landmarks(features, model_name, num_classes, class_names):
    tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5)

    print(f"\nTop 5 predictions for {model_name}:")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"- {class_names[idx.item()]}: {prob.item() * 100:.2f}%")

    best_label = class_names[top_indices[0][0].item()]
    best_confidence = top_probs[0][0].item()
    return best_label, best_confidence


def predict_image(image_path, model_name, num_classes, class_names):
    if is_landmark_model(model_name):
        with HandLandmarkExtractor(static_image_mode=True) as extractor:
            features = extractor.extract_from_path(image_path)

        if features is None:
            print("No hand detected in image.")
            return None, 0.0

        return predict_landmarks(features, model_name, num_classes, class_names)

    transform = get_val_transform(IMAGE_SIZE)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5)

    print(f"\nTop 5 predictions for {model_name}:")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"- {class_names[idx.item()]}: {prob.item() * 100:.2f}%")

    best_label = class_names[top_indices[0][0].item()]
    best_confidence = top_probs[0][0].item()
    return best_label, best_confidence


def evaluate_landmark_model(model_name, cache_path, class_names, batch_size=32, num_workers=4):
    landmarks, labels, _ = load_landmark_cache(cache_path)
    num_classes = len(class_names)

    dataset = TensorDataset(
        torch.from_numpy(landmarks),
        torch.from_numpy(labels.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            outputs = model(features)
            _, preds = outputs.max(1)

            total += batch_labels.size(0)
            correct += preds.eq(batch_labels).sum().item()

            for label_idx, pred_idx in zip(batch_labels.tolist(), preds.tolist()):
                label_name = class_names[label_idx]
                class_total[label_name] += 1
                if label_idx == pred_idx:
                    class_correct[label_name] += 1

    overall_accuracy = 100 * correct / total if total else 0.0
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")
    print(f"Total samples tested: {total}")

    print("\nPer-class accuracy:")
    for class_name in class_names:
        class_acc = (
            100 * class_correct[class_name] / class_total[class_name]
            if class_total[class_name]
            else 0.0
        )
        print(
            f"- {class_name}: {class_acc:.2f}% "
            f"({class_correct[class_name]}/{class_total[class_name]})"
        )

    return overall_accuracy


def evaluate_image_model(model_name, dataset_path, class_names, batch_size=32, num_workers=4):
    num_classes = len(class_names)
    dataset = datasets.ImageFolder(dataset_path, transform=get_val_transform(IMAGE_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            for label_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
                label_name = class_names[label_idx]
                class_total[label_name] += 1
                if label_idx == pred_idx:
                    class_correct[label_name] += 1

    overall_accuracy = 100 * correct / total if total else 0.0
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")
    print(f"Total samples tested: {total}")

    print("\nPer-class accuracy:")
    for class_name in class_names:
        class_acc = (
            100 * class_correct[class_name] / class_total[class_name]
            if class_total[class_name]
            else 0.0
        )
        print(
            f"- {class_name}: {class_acc:.2f}% "
            f"({class_correct[class_name]}/{class_total[class_name]})"
        )

    return overall_accuracy


def evaluate_model(model_name, dataset_path, class_names, batch_size=32, num_workers=4):
    if is_landmark_model(model_name):
        return evaluate_landmark_model(
            model_name,
            LANDMARKS_CACHE_PATH,
            class_names,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    return evaluate_image_model(
        model_name,
        dataset_path,
        class_names,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def prompt_model_name():
    print("Choose a model:")
    print("1. landmark_mlp (MediaPipe hand landmarks)")
    print("2. mobilenetv2 (image-based)")
    print("3. mobilenetv3 (image-based)")
    print("4. customcnn (image-based)")
    print("5. all models")

    while True:
        choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
        if choice == "1":
            return "landmark_mlp"
        if choice == "2":
            return "mobilenetv2"
        if choice == "3":
            return "mobilenetv3"
        if choice == "4":
            return "customcnn"
        if choice == "5":
            return "all"
        print("Invalid choice. Enter 1, 2, 3, 4, or 5.")


def prompt_mode():
    print("Choose an action:")
    print("1. Evaluate the full dataset")
    print("2. Test one image")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "evaluate"
        if choice == "2":
            return "image"
        print("Invalid choice. Enter 1 or 2.")


ALL_MODELS = ["landmark_mlp", "mobilenetv2", "mobilenetv3", "customcnn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to an image file")
    parser.add_argument(
        "--model",
        choices=["landmark_mlp", "mobilenetv2", "mobilenetv3", "customcnn", "all"],
        help="Model to test",
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model across the full dataset",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.dataset)

    if args.model:
        model_name = args.model
    else:
        model_name = prompt_model_name()

    if args.evaluate:
        if model_name == "all":
            for each_model in ALL_MODELS:
                print(f"\n=== {each_model} ===")
                evaluate_model(each_model, args.dataset, class_names)
        else:
            evaluate_model(model_name, args.dataset, class_names)
        return

    mode = "image"
    if not args.image and not args.evaluate:
        mode = prompt_mode()

    if mode == "evaluate":
        if model_name == "all":
            for each_model in ALL_MODELS:
                print(f"\n=== {each_model} ===")
                evaluate_model(each_model, args.dataset, class_names)
        else:
            evaluate_model(model_name, args.dataset, class_names)
        return

    image_path = args.image
    if not image_path:
        image_path = input("Enter the path to the image: ").strip().strip('"')

    if model_name == "all":
        for each_model in ALL_MODELS:
            print(f"\n=== {each_model} ===")
            predict_image(image_path, each_model, len(class_names), class_names)
    else:
        label, confidence = predict_image(
            image_path,
            model_name,
            len(class_names),
            class_names,
        )

        if label is not None:
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        input("\nPress Enter to exit...")
