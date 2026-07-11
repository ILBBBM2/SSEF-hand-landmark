import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import get_val_transform
from models import build_model
from utils import CHECKPOINT_DIR, DEVICE, IMAGE_SIZE, checkpoint_path_for, get_model_state_dict


def load_class_names(dataset_path):
    classes_path = os.path.join(CHECKPOINT_DIR, "class_names.json")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            return json.load(f)

    from torchvision import datasets

    dataset = datasets.ImageFolder(dataset_path)
    return dataset.classes


def predict(image_path, model_name, num_classes, class_names):
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


def evaluate_model(model_name, dataset_path, class_names, batch_size=32, num_workers=4):
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
        class_acc = 100 * class_correct[class_name] / class_total[class_name] if class_total[class_name] else 0.0
        print(f"- {class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")

    return overall_accuracy


def prompt_model_name():
    print("Choose a model:")
    print("1. mobilenetv2")
    print("2. mobilenetv3")
    print("3. customcnn")
    print("4. all models")

    while True:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice == "1":
            return "mobilenetv2"
        if choice == "2":
            return "mobilenetv3"
        if choice == "3":
            return "customcnn"
        if choice == "4":
            return "all"
        print("Invalid choice. Enter 1, 2, 3, or 4.")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to an image file")
    parser.add_argument("--model", choices=["mobilenetv2", "mobilenetv3", "customcnn", "all"], help="Model to test")
    parser.add_argument("--dataset", default="./American")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model across the full dataset")
    args = parser.parse_args()

    class_names = load_class_names(args.dataset)

    if args.model:
        model_name = args.model
    else:
        model_name = prompt_model_name()

    if args.evaluate:
        if model_name == "all":
            for each_model in ["mobilenetv2", "mobilenetv3", "customcnn"]:
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
            for each_model in ["mobilenetv2", "mobilenetv3", "customcnn"]:
                print(f"\n=== {each_model} ===")
                evaluate_model(each_model, args.dataset, class_names)
        else:
            evaluate_model(model_name, args.dataset, class_names)
        return

    image_path = args.image
    if not image_path:
        image_path = input("Enter the path to the image: ").strip().strip('"')

    if model_name == "all":
        for each_model in ["mobilenetv2", "mobilenetv3", "customcnn"]:
            print(f"\n=== {each_model} ===")
            predict(image_path, each_model, len(class_names), class_names)
    else:
        label, confidence = predict(
            image_path,
            model_name,
            len(class_names),
            class_names,
        )

        print(f"Prediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        input("\nPress Enter to exit...")
