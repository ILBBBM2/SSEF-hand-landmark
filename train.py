import argparse
import json
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from engine import train, validate
from landmark_dataset import create_landmark_dataloaders
from models import LANDMARK_MODELS, build_model
from utils import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_PATH,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    LANDMARKS_CACHE_PATH,
    LEARNING_RATE,
    NUM_WORKERS,
    checkpoint_path_for,
    class_names_path_for,
    load_checkpoint,
)


def is_landmark_model(model_name):
    return model_name in LANDMARK_MODELS


def create_dataloaders_for_model(model_name):
    if is_landmark_model(model_name):
        return create_landmark_dataloaders(
            LANDMARKS_CACHE_PATH,
            BATCH_SIZE,
            NUM_WORKERS,
        )

    train_loader, val_loader, class_names, num_classes = create_dataloaders(
        DATASET_PATH,
        BATCH_SIZE,
        NUM_WORKERS,
        IMAGE_SIZE,
    )
    return train_loader, val_loader, class_names, num_classes, None


def main(
    model_name="landmark_mlp",
    resume=False,
    device=DEVICE,
):
    loaders = create_dataloaders_for_model(model_name)
    train_loader, val_loader, class_names, num_classes = loaders[:4]
    sample_count = loaders[4] if len(loaders) > 4 else None

    print(class_names)
    print(f"Number of Classes: {num_classes}")
    if sample_count is not None:
        print(f"Landmark samples: {sample_count}")
    if device.type == "cuda":
        print(f"Training device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Training device: cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(class_names_path_for(model_name), "w") as f:
        json.dump(class_names, f)

    model, trainable_params = build_model(model_name, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    checkpoint_path = checkpoint_path_for(model_name)
    start_epoch = 0
    best_accuracy = 0.0

    if resume:
        if not os.path.exists(checkpoint_path):
            print(
                f"No checkpoint found for {model_name} at {checkpoint_path}. "
                "Starting from scratch."
            )
        else:
            start_epoch, saved_best = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            if saved_best is None:
                best_accuracy = validate(model, val_loader, device)
                print(
                    f"Resumed legacy checkpoint. "
                    f"Current validation accuracy: {best_accuracy:.2f}%"
                )
            else:
                best_accuracy = saved_best
                print(
                    f"Resumed from epoch {start_epoch}, "
                    f"best validation accuracy: {best_accuracy:.2f}%"
                )

    try:
        best_accuracy = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            EPOCHS,
            checkpoint_path,
            start_epoch=start_epoch,
            best_accuracy=best_accuracy,
        )
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        return

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


def prompt_resume(model_name, checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_path = checkpoint_path_for(model_name, checkpoint_dir=checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        print(
            f"No checkpoint found for {model_name} at {checkpoint_path}. "
            "Starting from scratch."
        )
        return False

    print(f"Found checkpoint: {checkpoint_path}")

    while True:
        answer = input("Continue from checkpoint? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            print("Starting from scratch.")
            return False
        print("Please enter y or n.")


def prompt_model_name():
    print("Choose a model:")
    print("1. landmark_mlp (MediaPipe hand landmarks)")
    print("2. mobilenetv2 (image-based)")
    print("3. mobilenetv3 (image-based)")
    print("4. customcnn (image-based)")

    while True:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice == "1":
            return "landmark_mlp"
        if choice == "2":
            return "mobilenetv2"
        if choice == "3":
            return "mobilenetv3"
        if choice == "4":
            return "customcnn"
        print("Invalid choice. Enter 1, 2, 3, or 4.")


def select_device(requested_device):
    """Choose an available training device, with an explicit CUDA option."""
    if requested_device == "auto":
        return DEVICE
    if requested_device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU. "
            "Use --device auto or --device cpu instead."
        )
    return torch.device("cuda")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["landmark_mlp", "mobilenetv2", "mobilenetv3", "customcnn"],
        help="Model to train",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training without prompting",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch without prompting",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA when available (default).",
    )
    args = parser.parse_args()

    if args.resume and args.fresh:
        parser.error("Use either --resume or --fresh, not both.")

    model_name = args.model or prompt_model_name()

    if args.resume:
        resume = True
    elif args.fresh:
        resume = False
    else:
        resume = prompt_resume(model_name)

    main(
        model_name=model_name,
        resume=resume,
        device=select_device(args.device),
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli()
