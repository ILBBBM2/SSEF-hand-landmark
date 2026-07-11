import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "./American"
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
CHECKPOINT_DIR = "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def checkpoint_path_for(model_name, checkpoint_dir=CHECKPOINT_DIR):
    safe_model_name = str(model_name).strip().lower().replace(" ", "_")
    return os.path.join(checkpoint_dir, f"{safe_model_name}_best.pth")


def get_model_state_dict(checkpoint_path, device=DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def save_checkpoint(path, model, optimizer, epoch, best_accuracy):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device=DEVICE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_accuracy", 0.0)

    model.load_state_dict(checkpoint)
    return 0, None
