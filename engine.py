import os

import torch
from tqdm import tqdm

from utils import save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()

    correct = 0
    total = 0

    progress = tqdm(loader)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress.set_description(f"Epoch {epoch + 1}/{epochs}")

    return 100 * correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100 * correct / total


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    start_epoch=0,
    best_accuracy=0.0,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    total_epochs = start_epoch + epochs
    completed_epoch = start_epoch

    try:
        for epoch in range(start_epoch, total_epochs):
            train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, total_epochs
            )
            val_acc = validate(model, val_loader, device)

            print(
                f"\nEpoch {epoch + 1}/{total_epochs}"
                f"\nTrain Accuracy : {train_acc:.2f}%"
                f"\nValidation Accuracy : {val_acc:.2f}%"
            )

            if val_acc > best_accuracy:
                best_accuracy = val_acc

            completed_epoch = epoch + 1
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_epoch,
                best_accuracy,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            completed_epoch,
            best_accuracy,
        )
        raise

    return best_accuracy
