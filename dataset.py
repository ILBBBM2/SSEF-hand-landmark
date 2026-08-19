import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from utils import IMAGENET_MEAN, IMAGENET_STD, TRAIN_SPLIT


def get_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_dataloaders(
    dataset_path,
    batch_size,
    num_workers,
    image_size,
    train_split=TRAIN_SPLIT,
):
    train_dataset = datasets.ImageFolder(
        dataset_path,
        transform=get_train_transform(image_size),
    )
    val_dataset = datasets.ImageFolder(
        dataset_path,
        transform=get_val_transform(image_size),
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
    )
    val_subset = Subset(val_dataset, val_subset.indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, class_names, num_classes
