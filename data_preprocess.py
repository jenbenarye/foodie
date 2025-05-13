import os
import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from datasets.features import Image
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_basic_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def add_gaussian_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

def visualize_noise_effect(image, output_dir, filename, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5]):
    os.makedirs(output_dir, exist_ok=True)
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 3))

    for i, noise_level in enumerate(noise_levels):
        noisy_img = img_tensor if noise_level == 0.0 else add_gaussian_noise(img_tensor, noise_level)
        noisy_pil = transforms.ToPILImage()(noisy_img.squeeze(0))
        axes[i].imshow(noisy_pil)
        axes[i].set_title(f"Noise: {noise_level}")
        axes[i].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def get_dataloaders(batch_size=32, use_augmentation=True):
    dataset = load_dataset("ethz/food101", cache_dir="./datasets_cache")
    dataset = dataset.cast_column("image", Image(decode=False))

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    train_transform = get_augmentation_transforms() if use_augmentation else get_basic_transforms()
    val_transform = get_basic_transforms()

    train_dataset = Food101Dataset(train_ds, transform=train_transform)
    val_dataset = Food101Dataset(val_ds, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return {'train': train_loader, 'val': val_loader}

def get_noisy_test_loader(val_loader, noise_level=0.1, batch_size=32):
    noisy_dataset = NoisyDataset(val_loader.dataset, noise_level)
    return DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

class Food101Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = PILImage.open(BytesIO(item["image"]["bytes"])).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, item["label"]
        except Exception:
            fallback = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                fallback = self.transform(fallback)
            return fallback, item["label"]

class NoisyDataset(Dataset):
    def __init__(self, dataset, noise_level):
        self.dataset = dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        noisy_image = add_gaussian_noise(image.unsqueeze(0), self.noise_level).squeeze(0)
        return noisy_image, label

if __name__ == "__main__":
    output_dir = "food101_preprocess"
    os.makedirs(output_dir, exist_ok=True)

    dataloaders = get_dataloaders(batch_size=32, use_augmentation=True)
    noisy_loader = get_noisy_test_loader(dataloaders['val'], noise_level=0.2)

    train_dataset = dataloaders['train'].dataset

    for i in range(5):
        idx = random.randint(0, len(train_dataset) - 1)
        original_image = train_dataset.dataset[idx]["image"]
        category = train_dataset.dataset[idx]["label"]
        visualize_noise_effect(
            original_image,
            os.path.join(output_dir, "noise_examples"),
            f"sample_{i}_category_{category}",
            noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5]
        )

    augmentation_dir = os.path.join(output_dir, "augmentation_examples")
    os.makedirs(augmentation_dir, exist_ok=True)

    idx = random.randint(0, len(train_dataset.dataset) - 1)
    original_image = train_dataset.dataset[idx]["image"]
    category = train_dataset.dataset[idx]["label"]

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    aug_transform = get_augmentation_transforms()
    for i in range(5):
        aug_image = aug_transform(original_image.copy())
        aug_pil = transforms.ToPILImage()(aug_image)
        axes[i+1].imshow(aug_pil)
        axes[i+1].set_title(f"Augmented #{i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(augmentation_dir, f"augmentations_category_{category}.jpg")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
