import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(batch_size=32, use_augmentation=True,
                    train_dir="segmented_dataset",
                    val_dir="segmented_val_dataset"):

    train_ds = datasets.ImageFolder(
        train_dir,
        transform=aug_transform if use_augmentation else basic_transform
    )
    val_ds = datasets.ImageFolder(
        val_dir,
        transform=basic_transform
    )

    # force val dataset to match train class_to_idx
    val_ds.class_to_idx = train_ds.class_to_idx
    val_ds.classes = train_ds.classes

    # rebuild val samples to use train mapping
    new_samples = []
    for path, _ in val_ds.samples:
        cls_name = os.path.basename(os.path.dirname(path))
        label = train_ds.class_to_idx[cls_name]
        new_samples.append((path, label))
    val_ds.samples = new_samples
    val_ds.targets = [label for _, label in new_samples]

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    }
