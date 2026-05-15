"""
Plant Disease Detection - PyTorch Training Script
===================================================
Downloads PlantVillage dataset via kagglehub and trains ResNet18
with transfer learning for accurate 38-class disease classification.
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / 'saved_models'
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 224


def download_dataset():
    """Download PlantVillage dataset via kagglehub."""
    print("=" * 60)
    print("Downloading PlantVillage Dataset via Kaggle...")
    print("=" * 60)

    import kagglehub
    path = kagglehub.dataset_download("abdulhasibuddin/plant-doc-dataset")
    print(f"Dataset downloaded to: {path}")

    # Find the directory with class folders
    dataset_path = Path(path)
    # Look for the directory containing class subdirectories
    for candidate in [
        dataset_path / 'PlantDoc-Dataset' / 'train',
        dataset_path / 'PlantVillage',
        dataset_path / 'plantvillage dataset' / 'color',
        dataset_path / 'color',
        dataset_path,
    ]:
        if candidate.exists() and any(candidate.iterdir()):
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if len(subdirs) >= 10:  # Should have ~38 class directories
                print(f"Found dataset at: {candidate} ({len(subdirs)} classes)")
                return str(candidate)

    # Walk deeper
    for root, dirs, files in os.walk(str(dataset_path)):
        if len(dirs) >= 10:
            has_images = False
            for d in dirs[:3]:
                subdir = Path(root) / d
                imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.JPG")) + list(subdir.glob("*.png"))
                if imgs:
                    has_images = True
                    break
            if has_images:
                print(f"Found dataset at: {root} ({len(dirs)} classes)")
                return root

    print(f"Could not auto-detect class folders. Contents of {dataset_path}:")
    for item in dataset_path.rglob("*"):
        if item.is_dir():
            print(f"  DIR: {item.relative_to(dataset_path)}")
    sys.exit(1)


def create_data_loaders(dataset_path):
    """Create train/val data loaders with augmentation."""

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    print(f"\nFound {len(full_dataset)} images in {len(full_dataset.classes)} classes")
    for i, cls in enumerate(full_dataset.classes):
        print(f"  {i:2d}: {cls}")

    class_to_idx = full_dataset.class_to_idx

    # Limit dataset size for rapid CPU training in this demonstration
    MAX_IMAGES = 5000
    if len(full_dataset) > MAX_IMAGES:
        print(f"Subsampling dataset to {MAX_IMAGES} images for rapid CPU training...")
        indices = torch.randperm(len(full_dataset))[:MAX_IMAGES]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        
    # Split 80/20
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Use the subset's indices for validation as well
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Training: {train_size} | Validation: {val_size}")
    return train_loader, val_loader, class_to_idx


def build_model(num_classes):
    """Build ResNet18 with transfer learning."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze early layers, keep later layers trainable
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, device):
    """Train the model with progress logging."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.1f}%", flush=True)

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        print(f"{'='*60}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  >> New best! Val Acc: {val_acc:.1f}%")

        scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_acc


def save_model(model, class_to_idx, best_acc):
    """Save model and metadata."""
    model_path = MODEL_DIR / 'disease_model_pytorch.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'num_classes': len(class_to_idx),
        'image_size': IMAGE_SIZE,
        'accuracy': best_acc,
    }, str(model_path))
    print(f"\nModel saved: {model_path} ({os.path.getsize(str(model_path))/(1024*1024):.1f} MB)")

    # Save class indices
    idx_path = MODEL_DIR / 'class_indices.json'
    with open(str(idx_path), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    # Save config
    config = {
        'model_type': 'ResNet18_PyTorch',
        'input_shape': [IMAGE_SIZE, IMAGE_SIZE, 3],
        'num_classes': len(class_to_idx),
        'classes': list(class_to_idx.keys()),
        'best_accuracy': best_acc,
        'framework': 'pytorch',
    }
    config_path = MODEL_DIR / 'disease_model_config.json'
    with open(str(config_path), 'w') as f:
        json.dump(config, f, indent=2)
    print("Metadata saved.")


def main():
    print("=" * 60)
    print("AGRIZONE - Plant Disease Model Training (PyTorch)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset_path = download_dataset()
    train_loader, val_loader, class_to_idx = create_data_loaders(dataset_path)

    num_classes = len(class_to_idx)
    model = build_model(num_classes).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    print(f"\nStarting training ({NUM_EPOCHS} epochs)...\n")
    model, best_acc = train_model(model, train_loader, val_loader, device)

    save_model(model, class_to_idx, best_acc)
    print(f"\nDONE! Best Val Accuracy: {best_acc:.1f}%")


if __name__ == '__main__':
    main()
