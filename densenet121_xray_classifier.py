"""
Train a DenseNet121 model pretrained on chest X-rays (CheXNet weights) using TorchXRayVision.
Adapted for custom organ classification (heart, lungs, bone, etc.)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------- MODEL CLASS (importable) ----------
class OrganClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------- MAIN TRAINING SCRIPT ----------
if __name__ == "__main__":

    # ---------- CONFIG ----------
    DATA_DIR = "dataset"  # must contain train/, val/, test/ subfolders
    BATCH_SIZE = 16
    NUM_EPOCHS_HEAD = 5      # first stage (frozen base)
    NUM_EPOCHS_FINE = 10     # fine-tuning stage
    LEARNING_RATE_HEAD = 1e-3
    LEARNING_RATE_FINE = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------------------------

    # ----------- DATASET PREP -----------
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    # ----------- MODEL SETUP -----------
    print("Loading DenseNet121 pretrained on CheXpert...")

    base = xrv.models.DenseNet(weights="densenet121-res224-chex")
    features = base.features

    model = OrganClassifier(features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # ----------- TRAINING STAGE 1 (FROZEN BASE) -----------
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_HEAD)

    print("\n--- Stage 1: Training classifier head ---")
    for epoch in range(NUM_EPOCHS_HEAD):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_HEAD}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_HEAD}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")

    # ----------- TRAINING STAGE 2 (FINE-TUNE) -----------
    print("\n--- Stage 2: Fine-tuning deeper layers ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE)

    for epoch in range(NUM_EPOCHS_FINE):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Fine-Tune {epoch+1}/{NUM_EPOCHS_FINE}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Fine-Tune [{epoch+1}/{NUM_EPOCHS_FINE}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")

    # ----------- EVALUATION & SAVE -----------
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    model_save_path = "densenet121_xray_organs.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
