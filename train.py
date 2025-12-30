import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from MultiModalClassifier import MultiModalClassifier

# -------------------------
# Fix random seed (must be called before model and DataLoader initialization)
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# -------------------------
# Dataset (example: two folders benign / malignant with jpg files)
# -------------------------
class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = ['benign', 'malignant']
        self.samples = []
        self.transform = transform

        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        return img_tensor, label


# -------------------------
# Self-supervised loss functions
# -------------------------


def reconstruction_loss(img, mask):
    """
    img: [B,3,H,W], mask: [B,1,H,W] in [0,1]
    Encourage img * mask to approximate img
    """
    img_rec = img * mask
    return F.l1_loss(img_rec, img)

def boundary_loss(img, mask):
    """
    Enforce mask boundary alignment with image edges using Sobel filters
    """
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)

    img_gray = img.mean(1, keepdim=True)
    Ix = F.conv2d(img_gray, sobel_x, padding=1)
    Iy = F.conv2d(img_gray, sobel_y, padding=1)

    Mx = F.conv2d(mask, sobel_x, padding=1)
    My = F.conv2d(mask, sobel_y, padding=1)

    return F.l1_loss(Ix, Mx) + F.l1_loss(Iy, My)

def affinity_loss(img, mask):
    """
    Compare local gradient magnitude of image and mask
    """
    img_gray = img.mean(1, keepdim=True)
    dx_img = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    dy_img = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    grad_img = torch.zeros_like(mask)
    grad_img[:, :, :-1, :] += dx_img
    grad_img[:, :, :, :-1] += dy_img

    dx_m = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
    dy_m = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    grad_m = torch.zeros_like(mask)
    grad_m[:, :, :-1, :] += dx_m
    grad_m[:, :, :, :-1] += dy_m

    grad_img = grad_img / (grad_img.mean(dim=[2, 3], keepdim=True) + 1e-8)
    grad_m = grad_m / (grad_m.mean(dim=[2, 3], keepdim=True) + 1e-8)

    return F.mse_loss(grad_img, grad_m)



# -------------------------
# Training and evaluation functions
# -------------------------
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    w_rec=1.0, w_aff=0.2, w_edge=0.1):
    model.train()
    preds, labels = [], []
    total_loss = 0.0

    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs, mask_logits, extras = model(img)
        cls_loss = criterion(outputs, label)

        mask = torch.sigmoid(mask_logits)

        L_rec = reconstruction_loss(img, mask)
        L_aff = affinity_loss(img, mask)
        L_edge = boundary_loss(img, mask)

        unsup_loss = w_rec * L_rec + w_aff * L_aff + w_edge * L_edge
        loss = cls_loss + unsup_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())

    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    f1 = f1_score(labels, preds, average='macro') if len(labels) > 0 else 0.0
    precision = precision_score(labels, preds, average='macro') if len(labels) > 0 else 0.0
    recall = recall_score(labels, preds, average='macro') if len(labels) > 0 else 0.0

    return total_loss / len(dataloader), acc, f1, precision, recall

def test_one_epoch(model, dataloader, criterion, device,
                   w_rec=1.0, w_aff=0.2, w_edge=0.1):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)

            outputs, mask_logits, extras = model(img)
            cls_loss = criterion(outputs, label)

            total_loss += cls_loss.item()
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    f1 = f1_score(labels, preds, average='macro') if len(labels) > 0 else 0.0
    precision = precision_score(labels, preds, average='macro') if len(labels) > 0 else 0.0
    recall = recall_score(labels, preds, average='macro') if len(labels) > 0 else 0.0

    return total_loss / len(dataloader), acc, f1, precision, recall


# -------------------------
# Main training script
# -------------------------
if __name__ == "__main__":
    root_dir = "Dataset/TN/TN5000/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MedicalDataset(root_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found under {root_dir}. "
            f"Expected structure: root/benign/*.jpg root/malignant/*.jpg"
        )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = MultiModalClassifier(
        num_classes=2, base_channels=32, num_scales=5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    log_list = []
    best_val_f1 = 0.0

    w_rec = 0.5
    w_aff = 1.0
    w_edge = 0.01

    n_epochs = 150
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_f1, train_prec, train_rec = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            w_rec=w_rec, w_aff=w_aff, w_edge=w_edge
        )

        val_loss, val_acc, val_f1, val_prec, val_rec = test_one_epoch(
            model, test_loader, criterion, device,
            w_rec=w_rec, w_aff=w_aff, w_edge=w_edge
        )

        print(
            f"Epoch [{epoch}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} "
            f"Prec: {train_prec:.4f} Rec: {train_rec:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} "
            f"Prec: {val_prec:.4f} Rec: {val_rec:.4f}"
        )

        log_list.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec
        })

        pd.DataFrame(log_list).to_csv(
            "validation_log_selfsup.csv", index=False
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_selfsup_model.pth")
            print(
                f"Saved best model at epoch {epoch} "
                f"with val_f1={val_f1:.4f}"
            )

    torch.save(model.state_dict(), "last_selfsup_model.pth")
    print("Training finished and models are saved.")
