import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedKFold

# set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# hyperparams
NUM_CLASSES = 100
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 60
LR = 0.15
WD = 5e-4
MIXUP_ALPHA = 0.2
LABEL_SMOOTH = 0.1
NUM_WORKERS = 8
PATIENCE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# dataset class for training images
class ButterflyDataset(Dataset):
    def __init__(self, df, img_dir, label2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["file_name"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[row["TARGET"]]
        return img, label


# dataset class for test images (no labels)
class TestDataset(Dataset):
    def __init__(self, img_dir, image_ids, transform=None):
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id


# data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

# no augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# test time augmentation - average predictions over different views
tta_transforms = [
    val_transform,
    transforms.Compose([  # horizontal flip
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # center crop from larger size
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # center crop + flip
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def build_model():
    # using torchvision resnet18 with NO pretrained weights
    model = resnet18(weights=None, num_classes=NUM_CLASSES)

    # add dropout before final fc layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    # kaiming init for better convergence
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    return model


# mixup: blend pairs of images and labels for regularization
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

        out = model(imgs_mix)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        total += labels.size(0)
        correct += lam * preds.eq(y_a).sum().item() + (1 - lam) * preds.eq(y_b).sum().item()

    scheduler.step()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)

        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return total_loss / total, 100.0 * correct / total


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ResNet-18 Training Curves - Butterfly Classification", fontsize=15, fontweight="bold")

    # loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#2196F3", lw=1.5)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val", color="#F44336", lw=1.5)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # accuracy
    axes[0, 1].plot(epochs, history["train_acc"], label="Train", color="#2196F3", lw=1.5)
    axes[0, 1].plot(epochs, history["val_acc"], label="Val", color="#F44336", lw=1.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # lr schedule
    axes[1, 0].plot(epochs, history["lr"], color="#4CAF50", lw=1.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("LR Schedule (Cosine Warm Restarts)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # val accuracy zoomed in
    axes[1, 1].plot(epochs, history["val_acc"], color="#F44336", lw=1.5)
    best_ep = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"])
    axes[1, 1].axhline(y=best_acc, color="gray", ls="--", alpha=0.5)
    axes[1, 1].axvline(x=best_ep, color="gray", ls="--", alpha=0.5)
    axes[1, 1].scatter([best_ep], [best_acc], color="#FF9800", s=100, zorder=5,
                       label=f"Best: {best_acc:.2f}% (ep {best_ep})")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Val Accuracy (zoomed)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(max(0, min(history["val_acc"]) - 5), 100)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_img_dir = os.path.join(script_dir, "train_images")
    test_img_dir = os.path.join(script_dir, "test_images")

    train_df = pd.read_csv(os.path.join(script_dir, "train.csv"))
    sub_df = pd.read_csv(os.path.join(script_dir, "sample_submission.csv"))

    classes = sorted(train_df["TARGET"].unique())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    print(f"{len(classes)} classes, {len(train_df)} training images")

    # 85/15 stratified split
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(train_df, train_df["TARGET"]))
    train_sub = train_df.iloc[train_idx]
    val_sub = train_df.iloc[val_idx]
    print(f"Train: {len(train_sub)} | Val: {len(val_sub)}")

    train_dataset = ButterflyDataset(train_sub, train_img_dir, label2idx, train_transform)
    val_dataset = ButterflyDataset(val_sub, train_img_dir, label2idx, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    model = build_model().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = 0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        v_loss, v_acc = validate(model, val_loader, criterion)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["lr"].append(lr_now)

        marker = ""
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(script_dir, "best_model.pth"))
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(f"Ep {epoch:3d}/{EPOCHS} | LR {lr_now:.5f} | "
              f"Train {t_loss:.4f} {t_acc:.1f}% | Val {v_loss:.4f} {v_acc:.1f}%{marker}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    plot_curves(history, os.path.join(script_dir, "training_curves.png"))

    # generate predictions with TTA
    print("\nRunning TTA on test set...")
    model.load_state_dict(torch.load(os.path.join(script_dir, "best_model.pth"), map_location=device))
    model.eval()

    test_ids = sub_df["ID"].tolist()
    all_probs = torch.zeros(len(test_ids), NUM_CLASSES)

    for i, tta_tf in enumerate(tta_transforms):
        test_ds = TestDataset(test_img_dir, test_ids, transform=tta_tf)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        probs = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                out = model(imgs.to(device))
                probs.append(torch.softmax(out, dim=1).cpu())
        all_probs += torch.cat(probs, dim=0)
        print(f"  TTA {i+1}/{len(tta_transforms)} done")

    all_probs /= len(tta_transforms)
    preds = all_probs.argmax(dim=1).numpy()

    # save submission
    sub_df = sub_df.rename(columns={"ID": "image_id"})
    sub_df["label"] = [idx2label[p] for p in preds]
    sub_df = sub_df[["image_id", "label"]]
    out_path = os.path.join(script_dir, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
