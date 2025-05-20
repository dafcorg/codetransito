#!/usr/bin/env python
"""Train ResNet‑18 on CIFAR‑10 with minimal dependencies.
Usage (single‑GPU):
    python train_resnet18_cifar10.py --epochs 90 --batch_size 256
Usage (multi‑GPU DataParallel):
    python train_resnet18_cifar10.py --epochs 90 --batch_size 256 --gpus 2
The script will:
    • download CIFAR‑10 to <data_dir> (default: ./data)
    • save checkpoints to <output_dir> (default: ./runs)
    • log metrics per epoch in JSON (<output_dir>/metrics.json)
Monitor
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,power.limit --format=csv -l 5 >> gpu_metrics.csv
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet‑18 on CIFAR‑10")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs (DataParallel)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(data_dir, batch_size):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl


def accuracy(output, target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
        return correct / target.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(outputs, labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(outputs, labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, test_dl = get_dataloaders(args.data_dir, args.batch_size)

    model = torchvision.models.resnet18(num_classes=10)
    if torch.cuda.device_count() >= args.gpus and args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    log = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_dl, criterion, device)
        scheduler.step()

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        log.append(epoch_stats)
        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "resnet18_cifar10_best.pt"))

    # final save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "resnet18_cifar10_last.pt"))

    # write metrics
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"metrics_{stamp}.json"), "w") as fp:
        json.dump(log, fp, indent=2)

    print(f"Training finished. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
