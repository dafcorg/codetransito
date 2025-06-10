import os
import time
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models

# --- Reproducibility settings ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_imagenet_dataloaders(data_dir, batch_size=256, num_workers=4):
    # Data augmentation and normalization for training
    # Just normalization for validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=90, device='cuda'):
    model.to(device)
    overall_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                # Re-seed per batch to ensure reproducibility of augmentation
                torch.manual_seed(SEED + epoch)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        epoch_end = time.time()
        print(f'Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds')
        print()

    overall_end = time.time()
    print(f'Training complete in {(overall_end - overall_start):.2f} seconds')
    return model


if __name__ == '__main__':
    # Paths and hyperparameters
    data_dir = '/path/to/imagenet'
    batch_size = 256
    num_workers = 8
    num_epochs = 90
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    step_size = 30  # Decay LR every 30 epochs
    gamma = 0.1

    # Set seeds for reproducibility again before data loading
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Dataloaders
    train_loader, val_loader = get_imagenet_dataloaders(data_dir, batch_size, num_workers)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Model, criterion, optimizer, scheduler
    model = models.resnet50(pretrained=False, num_classes=1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train
    trained_model = train_model(model, dataloaders, criterion,
                                optimizer, scheduler, num_epochs)

    # Save checkpoint
    torch.save(trained_model.state_dict(), 'resnet50_imagenet.pth')
