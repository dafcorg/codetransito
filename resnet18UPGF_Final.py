import argparse
import json
import os
import random
import time
import threading
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import pynvml

# --------------------------------------------------
# Hilo para muestreo completo de métricas GPU via NVML
# --------------------------------------------------
class GPUSampler(threading.Thread):
    def __init__(self, handle, interval=0.5):
        super().__init__(daemon=True)
        self.handle = handle
        self.interval = interval
        self.samples = []  # lista de dicts con métricas
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            ts = time.time()
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.samples.append({
                'timestamp': ts,
                'power_W': power_mW / 1000.0,
                'temperature_C': temp,
                'memory_used_MB': mem.used / (1024**2),
                'memory_total_MB': mem.total / (1024**2),
                'util_gpu_pct': util.gpu,
                'util_mem_pct': util.memory
            })
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def energy_wh(self):
        if len(self.samples) < 2:
            return 0.0
        energy_j = 0.0
        for i in range(1, len(self.samples)):
            t0, p0 = self.samples[i-1]['timestamp'], self.samples[i-1]['power_W']
            t1, p1 = self.samples[i]['timestamp'],   self.samples[i]['power_W']
            dt = t1 - t0
            energy_j += ((p0 + p1) / 2) * dt
        return energy_j / 3600.0

    def dump_csv(self, filepath):
        if not self.samples:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.samples[0].keys()))
            writer.writeheader()
            writer.writerows(self.samples)

# --------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 with full instrumentation")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir, batch_size):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean,std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False,download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl

def accuracy(output,target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        return preds.eq(target).sum().item()/target.size(0)

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    loss_sum, acc_sum = 0.0, 0.0
    for imgs,labels in loader:
        imgs,labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()*imgs.size(0)
        acc_sum  += accuracy(outputs, labels)*imgs.size(0)
    n = len(loader.dataset)
    return loss_sum/n, acc_sum/n

def evaluate(model,loader,criterion,device):
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()*imgs.size(0)
            acc_sum  += accuracy(outputs, labels)*imgs.size(0)
    n = len(loader.dataset)
    return loss_sum/n, acc_sum/n

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Init NVML
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    train_dl, test_dl = get_dataloaders(args.data_dir, args.batch_size)
    model = torchvision.models.resnet18(num_classes=10)
    if torch.cuda.device_count() >= args.gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.gpus)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Start global sampler and profiler
    sampler = GPUSampler(gpu_handle, interval=0.5)
    sampler.start()
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    )
    prof.__enter__()
    t0 = time.time()

    # Full training loop
    log = []
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_dl, criterion, device)
        scheduler.step()
        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f}, acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))
        log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # End global profiler and sampler
    t1 = time.time()
    prof.__exit__(None,None,None)
    sampler.stop()
    sampler.join()

    # Compute overall metrics
    total_time = t1 - t0
    total_energy = sampler.energy_wh()
    sampler.dump_csv(os.path.join(args.output_dir, 'gpu_metrics_full.csv'))
    total_flops = sum(evt.flops for evt in prof.key_averages())
    total_gflop = total_flops / 1e9
    avg_gflops  = total_gflop / total_time

    # Save summary
    summary = {
        'total_time_s': total_time,
        'total_energy_Wh': total_energy,
        'total_GFLOP': total_gflop,
        'avg_GFLOPS': avg_gflops,
        'best_val_acc': best_acc
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training finished in {total_time:.1f}s, energy {total_energy:.2f}Wh")
    print(f"GFLOP: {total_gflop:.1f}, GFLOPS avg: {avg_gflops:.1f}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
