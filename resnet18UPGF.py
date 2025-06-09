#python -m venv venv
#source venv/bin/activate      
#pip install torch torchvision tqdm pynvml nvidia-ml-py3
#mkdir -p ./runs
'''
python train_instrumented.py \
  --epochs 90 \
  --batch_size 128 \
  --lr 0.1 \
  --data_dir ./data \
  --output_dir ./runs \
  --gpus 1 \
  --seed 42
'''

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
from tqdm import tqdm
import pynvml
from torch.profiler import profile, record_function, ProfilerActivity

# --------------------------------------------------
# Hilo para muestreo completo de métricas GPU via NVML
# --------------------------------------------------
class GPUSampler(threading.Thread):
    def __init__(self, handle, interval=0.5):
        super().__init__(daemon=True)
        self.handle = handle
        self.interval = interval
        self.samples = []  # lista de dicts con métricas
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
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
        self._stop.set()

    def energy_wh(self):
        # integra potencia para obtener Wh
        if len(self.samples) < 2:
            return 0.0
        energy_j = 0.0
        for i in range(1, len(self.samples)):
            t0, p0 = self.samples[i-1]['timestamp'], self.samples[i-1]['power_W']
            t1, p1 = self.samples[i]['timestamp'],    self.samples[i]['power_W']
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
# Funciones de utilidad
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 (instrumented)")
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
    test_ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False,download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl


def accuracy(output, target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        return preds.eq(target).sum().item() / target.size(0)


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
        running_acc  += accuracy(outputs, labels) * imgs.size(0)
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
            running_acc  += accuracy(outputs, labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Inicializar NVML
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

    log = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Iniciar muestreo de GPU
        sampler = GPUSampler(gpu_handle, interval=0.5)
        sampler.start()

        # Cronometrar y perfilar FLOPs en entrenamiento
        t0 = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        t1 = time.time()

        # Parar sampler y calcular energía
        sampler.stop()
        sampler.join()
        epoch_time   = t1 - t0
        epoch_energy = sampler.energy_wh()

        # Volcar métricas GPU a CSV
        csv_path = os.path.join(args.output_dir, f"epoch_{epoch}_gpu_metrics.csv")
        sampler.dump_csv(csv_path)

        # Extraer FLOPs
        total_flops       = sum([evt.flops for evt in prof.key_averages()])
        epoch_gflop       = total_flops / 1e9
        gflops_throughput = epoch_gflop / epoch_time

        val_loss, val_acc = evaluate(model, test_dl, criterion, device)
        scheduler.step()

        stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
            "epoch_energy_Wh": epoch_energy,
            "epoch_gflop": epoch_gflop,
            "epoch_gflops": gflops_throughput,
            "gpu_metrics_csv": csv_path,
            "lr": scheduler.get_last_lr()[0]
        }
        log.append(stats)

        print(f"Train   → loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val     → loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print(f"Time    → {epoch_time:.1f}s | Energy → {epoch_energy:.2f}Wh")
        print(f"FLOPs   → {epoch_gflop:.1f} GFLOP | Throughput → {gflops_throughput:.1f} GFLOPS")
        print(f"GPU CSV → {csv_path}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))

    # Guardar métricas finales
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"metrics_{stamp}.json"), "w") as fp:
        json.dump(log, fp, indent=2)

    print(f"Training completed. Best val acc: {best_acc:.4f}")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
