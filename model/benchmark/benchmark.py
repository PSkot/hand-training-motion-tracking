# Hand Detection and Pose Estimation - Benchmark-Ready Version

import os, json, torch, requests, zipfile, time, sys, csv
import numpy as np
import psutil, threading, cpuinfo
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tabulate import tabulate

# ---------------------------- Parameters ----------------------------
data_path = "../dataset"
batch_size = 64
learning_rate = 1e-4
epochs = 3
subset_size = None  # Only applies to full training
quick_benchmark_subset = 100000
val_size = 0.1
use_mixed_precision = (
    True  # Unsupported systems will still use normal training if set to True
)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []


# ---------------------------- Benchmark Decorator ----------------------------
def benchmark_kpi(tag="", table=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            process = psutil.Process(os.getpid())

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # --- CPU usage sampling ---
            cpu_usage = []
            done = [False]

            def monitor_cpu():
                while not done[0]:
                    usage = psutil.cpu_percent(interval=0.1)
                    cpu_usage.append(usage)

            monitor = threading.Thread(target=monitor_cpu)
            monitor.start()

            result = func(*args, **kwargs)  # Run the training

            done[0] = True
            monitor.join()
            avg_cpu = round(np.mean(cpu_usage), 1) if cpu_usage else "-"
            if str(device) == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_gpu = torch.cuda.max_memory_allocated() / 1024**3
                device_name = torch.cuda.get_device_name(0)
            else:
                mem_gpu = 0
                device_name = cpuinfo.get_cpu_info().get("brand_raw", "CPU")
            end = time.time()
            mem_cpu = process.memory_info().rss / 1024**3  # in GB

            entry = {
                "Tag": tag,
                "Time (s)": round(end - start, 2),
                "Samples/sec": (
                    round(len(args[0].dataset) / (end - start), 2)
                    if hasattr(args[0], "dataset")
                    else "-"
                ),
                "GPU Mem (GB)": round(mem_gpu, 2) if mem_gpu else "-",
                "CPU Mem (GB)": round(mem_cpu, 2),
                "CPU Load (%)": avg_cpu,
                "CPU Threads Used": torch.get_num_threads(),
                "Device": device_name,
            }
            table.append(entry)
            return result

        return wrapper

    return decorator


# ---------------------------- Data Download ----------------------------
datasets = {
    "calib": "https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_calib_meta.zip",
    "xyz": "https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_xyz.zip",
    "rgb_merged": "https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_rgb_merged.zip",
}


def download_data(base_path=data_path):
    print("Downloading and extracting data (if needed)...")
    for folder, url in datasets.items():
        zip_path = os.path.join(base_path, f"{folder}.zip")
        if not os.path.exists(os.path.join(base_path, folder)):
            print(f"Downloading {folder}...")
            r = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                total_size = int(r.headers.get("content-length", 0))
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024**3,
                    desc=folder,
                    ncols=100,
                ) as t:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        t.update(len(chunk))
            print(f"Extracting {folder}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_path)
            os.remove(zip_path)
    print("All datasets ready.")


# ---------------------------- Dataset ----------------------------
class HandDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.cameras = [f"cam{n}" for n in range(8)]
        self.data = [
            (h, f.split(".")[0], c)
            for h in os.listdir(os.path.join(root, "rgb_merged"))
            for f in os.listdir(os.path.join(root, "rgb_merged", h, "cam0"))
            for c in self.cameras
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, f, c = self.data[idx]
        img_path = os.path.join(self.root, "rgb_merged", h, c, f"{f}.jpg")
        image = read_image(img_path).float() / 255.0
        with open(os.path.join(self.root, "xyz", h, f"{f}.json")) as jf:
            landmarks = torch.tensor(json.load(jf), dtype=torch.float32).view(-1)
        return {"image": image, "landmarks": landmarks}


# ---------------------------- Model ----------------------------
def get_model():
    print("Initializing model...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier[-1].in_features, 63)
    return model


# ---------------------------- MPT check ---------------------------
def supports_mixed_precision():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7


# ---------------------------- Training ----------------------------
@benchmark_kpi(tag="mobilenet_bs64", table=results)
def train(model, train_loader, val_loader):
    print("Starting training...")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    transforms_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    if supports_mixed_precision():
        scaler = torch.amp.GradScaler(
            device.type
        )  # torch.amp for mixed precision training. Might not be supported on older GPUs

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
            ncols=100,
            leave=True,
        )
        for batch in loop:
            imgs = transforms_train(batch["image"]).to(device)
            labels = batch["landmarks"].to(device)
            optimizer.zero_grad()

            if supports_mixed_precision():
                with torch.amp.autocast(device.type):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss = evaluate(model, val_loader)
        total_loss /= len(train_loader)
        print(
            f"Epoch {epoch+1}: Train Loss = {total_loss:.4f} | Val Loss = {val_loss:.4f}"
        )


# ---------------------------- Evaluation ----------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Evaluating", unit="batch", ncols=100, leave=True
        ):
            imgs = transforms_test(batch["image"]).to(device)
            labels = batch["landmarks"].to(device).view(-1, 63)
            preds = model(imgs)
            total_loss += criterion(preds, labels).item()
    return total_loss / len(loader)


# ---------------------------- Quick Benchmark ----------------------------
def quick_benchmark():
    print("Running quick benchmark...")
    torch.manual_seed(42)
    dataset = HandDataset(data_path)
    n = quick_benchmark_subset
    dataset = Subset(dataset, range(n))
    train_set, val_set = random_split(
        dataset, [n - int(val_size * n), int(val_size * n)]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    model = get_model()
    train(model, train_loader, val_loader)
    print("\nBenchmark Results:")
    print(tabulate(results, headers="keys"))
    csv_file = os.path.join(
        os.path.dirname(__file__), "hand_hardware_benchmark_results.csv"
    )
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)


# ---------------------------- Main ----------------------------
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hand Pose Training & Benchmarking — supports CPU/GPU control and threaded performance tuning."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run quick benchmark instead of full training.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if GPU is available.",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Force GPU execution (default if available)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads per operation (default: 4).",
    )
    parser.add_argument(
        "--interop",
        type=int,
        default=1,
        help="Number of inter-op CPU threads (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Thread control
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)
    # Device control
    global device
    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Threads: {args.threads}, Inter-op: {args.interop}")
    download_data()
    if args.benchmark:
        quick_benchmark()
        return
    print("Starting pipeline...")
    download_data()
    if "--benchmark" in sys.argv:
        quick_benchmark()
        return
    print("Loading dataset...")
    dataset = HandDataset(data_path)
    if subset_size:
        dataset = Subset(dataset, range(subset_size))
    n = len(dataset)
    print(f"Dataset size: {n}")
    train_set, val_set = random_split(
        dataset, [n - int(val_size * n), int(val_size * n)]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    model = get_model()
    train(model, train_loader, val_loader)
    print("\nBenchmark Results:")
    print(tabulate(results, headers="keys"))


if __name__ == "__main__":
    main()
