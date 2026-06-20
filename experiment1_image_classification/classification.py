from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common.utils import (
    AverageMeter,
    accuracy_from_logits,
    count_parameters,
    ensure_dir,
    get_device,
    limit_dataset,
    save_csv,
    save_json,
    set_seed,
)


class SyntheticClassificationDataset(Dataset):
    """Small deterministic image dataset used only for smoke tests."""

    def __init__(self, size: int, num_classes: int = 10, image_size: int = 32, channels: int = 1) -> None:
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index % self.num_classes
        gen = torch.Generator().manual_seed(index)
        x = torch.rand((self.channels, self.image_size, self.image_size), generator=gen) * 0.08
        stripe = max(2, self.image_size // 12)
        offset = (label * 3) % max(1, self.image_size - stripe)
        x[:, offset : offset + stripe, :] += 0.45
        x[:, :, offset : offset + stripe] += 0.25
        x = x.clamp(0, 1)
        return (x - 0.5) / 0.5, label


class HWDBTxtDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int = 64) -> None:
        self.root = Path(root)
        txt = self.root / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(
                f"Missing HWDB split file: {txt}. Expected lines in the form '<image_path> <label>'."
            )
        self.items: list[tuple[str, int]] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(maxsplit=1)
            self.items.append((path, int(label)))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("L")
        return self.transform(image), label


class MLP(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, c1: int, c3_reduce: int, c3: int, c5_reduce: int, c5: int, pool_proj: int) -> None:
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, c1, 1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c3_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_reduce, c3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c5_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5_reduce, c5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_ch, pool_proj, 1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception = nn.Sequential(
            InceptionBlock(192, 32, 48, 64, 8, 16, 16),
            InceptionBlock(128, 64, 64, 96, 16, 32, 32),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(224, 96, 48, 104, 8, 24, 24),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(248, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.inception(self.stem(x)))


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ResNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, blocks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.blocks = nn.Sequential(*[BasicBlock(64) for _ in range(blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


MODEL_DEPTH = {
    "mlp": 3,
    "lenet": 5,
    "alexnet": 8,
    "googlenet": 13,
    "resnet": 12,
}


def build_model(name: str, in_channels: int, image_size: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP(in_channels, image_size, num_classes)
    if name == "lenet":
        return LeNet(in_channels, num_classes)
    if name == "alexnet":
        return AlexNetSmall(in_channels, num_classes)
    if name == "googlenet":
        return GoogLeNetSmall(in_channels, num_classes)
    if name == "resnet":
        return ResNetSmall(in_channels, num_classes)
    raise ValueError(f"Unknown model: {name}")


def build_dataset(
    name: str,
    root: Path,
    train: bool,
    image_size: int,
    download: bool,
    synthetic_size: int,
) -> tuple[Dataset, int, int]:
    name = name.lower()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if name == "synthetic":
        return SyntheticClassificationDataset(synthetic_size, 10, image_size, 1), 1, 10
    if name == "mnist":
        ds = datasets.MNIST(root=str(root), train=train, transform=transform, download=download)
        return ds, 1, 10
    if name in {"fashion", "fashion_mnist", "fashion-mnist"}:
        ds = datasets.FashionMNIST(root=str(root), train=train, transform=transform, download=download)
        return ds, 1, 10
    if name == "hwdb":
        split = "train" if train else "test"
        ds = HWDBTxtDataset(root / "HWDB1", split=split, image_size=image_size)
        return ds, 1, 10
    raise ValueError(f"Unknown dataset: {name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: Callable, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), x.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))
    return loss_meter.avg, acc_meter.avg


def run_single(
    dataset_name: str,
    model_name: str,
    batch_size: int,
    args: argparse.Namespace,
    phase: str = "model_compare",
) -> dict:
    train_ds, in_channels, num_classes = build_dataset(
        dataset_name, Path(args.data_root), True, args.image_size, args.download, args.synthetic_train_size
    )
    test_ds, _, _ = build_dataset(
        dataset_name, Path(args.data_root), False, args.image_size, args.download, args.synthetic_test_size
    )
    train_ds = limit_dataset(train_ds, args.limit_train)
    test_ds = limit_dataset(test_ds, args.limit_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device(args.device)
    model = build_model(model_name, in_channels, args.image_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_train_loss = math.nan
    for _ in range(args.epochs):
        last_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = output_dir / "checkpoints" / f"{dataset_name}_{model_name}_{phase}_bs{batch_size}.pt"
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset": dataset_name,
            "model": model_name,
            "in_channels": in_channels,
            "image_size": args.image_size,
            "num_classes": num_classes,
            "phase": phase,
            "batch_size": batch_size,
        },
        checkpoint_path,
    )
    demo_path = output_dir / "demo_cases" / f"{dataset_name}_{model_name}_{phase}_bs{batch_size}.png"
    save_classification_demo(model, test_loader, device, demo_path, dataset_name)
    return {
        "phase": phase,
        "dataset": dataset_name,
        "model": model_name,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "params": count_parameters(model),
        "depth_score": MODEL_DEPTH.get(model_name, 0),
        "train_loss": round(last_train_loss, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "checkpoint": str(checkpoint_path),
        "demo_image": str(demo_path),
    }


@torch.no_grad()
def save_classification_demo(model: nn.Module, loader: DataLoader, device: torch.device, path: Path, dataset_name: str) -> None:
    ensure_dir(path.parent)
    model.eval()
    try:
        x, y = next(iter(loader))
    except StopIteration:
        return
    x = x.to(device)
    logits = model(x)
    pred = logits.argmax(dim=1).cpu().tolist()
    labels = label_names(dataset_name)
    count = min(8, x.size(0))
    grid = Image.new("RGB", (96 * count, 118), "white")
    for i in range(count):
        arr = x[i].detach().cpu()
        arr = (arr * 0.5 + 0.5).clamp(0, 1)
        if arr.size(0) == 1:
            img = Image.fromarray((arr[0].numpy() * 255).astype(np.uint8), mode="L").convert("RGB")
        else:
            img = Image.fromarray((arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode="RGB")
        tile = Image.new("RGB", (96, 118), "white")
        tile.paste(img.resize((64, 64), Image.Resampling.NEAREST), (16, 4))
        true_label = labels[int(y[i])] if int(y[i]) < len(labels) else str(int(y[i]))
        pred_label = labels[pred[i]] if pred[i] < len(labels) else str(pred[i])
        draw_text(tile, f"P:{pred_label}", (6, 74))
        draw_text(tile, f"T:{true_label}", (6, 92))
        grid.paste(tile, (96 * i, 0))
    grid.save(path)


def draw_text(image: Image.Image, text: str, xy: tuple[int, int]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.text(xy, text, fill=(0, 0, 0))


def label_names(dataset_name: str) -> list[str]:
    name = dataset_name.lower()
    if name in {"fashion", "fashion_mnist", "fashion-mnist"}:
        return ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return [str(i) for i in range(10)]


def run_experiment(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict] = []
    for dataset_name in args.datasets:
        try:
            build_dataset(
                dataset_name,
                Path(args.data_root),
                True,
                args.image_size,
                args.download,
                args.synthetic_train_size,
            )
        except Exception as exc:
            if not args.skip_missing:
                raise
            row = {"phase": "dataset_check", "dataset": dataset_name, "status": "skipped", "error": str(exc)}
            print(row)
            rows.append(row)
            save_csv(rows, output_dir / "classification_results.csv")
            save_json({"results": rows}, output_dir / "classification_results.json")
            continue
        for model_name in args.models:
            row = run_single(dataset_name, model_name, args.batch_size, args)
            print(row)
            rows.append(row)
            save_csv(rows, output_dir / "classification_results.csv")
            save_json({"results": rows}, output_dir / "classification_results.json")
        for batch_size in args.batch_sizes:
            row = run_single(dataset_name, args.batch_sweep_model, batch_size, args, phase="batch_sweep")
            print(row)
            rows.append(row)
            save_csv(rows, output_dir / "classification_results.csv")
            save_json({"results": rows}, output_dir / "classification_results.json")
    save_csv(rows, output_dir / "classification_results.csv")
    save_json({"results": rows}, output_dir / "classification_results.json")
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 1: image classification.")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "hwdb"])
    parser.add_argument("--models", nargs="+", default=["mlp", "lenet", "alexnet", "googlenet", "resnet"])
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", default="outputs/exp1")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[16, 32, 64])
    parser.add_argument("--batch-sweep-model", default="lenet")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--synthetic-train-size", type=int, default=256)
    parser.add_argument("--synthetic-test-size", type=int, default=128)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
