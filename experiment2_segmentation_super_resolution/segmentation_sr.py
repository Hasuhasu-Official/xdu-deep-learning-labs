from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.utils import AverageMeter, ensure_dir, get_device, save_csv, save_json, set_seed


MSRC_COLORS = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 64, 128],
        [0, 192, 128],
        [128, 192, 128],
        [64, 64, 0],
        [192, 64, 0],
    ],
    dtype=np.uint8,
)
NUM_SEG_CLASSES = 21
IGNORE_INDEX = 255


def rgb_to_msrc_mask(rgb: np.ndarray) -> np.ndarray:
    mask = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for color_index, color in enumerate(MSRC_COLORS[1:], start=1):
        matched = np.all(rgb == color, axis=-1)
        mask[matched] = color_index - 1
    return mask


def imread_rgb(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def random_square_crop(image: np.ndarray, mask: np.ndarray, min_scale: float = 0.65) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    side = min(h, w)
    crop = int(side * random.uniform(min_scale, 1.0))
    crop = max(16, min(crop, side))
    top = 0 if h == crop else random.randint(0, h - crop)
    left = 0 if w == crop else random.randint(0, w - crop)
    return image[top : top + crop, left : left + crop], mask[top : top + crop, left : left + crop]


def augment_seg_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image, mask = random_square_crop(image, mask)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.8:
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-18.0, 18.0)
        image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
    return image, mask


class MSRCSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 128,
        augment: bool = False,
        samples_per_image: int = 1,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self._find_dir(["images", "Images", "image", "JPEGImages"])
        self.gt_dir = self._find_dir(["gt", "GT", "groundtruth", "GroundTruth", "labels", "Labels"])
        split_file = self.root / f"{split}.txt"
        if split_file.exists():
            files = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            files = sorted(p.name for p in self.image_dir.glob("*.bmp"))
        if not files:
            raise FileNotFoundError(f"No MSRC images found under {self.root}.")
        self.files = files
        self.image_size = image_size
        self.augment = augment
        self.samples_per_image = max(1, int(samples_per_image))

    def _find_dir(self, names: list[str]) -> Path:
        for name in names:
            path = self.root / name
            if path.exists():
                return path
        lowered = {p.name.lower(): p for p in self.root.iterdir() if p.is_dir()} if self.root.exists() else {}
        for name in names:
            path = lowered.get(name.lower())
            if path is not None:
                return path
        raise FileNotFoundError(f"Could not find one of {names} under {self.root}.")

    def __len__(self) -> int:
        return len(self.files) * self.samples_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.files[index % len(self.files)]
        image_path = self.image_dir / name
        gt_path = self.gt_dir / name.replace(".bmp", "_GT.bmp")
        if not gt_path.exists():
            gt_path = self.gt_dir / name
        image = imread_rgb(image_path)
        gt = imread_rgb(gt_path)
        if image is None or gt is None:
            raise FileNotFoundError(f"Missing image or mask for {name}.")
        if self.augment:
            image, gt = augment_seg_pair(image, gt)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = rgb_to_msrc_mask(gt).astype(np.int64)
        x = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        x = (x - 0.5) / 0.5
        y = torch.from_numpy(mask).long()
        return x, y


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, size: int, image_size: int = 64, num_classes: int = NUM_SEG_CLASSES) -> None:
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(index)
        h = w = self.image_size
        mask = np.zeros((h, w), dtype=np.int64)
        for cls in range(1, min(self.num_classes, 7)):
            x0 = int(rng.integers(0, max(1, w - 12)))
            y0 = int(rng.integers(0, max(1, h - 12)))
            x1 = min(w, x0 + int(rng.integers(8, max(9, w // 2))))
            y1 = min(h, y0 + int(rng.integers(8, max(9, h // 2))))
            mask[y0:y1, x0:x1] = cls
        rgb = MSRC_COLORS[(mask + 1).clip(0, len(MSRC_COLORS) - 1)].astype(np.float32) / 255.0
        rgb += rng.normal(0, 0.03, size=rgb.shape).astype(np.float32)
        rgb = np.clip(rgb, 0, 1)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        return (x - 0.5) / 0.5, torch.from_numpy(mask).long()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_SEG_CLASSES, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = UpBlock(base * 16, base * 8, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.up4(b, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.out(x)


def segmentation_miou(logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_SEG_CLASSES) -> float:
    pred = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX
    ious: list[float] = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        target_c = (target == cls) & valid
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


class SegmentationIoUMeter:
    def __init__(self, num_classes: int = NUM_SEG_CLASSES) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = (target != IGNORE_INDEX) & (target >= 0) & (target < self.num_classes)
        if not bool(valid.any()):
            return
        encoded = target[valid].reshape(-1) * self.num_classes + pred[valid].reshape(-1).clamp(0, self.num_classes - 1)
        hist = torch.bincount(encoded.long(), minlength=self.num_classes * self.num_classes)
        self.matrix += hist.reshape(self.num_classes, self.num_classes).double()

    def miou(self) -> float:
        intersection = torch.diag(self.matrix)
        union = self.matrix.sum(dim=1) + self.matrix.sum(dim=0) - intersection
        present = union > 0
        if not bool(present.any()):
            return 0.0
        return float((intersection[present] / union[present].clamp_min(1.0)).mean().item())


@torch.no_grad()
def evaluate_segmentation_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dice_weight: float,
) -> tuple[float, float]:
    loss_meter = AverageMeter()
    iou_meter = SegmentationIoUMeter()
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if dice_weight > 0:
            loss = loss + dice_weight * soft_dice_loss(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        iou_meter.update(logits, y)
    return loss_meter.avg, iou_meter.miou()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_SEG_CLASSES) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    probs = torch.softmax(logits, dim=1)
    safe_target = target.clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1)
    probs = probs * valid_mask
    one_hot = one_hot * valid_mask
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    present = one_hot.sum(dims) > 0
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice[present].mean()


def compute_seg_class_weights(dataset: Dataset, num_classes: int = NUM_SEG_CLASSES) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for index in range(len(dataset)):
        _, y = dataset[index]
        valid = y != IGNORE_INDEX
        if bool(valid.any()):
            counts += torch.bincount(y[valid].reshape(-1), minlength=num_classes).double()
    if counts.sum() <= 0:
        return torch.ones(num_classes, dtype=torch.float32)
    freq = counts / counts.sum()
    weights = 1.0 / torch.log(1.02 + freq)
    weights[counts == 0] = 0.0
    present = weights > 0
    weights[present] = weights[present] / weights[present].mean()
    return weights.float()


def build_seg_dataset(name: str, root: Path, split: str, image_size: int, synthetic_size: int) -> Dataset:
    if name == "synthetic":
        return SyntheticSegmentationDataset(synthetic_size, image_size)
    return MSRCSegmentationDataset(root, split, image_size)


def train_segmentation(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_dir = ensure_dir(args.output_dir)
    if args.dataset == "synthetic":
        train_ds = SyntheticSegmentationDataset(args.synthetic_train_size, args.image_size)
        weight_ds = train_ds
        val_ds = SyntheticSegmentationDataset(args.synthetic_val_size, args.image_size)
    else:
        root = Path(args.data_root)
        train_ds = MSRCSegmentationDataset(
            root,
            "train",
            args.image_size,
            augment=args.augment,
            samples_per_image=args.samples_per_image,
        )
        weight_ds = MSRCSegmentationDataset(root, "train", args.image_size)
        val_ds = MSRCSegmentationDataset(root, "val", args.image_size)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    model = UNet(num_classes=NUM_SEG_CLASSES, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    class_weights = None if args.no_class_weights else compute_seg_class_weights(weight_ds).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
    dice_weight = float(args.dice_weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    best_miou = -1.0
    train_loss = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if dice_weight > 0:
                loss = loss + dice_weight * soft_dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), x.size(0))
        scheduler.step()
        if epoch == args.epochs or epoch % max(1, args.eval_every) == 0:
            val_loss, val_miou = evaluate_segmentation_model(model, val_loader, criterion, device, dice_weight)
            if val_miou > best_miou:
                best_miou = val_miou
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss.avg, 6),
                    "val_loss": round(val_loss, 6),
                    "val_miou": round(val_miou, 6),
                    "best_epoch": best_epoch,
                    "best_miou": round(best_miou, 6),
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_miou = evaluate_segmentation_model(model, val_loader, criterion, device, dice_weight)
    row = {
        "task": "segmentation",
        "dataset": args.dataset,
        "model": "unet",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_samples": len(weight_ds),
        "effective_train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "samples_per_image": getattr(args, "samples_per_image", 1),
        "augment": bool(getattr(args, "augment", False)),
        "loss": round(val_loss, 6),
        "miou": round(val_miou, 6),
    }
    checkpoint_path = output_dir / "checkpoints" / "unet_msrc.pt"
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": "unet",
            "dataset": args.dataset,
            "image_size": args.image_size,
            "num_classes": NUM_SEG_CLASSES,
            "base_channels": args.base_channels,
            "dice_weight": args.dice_weight,
            "best_epoch": best_epoch,
            "samples_per_image": getattr(args, "samples_per_image", 1),
            "augment": bool(getattr(args, "augment", False)),
            "class_weights": None if class_weights is None else class_weights.detach().cpu(),
        },
        checkpoint_path,
    )
    demo_path = output_dir / "demo_cases" / "segmentation_demo.png"
    save_segmentation_demo(model, val_loader, device, demo_path)
    row["checkpoint"] = str(checkpoint_path)
    row["demo_image"] = str(demo_path)
    save_csv([row], output_dir / "segmentation_results.csv")
    save_json(row, output_dir / "segmentation_results.json")
    print(row)
    return row


@torch.no_grad()
def save_segmentation_demo(model: nn.Module, loader: DataLoader, device: torch.device, path: Path) -> None:
    ensure_dir(path.parent)
    model.eval()
    best: tuple[float, torch.Tensor, torch.Tensor, np.ndarray] | None = None
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred_batch = logits.argmax(dim=1).cpu().numpy()
        for i in range(x.size(0)):
            target = y[i].cpu().numpy()
            score = segmentation_demo_score(pred_batch[i], target)
            if best is None or score > best[0]:
                best = (score, x[i].detach().cpu(), y[i].detach().cpu(), pred_batch[i])
    if best is None:
        return
    _, x, y, pred = best
    image = ((x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target = y.numpy()
    pred_color = mask_to_color(pred)
    target_color = mask_to_color(target)
    make_triptych(
        [Image.fromarray(image), Image.fromarray(target_color), Image.fromarray(pred_color)],
        ["input", "ground truth", "UNet prediction"],
        path,
    )


def mask_miou(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_SEG_CLASSES) -> float:
    valid = target != IGNORE_INDEX
    scores: list[float] = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        target_c = (target == cls) & valid
        union = np.logical_or(pred_c, target_c).sum()
        if union > 0:
            scores.append(float(np.logical_and(pred_c, target_c).sum() / union))
    return float(np.mean(scores)) if scores else 0.0


def segmentation_demo_score(pred: np.ndarray, target: np.ndarray) -> float:
    valid = target != IGNORE_INDEX
    if not np.any(valid):
        return 0.0
    values, counts = np.unique(target[valid], return_counts=True)
    diversity_bonus = 0.0
    if len(values) >= 2 and counts.max() / counts.sum() < 0.9:
        diversity_bonus = 1.0
    return diversity_bonus + mask_miou(pred, target)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    safe[safe == IGNORE_INDEX] = NUM_SEG_CLASSES
    palette = np.vstack([MSRC_COLORS[1:], np.array([[0, 0, 0]], dtype=np.uint8)])
    safe = np.clip(safe, 0, len(palette) - 1)
    return palette[safe]


class BSDS500SRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        scale: int = 2,
        image_size: int = 96,
        patches_per_image: int = 1,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.scale = scale
        self.image_size = image_size
        self.patches_per_image = max(1, int(patches_per_image))
        self.augment = augment
        candidates = [
            self.root / split,
            self.root / "images" / split,
            self.root / "data" / "images" / split,
            self.root / "BSDS500" / "data" / "images" / split,
            self.root / "BSR" / "BSDS500" / "data" / "images" / split,
            self.root / ("trainval" if split == "train" else split),
        ]
        if split == "train":
            candidates.extend(
                [
                    self.root / "val",
                    self.root / "images" / "val",
                    self.root / "data" / "images" / "val",
                    self.root / "BSDS500" / "data" / "images" / "val",
                    self.root / "BSR" / "BSDS500" / "data" / "images" / "val",
                ]
            )
        files: list[Path] = []
        for folder in candidates:
            files.extend(sorted(folder.glob("*.jpg")))
            files.extend(sorted(folder.glob("*.png")))
        if not files:
            raise FileNotFoundError(f"No BSDS500 images found under {self.root} for split '{split}'.")
        self.files = files

    def __len__(self) -> int:
        return len(self.files) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_index = index % len(self.files)
        image = imread_rgb(self.files[file_index])
        if image is None:
            raise FileNotFoundError(self.files[file_index])
        h, w = image.shape[:2]
        crop = min(h, w, self.image_size)
        crop -= crop % self.scale
        if self.augment or self.patches_per_image > 1:
            rng = np.random.default_rng(index)
            top = int(rng.integers(0, max(1, h - crop + 1)))
            left = int(rng.integers(0, max(1, w - crop + 1)))
        else:
            top = max(0, (h - crop) // 2)
            left = max(0, (w - crop) // 2)
        hr = image[top : top + crop, left : left + crop]
        if self.augment:
            rng = np.random.default_rng(index + 10_000_003)
            if bool(rng.integers(0, 2)):
                hr = np.ascontiguousarray(hr[:, ::-1])
            if bool(rng.integers(0, 2)):
                hr = np.ascontiguousarray(hr[::-1, :])
            rotations = int(rng.integers(0, 4))
            if rotations:
                hr = np.ascontiguousarray(np.rot90(hr, rotations))
        return make_sr_pair(hr, self.scale)


class SyntheticSRDataset(Dataset):
    def __init__(self, size: int, image_size: int = 48, scale: int = 2) -> None:
        self.size = size
        self.image_size = image_size
        self.scale = scale

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(index)
        h = w = self.image_size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        hr = np.stack(
            [
                (np.sin(x / 5.0) + 1) / 2,
                (np.cos(y / 7.0) + 1) / 2,
                (np.sin((x + y) / 9.0) + 1) / 2,
            ],
            axis=-1,
        )
        for _ in range(3):
            x0 = int(rng.integers(0, w - 8))
            y0 = int(rng.integers(0, h - 8))
            hr[y0 : y0 + 8, x0 : x0 + 8] = rng.random(3)
        return make_sr_pair((hr * 255).astype(np.uint8), self.scale)


def make_sr_pair(hr_uint8: np.ndarray, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = hr_uint8.shape[:2]
    h -= h % scale
    w -= w % scale
    hr_uint8 = hr_uint8[:h, :w]
    lr = cv2.resize(hr_uint8, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    bicubic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(bicubic.transpose(2, 0, 1)).float() / 255.0
    y = torch.from_numpy(hr_uint8.transpose(2, 0, 1)).float() / 255.0
    return x, y


class SRCNN(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.ndim == 4:
        mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
        scores = torch.where(
            mse <= 1e-12,
            torch.full_like(mse, 99.0),
            20.0 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-12))),
        )
        return float(scores.mean().item())
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    scores = []
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    for p, t in zip(pred_np, target_np):
        min_dim = min(p.shape[:2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        scores.append(structural_similarity(p, t, channel_axis=-1, data_range=1.0, win_size=max(3, win_size)))
    return float(np.mean(scores))


def build_sr_dataset(name: str, root: Path, split: str, image_size: int, scale: int, synthetic_size: int) -> Dataset:
    if name == "synthetic":
        return SyntheticSRDataset(synthetic_size, image_size=image_size, scale=scale)
    return BSDS500SRDataset(root, split, scale=scale, image_size=image_size)


def train_super_resolution(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_dir = ensure_dir(args.output_dir)
    if args.dataset == "synthetic":
        train_ds = SyntheticSRDataset(args.synthetic_train_size, image_size=args.image_size, scale=args.scale)
        test_ds = SyntheticSRDataset(args.synthetic_test_size, image_size=args.image_size, scale=args.scale)
        train_source_samples = len(train_ds)
    else:
        train_ds = BSDS500SRDataset(
            Path(args.data_root),
            "train",
            scale=args.scale,
            image_size=args.image_size,
            patches_per_image=args.patches_per_image,
            augment=args.augment,
        )
        test_ds = BSDS500SRDataset(Path(args.data_root), "test", scale=args.scale, image_size=args.image_size)
        train_source_samples = len(train_ds.files)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    model = SRCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), x.size(0))
        scheduler.step()
        if epoch == args.epochs or epoch % max(1, args.log_every) == 0:
            print({"epoch": epoch, "train_mse": round(train_loss.avg, 8)})

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    base_psnr_meter = AverageMeter()
    base_ssim_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).clamp(0, 1)
            loss_meter.update(criterion(pred, y).item(), x.size(0))
            psnr_meter.update(psnr(pred, y), x.size(0))
            ssim_meter.update(batch_ssim(pred, y), x.size(0))
            base_psnr_meter.update(psnr(x, y), x.size(0))
            base_ssim_meter.update(batch_ssim(x, y), x.size(0))
    row = {
        "task": "super_resolution",
        "dataset": args.dataset,
        "model": "srcnn",
        "scale": args.scale,
        "epochs": args.epochs,
        "train_samples": train_source_samples,
        "effective_train_patches": len(train_ds),
        "patches_per_image": getattr(args, "patches_per_image", 1),
        "augment": bool(getattr(args, "augment", False)),
        "test_samples": len(test_ds),
        "mse": round(loss_meter.avg, 6),
        "psnr": round(psnr_meter.avg, 6),
        "ssim": round(ssim_meter.avg, 6),
        "bicubic_psnr": round(base_psnr_meter.avg, 6),
        "bicubic_ssim": round(base_ssim_meter.avg, 6),
    }
    checkpoint_path = output_dir / "checkpoints" / "srcnn_bsds500.pt"
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": "srcnn",
            "dataset": args.dataset,
            "scale": args.scale,
            "image_size": args.image_size,
            "patches_per_image": getattr(args, "patches_per_image", 1),
            "augment": bool(getattr(args, "augment", False)),
            "residual": True,
        },
        checkpoint_path,
    )
    demo_path = output_dir / "demo_cases" / "super_resolution_demo.png"
    save_sr_demo(model, test_loader, device, demo_path)
    row["checkpoint"] = str(checkpoint_path)
    row["demo_image"] = str(demo_path)
    save_csv([row], output_dir / "super_resolution_results.csv")
    save_json(row, output_dir / "super_resolution_results.json")
    print(row)
    return row


@torch.no_grad()
def save_sr_demo(model: nn.Module, loader: DataLoader, device: torch.device, path: Path) -> None:
    ensure_dir(path.parent)
    model.eval()
    best: tuple[float, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for x, y in loader:
        x = x.to(device)
        pred_batch = model(x).detach().cpu().clamp(0, 1)
        x_cpu = x.detach().cpu().clamp(0, 1)
        y_cpu = y.detach().cpu().clamp(0, 1)
        for i in range(x_cpu.size(0)):
            pred = pred_batch[i]
            inp = x_cpu[i]
            target = y_cpu[i]
            psnr_gain = psnr(pred, target) - psnr(inp, target)
            ssim_gain = batch_ssim(pred.unsqueeze(0), target.unsqueeze(0)) - batch_ssim(inp.unsqueeze(0), target.unsqueeze(0))
            score = psnr_gain + 20.0 * ssim_gain
            if best is None or score > best[0]:
                best = (score, inp, pred, target)
    if best is None:
        return
    _, inp, pred, target = best
    images = [tensor_to_image(inp), tensor_to_image(pred), tensor_to_image(target)]
    make_triptych(images, ["bicubic input", "SRCNN output", "ground truth"], path)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def make_triptych(images: list[Image.Image], titles: list[str], path: Path) -> None:
    width = 220
    height = 250
    canvas = Image.new("RGB", (width * len(images), height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, title) in enumerate(zip(images, titles)):
        img = image.convert("RGB")
        max_w = width - 20
        max_h = height - 50
        scale = min(max_w / img.width, max_h / img.height)
        if scale != 1:
            resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), resample)
        x0 = i * width + (width - img.width) // 2
        canvas.paste(img, (x0, 34))
        draw.text((i * width + 10, 10), title, fill=(0, 0, 0))
    canvas.save(path)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", default="outputs/exp2")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2: semantic segmentation and super-resolution.")
    sub = parser.add_subparsers(dest="task", required=True)
    seg = sub.add_parser("segmentation")
    add_common(seg)
    seg.add_argument("--base-channels", type=int, default=16)
    seg.add_argument("--dice-weight", type=float, default=0.5)
    seg.add_argument("--no-class-weights", action="store_true")
    seg.add_argument("--augment", action="store_true")
    seg.add_argument("--samples-per-image", type=int, default=1)
    seg.add_argument("--eval-every", type=int, default=1)
    seg.add_argument("--synthetic-train-size", type=int, default=32)
    seg.add_argument("--synthetic-val-size", type=int, default=12)

    sr = sub.add_parser("super-resolution")
    add_common(sr)
    sr.add_argument("--scale", type=int, default=2)
    sr.add_argument("--patches-per-image", type=int, default=1)
    sr.add_argument("--augment", action="store_true")
    sr.add_argument("--log-every", type=int, default=10)
    sr.add_argument("--synthetic-train-size", type=int, default=32)
    sr.add_argument("--synthetic-test-size", type=int, default=12)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.task == "segmentation":
        train_segmentation(args)
    elif args.task == "super-resolution":
        train_super_resolution(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
