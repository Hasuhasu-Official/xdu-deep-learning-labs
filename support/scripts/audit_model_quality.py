from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment1_image_classification.classification import build_dataset, build_model, evaluate  # noqa: E402
from experiment2_segmentation_super_resolution.segmentation_sr import (  # noqa: E402
    BSDS500SRDataset,
    MSRCSegmentationDataset,
    SRCNN,
    SegmentationIoUMeter,
    UNet,
    batch_ssim,
    psnr,
)
from experiment3_recurrent_neural_networks.sequence import (  # noqa: E402
    CharRNN,
    SequenceRegressor,
    WindowedArrayDataset,
    generate_text,
    preprocess_jena,
)


OUT = ROOT / "outputs" / "full_real"
AUDIT = ROOT / "outputs" / "model_audit"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audit_classification(dev: torch.device) -> list[dict]:
    rows = []
    result_rows = read_csv(OUT / "exp1" / "classification_results.csv")
    for row in result_rows:
        if row.get("phase") != "model_compare":
            continue
        dataset = row["dataset"]
        model_name = row["model"]
        ckpt_path = resolve(row["checkpoint"])
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = build_model(model_name, int(ckpt["in_channels"]), int(ckpt["image_size"]), int(ckpt["num_classes"])).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        _, _, num_classes = build_dataset(dataset, ROOT / "datasets", False, int(ckpt["image_size"]), False, 128)
        test_ds, _, _ = build_dataset(dataset, ROOT / "datasets", False, int(ckpt["image_size"]), False, 128)
        loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        _, acc = evaluate(model, loader, torch.nn.CrossEntropyLoss(), dev)
        if dataset == "mnist":
            threshold = 0.97
        elif dataset == "fashion_mnist":
            threshold = 0.84
        elif dataset == "hwdb":
            threshold = 0.75 if model_name != "mlp" else 0.55
        else:
            threshold = 0.0
        rows.append(
            {
                "task": "classification",
                "dataset": dataset,
                "model": model_name,
                "classes": num_classes,
                "checkpoint": str(ckpt_path.relative_to(ROOT)),
                "stored_acc": row.get("test_acc", ""),
                "recomputed_acc": round(float(acc), 6),
                "threshold": threshold,
                "status": "PASS" if acc >= threshold else "WARN",
            }
        )
    return rows


@torch.no_grad()
def audit_segmentation(dev: torch.device) -> list[dict]:
    result = read_csv(OUT / "exp2_seg" / "segmentation_results.csv")[0]
    ckpt_path = resolve(result["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = UNet(num_classes=int(ckpt["num_classes"]), base=int(ckpt["base_channels"])).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = MSRCSegmentationDataset(ROOT / "datasets" / "msrc2_seg", "val", int(ckpt["image_size"]))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    meter = SegmentationIoUMeter(int(ckpt["num_classes"]))
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        meter.update(model(x), y)
    miou = meter.miou()
    return [
        {
            "task": "segmentation",
            "dataset": "msrc",
            "model": "unet",
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "stored_miou": result.get("miou", ""),
            "recomputed_miou": round(miou, 6),
            "threshold": 0.45,
            "status": "PASS" if miou >= 0.45 else "WARN",
        }
    ]


@torch.no_grad()
def audit_super_resolution(dev: torch.device) -> list[dict]:
    result = read_csv(OUT / "exp2_sr" / "super_resolution_results.csv")[0]
    ckpt_path = resolve(result["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = SRCNN().to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = BSDS500SRDataset(ROOT / "datasets" / "bsds500", "test", int(ckpt["scale"]), int(ckpt["image_size"]))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    srcnn_psnr = []
    srcnn_ssim = []
    base_psnr = []
    base_ssim = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).clamp(0, 1)
        srcnn_psnr.append(psnr(pred, y))
        srcnn_ssim.append(batch_ssim(pred, y))
        base_psnr.append(psnr(x, y))
        base_ssim.append(batch_ssim(x, y))
    sr_psnr = float(np.mean(srcnn_psnr))
    bi_psnr = float(np.mean(base_psnr))
    sr_ssim = float(np.mean(srcnn_ssim))
    bi_ssim = float(np.mean(base_ssim))
    psnr_gain = sr_psnr - bi_psnr
    ssim_gain = sr_ssim - bi_ssim
    return [
        {
            "task": "super_resolution",
            "dataset": "bsds500",
            "model": "residual_srcnn",
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "stored_psnr": result.get("psnr", ""),
            "recomputed_psnr": round(sr_psnr, 6),
            "bicubic_psnr": round(bi_psnr, 6),
            "stored_ssim": result.get("ssim", ""),
            "recomputed_ssim": round(sr_ssim, 6),
            "bicubic_ssim": round(bi_ssim, 6),
            "psnr_gain": round(psnr_gain, 6),
            "ssim_gain": round(ssim_gain, 6),
            "threshold": "PSNR +0.30 and SSIM +0.005",
            "status": "PASS" if psnr_gain >= 0.30 and ssim_gain >= 0.005 else "WARN",
        }
    ]


@torch.no_grad()
def audit_weather(dev: torch.device) -> list[dict]:
    rows = []
    result_rows = read_csv(OUT / "exp3_weather" / "weather_results.csv")
    _, _, test, _, target_index = preprocess_jena(ROOT / "datasets" / "jena_climate_2009_2016.csv")
    for result in result_rows:
        name = result["model"]
        ckpt_path = resolve(result["checkpoint"])
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = SequenceRegressor(
            name,
            int(ckpt["input_dim"]),
            int(ckpt["horizon"]),
            int(ckpt["hidden_size"]),
            int(ckpt["num_layers"]),
            float(ckpt["dropout"]),
        ).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ds = WindowedArrayDataset(test, int(ckpt["input_width"]), int(ckpt["horizon"]), target_index)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        mse_sum = 0.0
        mae_sum = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            mse_sum += F.mse_loss(pred, y, reduction="sum").item()
            mae_sum += torch.abs(pred - y).sum().item()
            count += y.numel()
        mse = mse_sum / count
        mae = mae_sum / count
        rows.append(
            {
                "task": "weather",
                "dataset": "jena",
                "model": name,
                "checkpoint": str(ckpt_path.relative_to(ROOT)),
                "stored_mse": result.get("mse", ""),
                "recomputed_mse": round(mse, 6),
                "stored_mae": result.get("mae", ""),
                "recomputed_mae": round(mae, 6),
                "threshold_mse": 0.25,
                "status": "PASS" if mse <= 0.25 else "WARN",
            }
        )
    return rows


@torch.no_grad()
def audit_shakespeare(dev: torch.device) -> list[dict]:
    result_path = OUT / "exp3_shakespeare" / "shakespeare_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ckpt_path = resolve(result["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = CharRNN(len(ckpt["vocab"]), ckpt["model"], int(ckpt["embed_dim"]), int(ckpt["hidden_size"]), int(ckpt["num_layers"])).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = type("Vocab", (), {})()
    vocab.chars = ckpt["vocab"]
    vocab.stoi = {ch: i for i, ch in enumerate(vocab.chars)}
    vocab.itos = {i: ch for ch, i in vocab.stoi.items()}
    generated = generate_text(model, vocab, "ROMEO:", 400, 0.8, dev)
    generated_path = AUDIT / "shakespeare_audit_sample.txt"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    unique_chars = len(set(generated))
    return [
        {
            "task": "text_generation",
            "dataset": "shakespeare",
            "model": ckpt["model"],
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "train_loss": result.get("train_loss", ""),
            "generated_chars": len(generated),
            "unique_chars": unique_chars,
            "sample": str(generated_path.relative_to(ROOT)),
            "status": "PASS" if len(generated) >= 400 and unique_chars >= 20 else "WARN",
        }
    ]


def write_markdown(all_rows: dict[str, list[dict]]) -> None:
    lines = ["# 模型质量自动审计", ""]
    for name, rows in all_rows.items():
        lines.append(f"## {name}")
        if not rows:
            lines.append("No rows.")
            lines.append("")
            continue
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        lines.append("")
    summary = []
    for rows in all_rows.values():
        summary.extend(row.get("status") for row in rows)
    lines.insert(2, f"总体状态：{'PASS' if all(s == 'PASS' for s in summary) else 'WARN'}")
    lines.insert(3, "")
    (AUDIT / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    AUDIT.mkdir(parents=True, exist_ok=True)
    dev = device()
    all_rows = {
        "classification": audit_classification(dev),
        "segmentation": audit_segmentation(dev),
        "super_resolution": audit_super_resolution(dev),
        "weather": audit_weather(dev),
        "shakespeare": audit_shakespeare(dev),
    }
    for name, rows in all_rows.items():
        write_csv(AUDIT / f"{name}.csv", rows)
    write_markdown(all_rows)
    print((AUDIT / "audit_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
