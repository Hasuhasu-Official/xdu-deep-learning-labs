from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


PROFILES = {
    "acceptance": {
        "exp1_epochs": "20",
        "seg_epochs": "120",
        "sr_epochs": "120",
        "weather_epochs": "20",
        "shakespeare_epochs": "2",
    },
    "full": {
        "exp1_epochs": "20",
        "seg_epochs": "180",
        "sr_epochs": "160",
        "weather_epochs": "20",
        "shakespeare_epochs": "20",
    },
}


def data_status() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [PYTHON, "scripts/prepare_real_datasets.py", "--status"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def missing_reasons(status: dict) -> list[str]:
    reasons = []
    if not status.get("mnist"):
        reasons.append("MNIST not found; it will be created by torchvision only when experiment 1 runs with --download.")
    if not status.get("fashion_mnist"):
        reasons.append("FashionMNIST not found; it will be created by torchvision only when experiment 1 runs with --download.")
    hwdb = status.get("hwdb1", {})
    if not hwdb.get("exists"):
        reasons.append("HWDB1 is missing. Expected datasets/HWDB1/train.txt and test.txt.")
    elif hwdb.get("train_lines") != hwdb.get("expected_train") or hwdb.get("test_lines") != hwdb.get("expected_test"):
        reasons.append(
            f"HWDB1 line counts are train={hwdb.get('train_lines')} test={hwdb.get('test_lines')}; "
            f"expected {hwdb.get('expected_train')}/{hwdb.get('expected_test')}."
        )
    msrc = status.get("msrc2", {})
    if msrc.get("images", 0) == 0 or msrc.get("gt", 0) == 0 or not (msrc.get("train_txt") and msrc.get("val_txt")):
        reasons.append("MSRC-V2 segmentation data or train/val split files are missing under datasets/msrc2_seg.")
    bsds = status.get("bsds500", {})
    if bsds.get("train", 0) == 0 or bsds.get("val", 0) == 0 or bsds.get("test", 0) == 0:
        reasons.append("BSDS500 images are missing under datasets/bsds500.")
    if not status.get("jena"):
        reasons.append("Jena climate CSV is missing.")
    if not status.get("shakespeare"):
        reasons.append("Shakespeare corpus is missing.")
    return reasons


def run_step(name: str, command: list[str], continue_on_error: bool) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.time()
    print(f"\n== {name} ==")
    print(" ".join(command))
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True)
    item = {
        "name": name,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 2),
    }
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")
    return item


def commands(profile: dict, device: str) -> list[tuple[str, list[str]]]:
    return [
        (
            "experiment1_image_classification",
            [
                PYTHON,
                "-m",
                "experiment1_image_classification.classification",
                "--datasets",
                "mnist",
                "fashion_mnist",
                "hwdb",
                "--models",
                "mlp",
                "lenet",
                "alexnet",
                "googlenet",
                "resnet",
                "--data-root",
                "datasets",
                "--download",
                "--image-size",
                "64",
                "--epochs",
                profile["exp1_epochs"],
                "--batch-size",
                "16",
                "--batch-sizes",
                "16",
                "32",
                "64",
                "128",
                "--device",
                device,
                "--output-dir",
                "outputs/full_real/exp1",
            ],
        ),
        (
            "experiment2_semantic_segmentation",
            [
                PYTHON,
                "-m",
                "experiment2_segmentation_super_resolution.segmentation_sr",
                "segmentation",
                "--dataset",
                "msrc",
                "--data-root",
                "datasets/msrc2_seg",
                "--image-size",
                "128",
                "--epochs",
                profile["seg_epochs"],
                "--batch-size",
                "8",
                "--base-channels",
                "32",
                "--augment",
                "--samples-per-image",
                "4",
                "--eval-every",
                "1",
                "--device",
                device,
                "--output-dir",
                "outputs/full_real/exp2_seg",
            ],
        ),
        (
            "experiment2_super_resolution",
            [
                PYTHON,
                "-m",
                "experiment2_segmentation_super_resolution.segmentation_sr",
                "super-resolution",
                "--dataset",
                "bsds500",
                "--data-root",
                "datasets/bsds500",
                "--scale",
                "4",
                "--image-size",
                "96",
                "--epochs",
                profile["sr_epochs"],
                "--batch-size",
                "16",
                "--patches-per-image",
                "16",
                "--augment",
                "--log-every",
                "10",
                "--device",
                device,
                "--output-dir",
                "outputs/full_real/exp2_sr",
            ],
        ),
        (
            "experiment3_weather",
            [
                PYTHON,
                "-m",
                "experiment3_recurrent_neural_networks.sequence",
                "weather",
                "--dataset",
                "jena",
                "--csv-path",
                "datasets/jena_climate_2009_2016.csv",
                "--models",
                "rnn",
                "gru",
                "lstm",
                "--input-width",
                "168",
                "--horizon",
                "168",
                "--epochs",
                profile["weather_epochs"],
                "--batch-size",
                "64",
                "--device",
                device,
                "--output-dir",
                "outputs/full_real/exp3_weather",
            ],
        ),
        (
            "experiment3_shakespeare",
            [
                PYTHON,
                "-m",
                "experiment3_recurrent_neural_networks.sequence",
                "shakespeare",
                "--dataset",
                "shakespeare",
                "--text-file",
                "datasets/shakespeare.txt",
                "--rnn-type",
                "lstm",
                "--seq-len",
                "100",
                "--epochs",
                profile["shakespeare_epochs"],
                "--batch-size",
                "64",
                "--device",
                device,
                "--seed-text",
                "ROMEO:",
                "--generate-length",
                "1000",
                "--output-dir",
                "outputs/full_real/exp3_shakespeare",
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three labs on the real full datasets.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-missing", action="store_true", help="Start anyway. Missing datasets will fail at the corresponding step.")
    args = parser.parse_args()

    status = data_status()
    reasons = missing_reasons(status)
    manifest_dir = ROOT / "outputs" / "full_real"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "dataset_status_before.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if reasons and not args.allow_missing:
        print("Real dataset check failed:")
        for reason in reasons:
            print(f"- {reason}")
        print("\nRun scripts/prepare_real_datasets.py --all-public to fetch public datasets; provide HWDB1 manually.")
        raise SystemExit(2)

    summary = []
    for name, command in commands(PROFILES[args.profile], args.device):
        summary.append(run_step(name, command, args.continue_on_error))
    (manifest_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll requested real experiments finished. Summary: {manifest_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
