from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(name: str, args: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.time()
    print(f"\n== {name} ==")
    print(" ".join(args))
    proc = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    elapsed = round(time.time() - start, 2)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")
    return {"name": name, "seconds": elapsed, "command": " ".join(args)}


def main() -> None:
    out = ROOT / "outputs" / "smoke"
    commands = [
        (
            "experiment1_classification",
            [
                PYTHON,
                "-m",
                "experiment1_image_classification.classification",
                "--datasets",
                "synthetic",
                "--models",
                "mlp",
                "lenet",
                "alexnet",
                "googlenet",
                "resnet",
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--batch-sizes",
                "4",
                "8",
                "--image-size",
                "32",
                "--limit-train",
                "24",
                "--limit-test",
                "12",
                "--synthetic-train-size",
                "24",
                "--synthetic-test-size",
                "12",
                "--device",
                "cpu",
                "--output-dir",
                str(out / "exp1"),
            ],
        ),
        (
            "experiment2_segmentation",
            [
                PYTHON,
                "-m",
                "experiment2_segmentation_super_resolution.segmentation_sr",
                "segmentation",
                "--dataset",
                "synthetic",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--image-size",
                "32",
                "--base-channels",
                "4",
                "--synthetic-train-size",
                "8",
                "--synthetic-val-size",
                "4",
                "--device",
                "cpu",
                "--output-dir",
                str(out / "exp2_seg"),
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
                "synthetic",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--image-size",
                "32",
                "--synthetic-train-size",
                "8",
                "--synthetic-test-size",
                "4",
                "--device",
                "cpu",
                "--output-dir",
                str(out / "exp2_sr"),
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
                "synthetic",
                "--models",
                "rnn",
                "gru",
                "lstm",
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--input-width",
                "24",
                "--horizon",
                "12",
                "--hidden-size",
                "16",
                "--synthetic-train-size",
                "32",
                "--synthetic-val-size",
                "16",
                "--device",
                "cpu",
                "--output-dir",
                str(out / "exp3_weather"),
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
                "synthetic",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--seq-len",
                "32",
                "--embed-dim",
                "16",
                "--hidden-size",
                "32",
                "--num-layers",
                "1",
                "--generate-length",
                "120",
                "--device",
                "cpu",
                "--output-dir",
                str(out / "exp3_shakespeare"),
            ],
        ),
    ]
    summary = [run(name, cmd) for name, cmd in commands]
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSmoke tests completed.")


if __name__ == "__main__":
    main()
