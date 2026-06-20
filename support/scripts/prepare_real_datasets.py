from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

URLS = {
    "jena": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    "shakespeare": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    "bsds500": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    "msrc2": "http://download.microsoft.com/download/3/3/9/339D8A24-47D7-412F-A1E8-1A415BC48A15/msrc_objcategimagedatabase_v2.zip",
    "texton_splits": "http://jamie.shotton.org/work/data/TextonBoostSplits.zip",
}

HWDB_GITHUB = "https://github.com/Eyelessdude/handwritten-chinese-characters-dataset.git"


def download(url: str, path: Path, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"[skip] {path} already exists")
        return path
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[ok] {path} ({path.stat().st_size} bytes)")
    return path


def extract_zip(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)


def extract_tgz(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(dest)


def prepare_jena(force: bool = False) -> None:
    archive = DATASETS / "archives" / "jena_climate_2009_2016.csv.zip"
    csv_path = DATASETS / "jena_climate_2009_2016.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {csv_path}")
        return
    download(URLS["jena"], archive, force=force)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATASETS)
    print(f"[ok] {csv_path}")


def prepare_shakespeare(force: bool = False) -> None:
    download(URLS["shakespeare"], DATASETS / "shakespeare.txt", force=force)


def prepare_bsds500(force: bool = False) -> None:
    archive = DATASETS / "archives" / "BSR_bsds500.tgz"
    root = DATASETS / "bsds500"
    marker = root / "BSR" / "BSDS500" / "data" / "images" / "train"
    if marker.exists() and not force:
        print(f"[skip] {root}")
        return
    download(URLS["bsds500"], archive, force=force)
    extract_tgz(archive, root)
    print(f"[ok] {root}")


def find_first_dir(root: Path, names: list[str]) -> Path | None:
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    targets = {n.lower() for n in names}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in targets:
            return path
    return None


def flatten_msrc_layout(root: Path) -> None:
    image_dir = find_first_dir(root, ["images", "Images"])
    gt_dir = find_first_dir(root, ["gt", "GT", "groundtruth", "GroundTruth", "labels", "Labels"])
    target_images = root / "images"
    target_gt = root / "gt"
    if image_dir and image_dir != target_images:
        target_images.mkdir(parents=True, exist_ok=True)
        for src in image_dir.glob("*.bmp"):
            dst = target_images / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    if gt_dir and gt_dir != target_gt:
        target_gt.mkdir(parents=True, exist_ok=True)
        for src in gt_dir.glob("*.bmp"):
            dst = target_gt / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def prepare_msrc2(force: bool = False) -> None:
    archive = DATASETS / "archives" / "msrc_objcategimagedatabase_v2.zip"
    root = DATASETS / "msrc2_seg"
    if (root / "images").exists() and (root / "gt").exists() and not force:
        print(f"[skip] {root}")
    else:
        download(URLS["msrc2"], archive, force=force)
        extract_zip(archive, root)
        flatten_msrc_layout(root)
        print(f"[ok] {root}")
    write_msrc_splits(root)


def write_msrc_splits(root: Path, seed: int = 42) -> None:
    image_dir = find_first_dir(root, ["images", "Images"])
    if image_dir is None:
        raise FileNotFoundError(f"MSRC images directory not found under {root}.")
    files = sorted(p.name for p in image_dir.glob("*.bmp"))
    if len(files) < 591:
        print(f"[warn] MSRC image count is {len(files)}, expected around 591.")
    rng = random.Random(seed)
    rng.shuffle(files)
    train = files[:276]
    val = files[276:335]
    test = files[335:591] if len(files) >= 591 else files[335:]
    for name, split in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
        (root / name).write_text("\n".join(split) + "\n", encoding="utf-8")
    print(f"[ok] MSRC splits: train={len(train)} val={len(val)} test={len(test)}")


def hwdb_status() -> dict:
    root = DATASETS / "HWDB1"
    train_txt = root / "train.txt"
    test_txt = root / "test.txt"
    train = len(train_txt.read_text(encoding="utf-8").splitlines()) if train_txt.exists() else 0
    test = len(test_txt.read_text(encoding="utf-8").splitlines()) if test_txt.exists() else 0
    return {
        "exists": root.exists() and train_txt.exists() and test_txt.exists(),
        "root": str(root),
        "train_lines": train,
        "test_lines": test,
        "expected_train": 2382,
        "expected_test": 601,
    }


def prepare_hwdb1_from_github(force: bool = False, seed: int = 42) -> None:
    """Create the 10-class HWDB1 subset expected by the PPT.

    The source repository contains PNG images converted from the Kaggle CASIA-HWDB
    image dataset. We select the 10 most populated labels and materialize exactly
    2382 train images and 601 test images, matching the slide counts.
    """

    repo = DATASETS / "archives" / "hwdb_github"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", HWDB_GITHUB, str(repo)], check=True)
    train_src = repo / "CASIA-HWDB_Train" / "Train"
    test_src = repo / "CASIA-HWDB_Test" / "Test"
    if not train_src.exists() or not test_src.exists():
        raise FileNotFoundError(f"Expected CASIA-HWDB_Train/Train and CASIA-HWDB_Test/Test under {repo}.")

    root = DATASETS / "HWDB1"
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    class_counts: list[tuple[int, str, int]] = []
    for folder in train_src.iterdir():
        if not folder.is_dir():
            continue
        test_folder = test_src / folder.name
        if not test_folder.exists():
            continue
        class_counts.append((len(list(folder.glob("*.png"))), folder.name, len(list(test_folder.glob("*.png")))))
    selected = [name for _, name, _ in sorted(class_counts, reverse=True)[:10]]

    train_targets = [239, 239] + [238] * 8
    test_targets = [61] + [60] * 9
    train_lines: list[str] = []
    test_lines: list[str] = []
    labels: dict[str, str] = {}
    for label, folder_name in enumerate(selected):
        labels[str(label)] = folder_name
        for split_name, source_base, targets, lines in [
            ("train", train_src, train_targets, train_lines),
            ("test", test_src, test_targets, test_lines),
        ]:
            source_folder = source_base / folder_name
            images = sorted(source_folder.glob("*.png"))
            rng.shuffle(images)
            need = targets[label]
            if len(images) < need:
                raise RuntimeError(f"Not enough images for class {folder_name}: need {need}, got {len(images)}.")
            target_folder = root / "images" / split_name / str(label)
            target_folder.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(images[:need]):
                dst = target_folder / f"{label}_{i:04d}.png"
                if not dst.exists():
                    shutil.copy2(src, dst)
                lines.append(f"{dst.relative_to(root).as_posix()} {label}")
    (root / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (root / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    (root / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] HWDB1 subset created at {root}: train={len(train_lines)} test={len(test_lines)} classes=10")


def status() -> dict:
    checks = {
        "mnist": (DATASETS / "MNIST").exists(),
        "fashion_mnist": (DATASETS / "FashionMNIST").exists(),
        "hwdb1": hwdb_status(),
        "msrc2": {
            "images": len(list((DATASETS / "msrc2_seg" / "images").glob("*.bmp"))),
            "gt": len(list((DATASETS / "msrc2_seg" / "gt").glob("*.bmp"))),
            "train_txt": (DATASETS / "msrc2_seg" / "train.txt").exists(),
            "val_txt": (DATASETS / "msrc2_seg" / "val.txt").exists(),
            "test_txt": (DATASETS / "msrc2_seg" / "test.txt").exists(),
        },
        "bsds500": {
            "train": len(list((DATASETS / "bsds500").glob("**/images/train/*.jpg"))),
            "val": len(list((DATASETS / "bsds500").glob("**/images/val/*.jpg"))),
            "test": len(list((DATASETS / "bsds500").glob("**/images/test/*.jpg"))),
        },
        "jena": (DATASETS / "jena_climate_2009_2016.csv").exists(),
        "shakespeare": (DATASETS / "shakespeare.txt").exists(),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or verify the real datasets used by the lab project.")
    parser.add_argument("--all-public", action="store_true", help="Download all public datasets except HWDB1.")
    parser.add_argument("--jena", action="store_true")
    parser.add_argument("--shakespeare", action="store_true")
    parser.add_argument("--bsds500", action="store_true")
    parser.add_argument("--msrc2", action="store_true")
    parser.add_argument("--hwdb1", action="store_true", help="Prepare a 10-class HWDB1-compatible subset from a public CASIA-HWDB PNG mirror.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    DATASETS.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return
    if args.all_public or args.jena:
        prepare_jena(args.force)
    if args.all_public or args.shakespeare:
        prepare_shakespeare(args.force)
    if args.all_public or args.bsds500:
        prepare_bsds500(args.force)
    if args.all_public or args.msrc2:
        prepare_msrc2(args.force)
    if args.all_public or args.hwdb1:
        prepare_hwdb1_from_github(args.force)
    print(json.dumps(status(), ensure_ascii=False, indent=2))
    print("[note] HWDB1 is prepared from a public CASIA-HWDB PNG mirror when --hwdb1 or --all-public is used.")


if __name__ == "__main__":
    main()
