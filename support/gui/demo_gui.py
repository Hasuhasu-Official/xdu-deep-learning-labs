from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from experiment1_image_classification.classification import build_dataset, build_model, label_names
from experiment2_segmentation_super_resolution.segmentation_sr import (
    BSDS500SRDataset,
    MSRCSegmentationDataset,
    SRCNN,
    UNet,
    batch_ssim,
    make_sr_pair,
    make_triptych,
    mask_to_color,
    psnr,
    tensor_to_image,
)
from experiment3_recurrent_neural_networks.sequence import (
    CharRNN,
    SequenceRegressor,
    WindowedArrayDataset,
    generate_text,
    preprocess_jena,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "full_real"
TMP = ROOT / "outputs" / "demo_runtime"
CLASSIFICATION_MODELS = ["mlp", "lenet", "alexnet", "googlenet", "resnet"]
MSRC_CLASS_NAMES = [
    "建筑",
    "草地",
    "树木",
    "牛",
    "羊",
    "天空",
    "飞机",
    "水面",
    "人脸",
    "汽车",
    "自行车",
    "花",
    "标识",
    "鸟",
    "书",
    "椅子",
    "道路",
    "猫",
    "狗",
    "人体",
    "船",
]


def set_pixmap(label: QLabel, path: Path, max_width: int = 760, max_height: int = 420) -> None:
    pixmap = QPixmap(str(path))
    label.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def tensor_gray_to_png(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    if x.ndim == 3 and x.size(0) == 1:
        img = Image.fromarray((x[0].numpy() * 255).astype(np.uint8), mode="L").convert("RGB")
    elif x.ndim == 3:
        img = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray((x.numpy() * 255).astype(np.uint8), mode="L").convert("RGB")
    img.save(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_box(title: str, layout) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(layout)
    return box


def square_padded(image: Image.Image, mode: str, fill: int | tuple[int, int, int]) -> Image.Image:
    image = image.convert(mode)
    side = max(image.width, image.height)
    canvas = Image.new(mode, (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def classification_tensor_from_pil(image: Image.Image, image_size: int, invert: bool) -> torch.Tensor:
    image = square_padded(image, "L", 255)
    if invert:
        image = ImageOps.invert(image)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def pil_from_classification_tensor(x: torch.Tensor) -> Image.Image:
    x = (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    if x.ndim == 3 and x.size(0) == 1:
        arr = (x[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if x.ndim == 3:
        arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB").convert("L")
    arr = (x.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def seg_tensor_from_pil(image: Image.Image, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    preview = square_padded(image, "RGB", (0, 0, 0)).resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(preview, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return (tensor - 0.5) / 0.5, preview


def sr_uint8_from_pil(image: Image.Image, image_size: int, scale: int) -> np.ndarray:
    image = square_padded(image, "RGB", (0, 0, 0))
    size = min(image_size, image.width, image.height)
    size = max(scale, size - size % scale)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    if size != image_size:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    return arr[: h - h % scale, : w - w % scale]


def segmentation_area_summary(mask: np.ndarray, limit: int = 5) -> str:
    valid = mask[(mask >= 0) & (mask < len(MSRC_CLASS_NAMES))]
    if valid.size == 0:
        return "未识别到有效区域"
    values, counts = np.unique(valid, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    parts = []
    total = float(valid.size)
    for i in order:
        cls = int(values[i])
        parts.append(f"{MSRC_CLASS_NAMES[cls]} {counts[i] / total * 100:.1f}%")
    return "，".join(parts)


class DrawingCanvas(QWidget):
    def __init__(self, size: int = 280) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.image = QImage(size, size, QImage.Format_RGB32)
        self.last_pos = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setAttribute(Qt.WA_StaticContents)
        self.setStyleSheet("border: 1px solid #8a8f98; background: white;")
        self.clear_canvas()

    def clear_canvas(self) -> None:
        self.image.fill(Qt.white)
        self.update()

    def to_pil(self) -> Image.Image:
        TMP.mkdir(parents=True, exist_ok=True)
        path = TMP / "drawing_canvas.png"
        self.image.save(str(path))
        return Image.open(path).convert("RGB")

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(Qt.gray, 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.last_pos = self._event_pos(event)
            self._draw_point(self.last_pos)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton and self.last_pos is not None:
            current = self._event_pos(event)
            self._draw_line(self.last_pos, current)
            self.last_pos = current
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            current = self._event_pos(event)
            if self.last_pos is not None:
                self._draw_line(self.last_pos, current)
            self.last_pos = None
            self.update()

    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _draw_point(self, point) -> None:
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 18, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(point)
        painter.end()
        self.update()

    def _draw_line(self, start, end) -> None:
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 18, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        painter.end()


class ClassificationDemo(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model_cache: dict[tuple[str, str], tuple[torch.nn.Module, dict]] = {}
        self.dataset_box = QComboBox()
        self.dataset_box.addItems(["mnist", "fashion_mnist", "hwdb"])
        self.dataset_box.currentTextChanged.connect(self.sync_invert_default)
        self.invert_box = QCheckBox("输入反色")
        self.invert_box.setChecked(True)
        self.image = QLabel(alignment=Qt.AlignCenter)
        self.image.setMinimumHeight(300)
        self.canvas = DrawingCanvas()
        self.compare_table = QTableWidget(0, 6)
        self.compare_table.setHorizontalHeaderLabels(["模型", "预测类别", "置信度", "Top-3", "测试准确率", "参数量"])
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result = QTextEdit(readOnly=True)
        self.result.setMaximumHeight(135)

        sample_btn = QPushButton("真实样例五模型识别")
        sample_btn.clicked.connect(self.run_case)
        sample_btn.setToolTip("从所选数据集的测试集抽取一张真实样本，并同时运行五个分类模型。")
        upload_btn = QPushButton("上传图片五模型识别")
        upload_btn.clicked.connect(self.upload_image)
        upload_btn.setToolTip("上传一张本地图片，预处理后同时运行五个分类模型。")
        draw_btn = QPushButton("画板五模型识别")
        draw_btn.clicked.connect(self.run_drawing)
        draw_btn.setToolTip("将右侧画板内容送入五个分类模型。")
        clear_btn = QPushButton("清空画板")
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        clear_btn.setToolTip("清除当前手写输入。")

        top = QHBoxLayout()
        top.addWidget(QLabel("数据集"))
        top.addWidget(self.dataset_box)
        top.addWidget(self.invert_box)
        top.addWidget(sample_btn)
        top.addWidget(upload_btn)

        middle = QHBoxLayout()
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.image, stretch=1)
        middle.addWidget(group_box("预处理预览", preview_layout), stretch=1)
        draw_col = QVBoxLayout()
        draw_col.addWidget(self.canvas, alignment=Qt.AlignCenter)
        draw_row = QHBoxLayout()
        draw_row.addWidget(draw_btn)
        draw_row.addWidget(clear_btn)
        draw_col.addLayout(draw_row)
        middle.addWidget(group_box("手写输入", draw_col))

        layout = QVBoxLayout(self)
        layout.addWidget(group_box("输入来源", top))
        layout.addLayout(middle, stretch=1)
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.compare_table)
        layout.addWidget(group_box("五模型对比结果", result_layout))
        detail_layout = QVBoxLayout()
        detail_layout.addWidget(self.result)
        layout.addWidget(group_box("运行记录", detail_layout))

    def sync_invert_default(self) -> None:
        self.invert_box.setChecked(self.dataset_box.currentText() in {"mnist", "fashion_mnist"})

    def checkpoint_path(self, dataset_name: str, model_name: str) -> Path:
        rows = read_rows(OUT / "exp1" / "classification_results.csv")
        row = next(
            (
                r
                for r in rows
                if r.get("dataset") == dataset_name and r.get("model") == model_name and r.get("phase") == "model_compare"
            ),
            None,
        )
        if row and row.get("checkpoint"):
            path = Path(row["checkpoint"])
            return path if path.is_absolute() else ROOT / path
        return OUT / "exp1" / "checkpoints" / f"{dataset_name}_{model_name}_model_compare_bs64.pt"

    def load_model(self, dataset_name: str, model_name: str) -> tuple[torch.nn.Module, dict] | None:
        key = (dataset_name, model_name)
        if key in self.model_cache:
            return self.model_cache[key]
        ckpt_path = self.checkpoint_path(dataset_name, model_name)
        if not ckpt_path.exists():
            self.result.setPlainText(f"缺少模型文件: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(model_name, int(ckpt["in_channels"]), int(ckpt["image_size"]), int(ckpt["num_classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model_cache[key] = (model, ckpt)
        return model, ckpt

    def classify_image(self, source: str, image: Image.Image, true_label: int | None = None, invert: bool = False) -> None:
        dataset_name = self.dataset_box.currentText()
        labels = label_names(dataset_name)
        rows = read_rows(OUT / "exp1" / "classification_results.csv")
        table_rows: list[list[str]] = []
        preview_tensor: torch.Tensor | None = None
        for model_name in CLASSIFICATION_MODELS:
            loaded = self.load_model(dataset_name, model_name)
            if loaded is None:
                table_rows.append([model_name, "缺少checkpoint", "-", "-", "-", "-"])
                continue
            model, ckpt = loaded
            x = classification_tensor_from_pil(image, int(ckpt["image_size"]), invert)
            if preview_tensor is None:
                preview_tensor = x
            with torch.no_grad():
                probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]
            pred = int(probs.argmax())
            top_values, top_indices = torch.topk(probs, k=min(3, probs.numel()))
            top_lines = []
            for value, index in zip(top_values.tolist(), top_indices.tolist()):
                name = labels[index] if index < len(labels) else str(index)
                top_lines.append(f"{name}:{value:.3f}")
            metric = next(
                (
                    r
                    for r in rows
                    if r.get("dataset") == dataset_name and r.get("model") == model_name and r.get("phase") == "model_compare"
                ),
                {},
            )
            pred_text = labels[pred] if pred < len(labels) else str(pred)
            table_rows.append(
                [
                    model_name,
                    pred_text,
                    f"{float(probs[pred]) * 100:.2f}%",
                    " | ".join(top_lines),
                    metric.get("test_acc", "n/a"),
                    metric.get("params", "n/a"),
                ]
            )
        self.update_compare_table(table_rows)
        if preview_tensor is not None:
            img_path = TMP / "classification_live.png"
            tensor_gray_to_png(preview_tensor, img_path)
            set_pixmap(self.image, img_path, 500, 360)
        true_text = "无"
        if true_label is not None:
            true_text = labels[true_label] if true_label < len(labels) else str(true_label)
        self.result.setPlainText(
            f"输入来源: {source}\n"
            f"真实标签: {true_text}\n"
            f"处理方式: 同一张输入图像同时送入 MLP、LeNet、AlexNet、GoogLeNet、ResNet，表格展示各模型预测、置信度和训练验收指标。\n"
            f"checkpoint目录: {OUT / 'exp1' / 'checkpoints'}"
        )

    def update_compare_table(self, rows: list[list[str]]) -> None:
        self.compare_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index in {0, 1, 2}:
                    item.setTextAlignment(Qt.AlignCenter)
                self.compare_table.setItem(row_index, col_index, item)

    def classify_tensor(self, source: str, x: torch.Tensor, true_label: int | None = None) -> None:
        self.classify_image(source, pil_from_classification_tensor(x), true_label=true_label, invert=False)

    def run_case(self) -> None:
        dataset_name = self.dataset_box.currentText()
        loaded = self.load_model(dataset_name, CLASSIFICATION_MODELS[0])
        if loaded is None:
            return
        _, ckpt = loaded
        ds, _, _ = build_dataset(dataset_name, ROOT / "datasets", False, int(ckpt["image_size"]), False, 128)
        idx = random.randrange(len(ds))
        x, y = ds[idx]
        self.classify_tensor(f"真实测试集样本 #{idx}", x, int(y))

    def upload_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", str(ROOT), "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.run_uploaded_image(Path(path))

    def run_uploaded_image(self, path: Path) -> None:
        image = Image.open(path)
        self.classify_image(f"上传图片 {path.name}", image, invert=self.invert_box.isChecked())

    def run_drawing(self) -> None:
        self.classify_image("手写画板", self.canvas.to_pil(), invert=self.invert_box.isChecked())


class VisionDemo(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.seg_model_cache: tuple[torch.nn.Module, dict] | None = None
        self.sr_model_cache: tuple[torch.nn.Module, dict] | None = None
        self.seg_image = QLabel(alignment=Qt.AlignCenter)
        self.seg_image.setMinimumHeight(250)
        self.sr_image = QLabel(alignment=Qt.AlignCenter)
        self.sr_image.setMinimumHeight(250)
        self.info = QTextEdit(readOnly=True)
        self.info.setMaximumHeight(130)

        seg_btn = QPushButton("MSRC真实样例分割")
        seg_btn.clicked.connect(self.run_segmentation)
        seg_btn.setToolTip("从 MSRC-V2 验证集抽取一张真实图像，输出 UNet 分割结果和区域占比。")
        seg_upload_btn = QPushButton("上传图片分割")
        seg_upload_btn.clicked.connect(self.upload_segmentation)
        seg_upload_btn.setToolTip("上传本地图像，输出 UNet 分割预测。")
        sr_btn = QPushButton("BSDS真实样例超分")
        sr_btn.clicked.connect(self.run_sr)
        sr_btn.setToolTip("从 BSDS500 测试集抽取一张真实图像，对比 bicubic 与 SRCNN。")
        sr_upload_btn = QPushButton("上传图片超分")
        sr_upload_btn.clicked.connect(self.upload_sr)
        sr_upload_btn.setToolTip("上传本地图像，模拟低分辨率输入并运行 SRCNN 超分。")

        row = QHBoxLayout()
        row.addWidget(seg_btn)
        row.addWidget(seg_upload_btn)
        row.addWidget(sr_btn)
        row.addWidget(sr_upload_btn)

        seg_layout = QVBoxLayout()
        seg_layout.addWidget(self.seg_image, stretch=1)
        seg_tab = QWidget()
        seg_tab.setLayout(seg_layout)

        sr_layout = QVBoxLayout()
        sr_layout.addWidget(self.sr_image, stretch=1)
        sr_tab = QWidget()
        sr_tab.setLayout(sr_layout)

        visual_tabs = QTabWidget()
        visual_tabs.addTab(seg_tab, "语义分割：场景理解")
        visual_tabs.addTab(sr_tab, "图像超分：清晰化")

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.info)

        layout = QVBoxLayout(self)
        layout.addWidget(group_box("应用入口", row))
        layout.addWidget(visual_tabs, stretch=1)
        layout.addWidget(group_box("运行记录与指标", info_layout))

    def load_seg_model(self) -> tuple[torch.nn.Module, dict] | None:
        if self.seg_model_cache is not None:
            return self.seg_model_cache
        ckpt_path = OUT / "exp2_seg" / "checkpoints" / "unet_msrc.pt"
        if not ckpt_path.exists():
            self.info.setPlainText(f"缺少模型文件: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = UNet(num_classes=int(ckpt["num_classes"]), base=int(ckpt["base_channels"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.seg_model_cache = (model, ckpt)
        return self.seg_model_cache

    def load_sr_model(self) -> tuple[torch.nn.Module, dict] | None:
        if self.sr_model_cache is not None:
            return self.sr_model_cache
        ckpt_path = OUT / "exp2_sr" / "checkpoints" / "srcnn_bsds500.pt"
        if not ckpt_path.exists():
            self.info.setPlainText(f"缺少模型文件: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SRCNN()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.sr_model_cache = (model, ckpt)
        return self.sr_model_cache

    def set_seg_info(self, source: str, area_summary: str = "") -> None:
        rows = read_rows(OUT / "exp2_seg" / "segmentation_results.csv")
        area_line = f"预测区域占比: {area_summary}\n" if area_summary else ""
        self.info.setPlainText(
            f"分割输入: {source}\n"
            f"{area_line}"
            f"模型: UNet    mIoU: {rows[0].get('miou') if rows else 'n/a'}\n"
            f"checkpoint: {OUT / 'exp2_seg' / 'checkpoints' / 'unet_msrc.pt'}"
        )

    def set_sr_info(self, source: str, case_metrics: str = "") -> None:
        rows = read_rows(OUT / "exp2_sr" / "super_resolution_results.csv")
        metric_line = f"当前图像指标: {case_metrics}\n" if case_metrics else ""
        self.info.setPlainText(
            f"超分输入: {source}\n"
            f"{metric_line}"
            f"模型: SRCNN    PSNR: {rows[0].get('psnr') if rows else 'n/a'}    SSIM: {rows[0].get('ssim') if rows else 'n/a'}\n"
            f"checkpoint: {OUT / 'exp2_sr' / 'checkpoints' / 'srcnn_bsds500.pt'}"
        )

    def run_segmentation(self) -> None:
        loaded = self.load_seg_model()
        if loaded is None:
            return
        model, ckpt = loaded
        ds = MSRCSegmentationDataset(ROOT / "datasets" / "msrc2_seg", "val", int(ckpt["image_size"]))
        idx = random.randrange(len(ds))
        x, y = ds[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).argmax(dim=1)[0].numpy()
        image = ((x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        path = TMP / "segmentation_live.png"
        make_triptych(
            [Image.fromarray(image), Image.fromarray(mask_to_color(y.numpy())), Image.fromarray(mask_to_color(pred))],
            ["input", "ground truth", "UNet prediction"],
            path,
        )
        set_pixmap(self.seg_image, path, 760, 300)
        self.set_seg_info(f"MSRC-V2 验证集样本 #{idx}", segmentation_area_summary(pred))

    def upload_segmentation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", str(ROOT), "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.run_uploaded_segmentation(Path(path))

    def run_uploaded_segmentation(self, path: Path) -> None:
        loaded = self.load_seg_model()
        if loaded is None:
            return
        model, ckpt = loaded
        x, preview = seg_tensor_from_pil(Image.open(path), int(ckpt["image_size"]))
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).argmax(dim=1)[0].numpy()
        out_path = TMP / "segmentation_upload.png"
        make_triptych(
            [preview, Image.fromarray(mask_to_color(pred))],
            ["uploaded input", "UNet prediction"],
            out_path,
        )
        set_pixmap(self.seg_image, out_path, 760, 300)
        self.set_seg_info(f"上传图片 {path.name}", segmentation_area_summary(pred))

    def run_sr(self) -> None:
        loaded = self.load_sr_model()
        if loaded is None:
            return
        model, ckpt = loaded
        ds = BSDS500SRDataset(ROOT / "datasets" / "bsds500", "test", int(ckpt["scale"]), int(ckpt["image_size"]))
        idx = random.randrange(len(ds))
        x, y = ds[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0))[0].clamp(0, 1)
        path = TMP / "sr_live.png"
        make_triptych(
            [tensor_to_image(x), tensor_to_image(pred), tensor_to_image(y)],
            ["bicubic input", "SRCNN output", "ground truth"],
            path,
        )
        set_pixmap(self.sr_image, path, 760, 300)
        case_metrics = (
            f"SRCNN PSNR={psnr(pred, y):.3f}, SSIM={batch_ssim(pred.unsqueeze(0), y.unsqueeze(0)):.3f}; "
            f"Bicubic PSNR={psnr(x, y):.3f}, SSIM={batch_ssim(x.unsqueeze(0), y.unsqueeze(0)):.3f}"
        )
        self.set_sr_info(f"BSDS500 测试集样本 #{idx}", case_metrics)

    def upload_sr(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", str(ROOT), "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.run_uploaded_sr(Path(path))

    def run_uploaded_sr(self, path: Path) -> None:
        loaded = self.load_sr_model()
        if loaded is None:
            return
        model, ckpt = loaded
        arr = sr_uint8_from_pil(Image.open(path), int(ckpt["image_size"]), int(ckpt["scale"]))
        x, y = make_sr_pair(arr, int(ckpt["scale"]))
        with torch.no_grad():
            pred = model(x.unsqueeze(0))[0].clamp(0, 1)
        out_path = TMP / "sr_upload.png"
        make_triptych(
            [tensor_to_image(x), tensor_to_image(pred), tensor_to_image(y)],
            ["bicubic input", "SRCNN output", "source crop"],
            out_path,
        )
        set_pixmap(self.sr_image, out_path, 760, 300)
        case_metrics = (
            f"SRCNN PSNR={psnr(pred, y):.3f}, SSIM={batch_ssim(pred.unsqueeze(0), y.unsqueeze(0)):.3f}; "
            f"Bicubic PSNR={psnr(x, y):.3f}, SSIM={batch_ssim(x.unsqueeze(0), y.unsqueeze(0)):.3f}"
        )
        self.set_sr_info(f"上传图片 {path.name}", case_metrics)


class SequenceDemo(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.weather_model_cache: dict[str, tuple[torch.nn.Module, dict]] = {}
        self.char_model_cache: tuple[CharRNN, object, dict] | None = None
        self.weather_csv_path = ROOT / "datasets" / "jena_climate_2009_2016.csv"
        self.last_generated_text = ""

        self.weather_box = QComboBox()
        self.weather_box.addItems(["rnn", "gru", "lstm"])
        self.weather_index = QSpinBox()
        self.weather_index.setRange(0, 999999)
        self.weather_index.setValue(0)
        self.weather_img = QLabel(alignment=Qt.AlignCenter)
        self.weather_img.setMinimumHeight(330)
        self.weather_info = QTextEdit(readOnly=True)
        self.weather_info.setMaximumHeight(170)

        self.prompt = QLineEdit("ROMEO:")
        self.seed_box = QComboBox()
        self.seed_box.addItems(["ROMEO:", "JULIET:", "HAMLET:", "KING:", "To be, or not to be"])
        self.seed_box.currentTextChanged.connect(self.prompt.setText)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(100, 2000)
        self.length_spin.setSingleStep(100)
        self.length_spin.setValue(800)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.2, 1.5)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.8)
        self.generated = QTextEdit(readOnly=True)
        self.char_info = QTextEdit(readOnly=True)
        self.char_info.setMaximumHeight(145)

        random_weather_btn = QPushButton("随机窗口预测")
        random_weather_btn.clicked.connect(self.run_weather)
        random_weather_btn.setToolTip("随机抽取 Jena 测试窗口并运行当前模型。")
        indexed_weather_btn = QPushButton("指定索引预测")
        indexed_weather_btn.clicked.connect(self.run_weather_index)
        indexed_weather_btn.setToolTip("使用样本索引框中的测试窗口运行当前模型。")
        latest_weather_btn = QPushButton("最近窗口预测")
        latest_weather_btn.clicked.connect(self.run_weather_latest)
        latest_weather_btn.setToolTip("选择当前 CSV 测试段最后一个可用窗口。")
        compare_weather_btn = QPushButton("三模型同窗对比")
        compare_weather_btn.clicked.connect(self.run_weather_compare)
        compare_weather_btn.setToolTip("同一个天气窗口同时运行 RNN、GRU、LSTM。")
        upload_weather_btn = QPushButton("上传Jena格式CSV")
        upload_weather_btn.clicked.connect(self.upload_weather_csv)
        upload_weather_btn.setToolTip("上传包含 Jena Climate 字段的 CSV 后重新构建测试窗口。")

        random_seed_btn = QPushButton("随机语料开头")
        random_seed_btn.clicked.connect(self.pick_random_text_seed)
        random_seed_btn.setToolTip("从真实 Shakespeare 语料中抽取一段开头。")
        gen_btn = QPushButton("按参数生成")
        gen_btn.clicked.connect(self.run_generation)
        gen_btn.setToolTip("按当前开头、长度和温度生成文本。")
        compare_gen_btn = QPushButton("多温度对比")
        compare_gen_btn.clicked.connect(self.run_generation_compare)
        compare_gen_btn.setToolTip("同一开头分别用 temperature 0.4、0.8、1.2 生成。")
        export_btn = QPushButton("导出生成文本")
        export_btn.clicked.connect(self.export_generated_text)
        export_btn.setToolTip("将当前生成结果保存为 txt。")

        weather_tab = QWidget()
        weather_layout = QVBoxLayout(weather_tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("气温模型"))
        row.addWidget(self.weather_box)
        row.addWidget(QLabel("样本索引"))
        row.addWidget(self.weather_index)
        row.addWidget(random_weather_btn)
        row.addWidget(indexed_weather_btn)
        row.addWidget(latest_weather_btn)
        row.addWidget(compare_weather_btn)
        row.addWidget(upload_weather_btn)
        weather_layout.addWidget(group_box("预测窗口与模型", row))
        weather_plot_layout = QVBoxLayout()
        weather_plot_layout.addWidget(self.weather_img, stretch=1)
        weather_layout.addWidget(group_box("预测曲线", weather_plot_layout), stretch=1)
        weather_info_layout = QVBoxLayout()
        weather_info_layout.addWidget(self.weather_info)
        weather_layout.addWidget(group_box("指标与模型产物", weather_info_layout))

        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("预设开头"))
        seed_row.addWidget(self.seed_box)
        seed_row.addWidget(random_seed_btn)
        text_layout.addWidget(group_box("文本开头", seed_row))
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("输入开头"))
        gen_row.addWidget(self.prompt)
        gen_row.addWidget(QLabel("长度"))
        gen_row.addWidget(self.length_spin)
        gen_row.addWidget(QLabel("温度"))
        gen_row.addWidget(self.temp_spin)
        gen_row.addWidget(gen_btn)
        gen_row.addWidget(compare_gen_btn)
        gen_row.addWidget(export_btn)
        text_layout.addWidget(group_box("生成参数", gen_row))
        generated_layout = QVBoxLayout()
        generated_layout.addWidget(self.generated, stretch=1)
        text_layout.addWidget(group_box("生成结果", generated_layout), stretch=1)
        char_info_layout = QVBoxLayout()
        char_info_layout.addWidget(self.char_info)
        text_layout.addWidget(group_box("指标与模型产物", char_info_layout))

        tabs = QTabWidget()
        tabs.addTab(weather_tab, "Jena气温预测")
        tabs.addTab(text_tab, "Shakespeare生成")
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        self.refresh_weather_limit()

    def load_weather_model(self, name: str) -> tuple[torch.nn.Module, dict] | None:
        if name in self.weather_model_cache:
            return self.weather_model_cache[name]
        ckpt_path = OUT / "exp3_weather" / "checkpoints" / f"weather_{name}.pt"
        if not ckpt_path.exists():
            self.weather_info.setPlainText(f"缺少模型文件: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SequenceRegressor(
            name,
            int(ckpt["input_dim"]),
            int(ckpt["horizon"]),
            int(ckpt["hidden_size"]),
            int(ckpt["num_layers"]),
            float(ckpt["dropout"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.weather_model_cache[name] = (model, ckpt)
        return self.weather_model_cache[name]

    def build_weather_dataset(self, csv_path: Path, ckpt: dict) -> tuple[WindowedArrayDataset, int] | None:
        try:
            _, _, test, _, target_index = preprocess_jena(csv_path)
        except Exception as exc:
            self.weather_info.setPlainText(f"无法读取Jena格式CSV: {csv_path}\n{exc}")
            return None
        ds = WindowedArrayDataset(test, int(ckpt["input_width"]), int(ckpt["horizon"]), target_index)
        if len(ds) == 0:
            self.weather_info.setPlainText(
                f"CSV可用测试窗口不足: {csv_path}\n"
                f"需要至少 input_width={ckpt['input_width']} + horizon={ckpt['horizon']} 个归一化时间步。"
            )
            return None
        return ds, target_index

    def refresh_weather_limit(self) -> None:
        loaded = self.load_weather_model("rnn")
        if loaded is None:
            return
        _, ckpt = loaded
        built = self.build_weather_dataset(self.weather_csv_path, ckpt)
        if built is None:
            return
        ds, _ = built
        self.weather_index.setMaximum(max(0, len(ds) - 1))

    def run_weather(self) -> None:
        self.run_weather_single(random_case=True)

    def run_weather_index(self) -> None:
        self.run_weather_single(random_case=False)

    def run_weather_latest(self) -> None:
        loaded = self.load_weather_model(self.weather_box.currentText())
        if loaded is None:
            return
        _, ckpt = loaded
        built = self.build_weather_dataset(self.weather_csv_path, ckpt)
        if built is None:
            return
        ds, _ = built
        self.weather_index.setValue(max(0, len(ds) - 1))
        self.run_weather_index()

    def run_weather_single(self, random_case: bool) -> None:
        name = self.weather_box.currentText()
        loaded = self.load_weather_model(name)
        if loaded is None:
            return
        model, ckpt = loaded
        built = self.build_weather_dataset(self.weather_csv_path, ckpt)
        if built is None:
            return
        ds, target_index = built
        idx = random.randrange(len(ds)) if random_case else min(self.weather_index.value(), len(ds) - 1)
        self.weather_index.setValue(idx)
        x, y = ds[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0))[0].numpy()
        path = self.plot_weather_case(
            x,
            y,
            target_index,
            {name.upper(): pred},
            f"{name.upper()} next-week temperature forecast",
            f"weather_{name}_live.png",
        )
        set_pixmap(self.weather_img, path, 880, 420)
        mse = float(np.mean((pred - y.numpy()) ** 2))
        mae = float(np.mean(np.abs(pred - y.numpy())))
        rows = read_rows(OUT / "exp3_weather" / "weather_results.csv")
        metric = next((r for r in rows if r.get("model") == name), {})
        self.weather_info.setPlainText(
            f"输入来源: {self.weather_csv_path.name} 测试窗口 #{idx}\n"
            f"当前样本误差: MSE={mse:.6f}    MAE={mae:.6f}\n"
            f"整体测试指标: {name.upper()} MSE={metric.get('mse', 'n/a')}    MAE={metric.get('mae', 'n/a')}\n"
            f"输入长度/预测长度: {ckpt['input_width']}h -> {ckpt['horizon']}h\n"
            f"checkpoint: {OUT / 'exp3_weather' / 'checkpoints' / f'weather_{name}.pt'}"
        )

    def run_weather_compare(self) -> None:
        base_loaded = self.load_weather_model("rnn")
        if base_loaded is None:
            return
        _, base_ckpt = base_loaded
        built = self.build_weather_dataset(self.weather_csv_path, base_ckpt)
        if built is None:
            return
        ds, target_index = built
        idx = min(self.weather_index.value(), len(ds) - 1)
        self.weather_index.setValue(idx)
        x, y = ds[idx]
        predictions: dict[str, np.ndarray] = {}
        metric_lines = []
        rows = read_rows(OUT / "exp3_weather" / "weather_results.csv")
        for name in ["rnn", "gru", "lstm"]:
            loaded = self.load_weather_model(name)
            if loaded is None:
                return
            model, _ = loaded
            with torch.no_grad():
                pred = model(x.unsqueeze(0))[0].numpy()
            predictions[name.upper()] = pred
            mse = float(np.mean((pred - y.numpy()) ** 2))
            mae = float(np.mean(np.abs(pred - y.numpy())))
            metric = next((r for r in rows if r.get("model") == name), {})
            metric_lines.append(
                f"{name.upper()}: 当前窗口 MSE={mse:.6f}, MAE={mae:.6f}; "
                f"整体测试 MSE={metric.get('mse', 'n/a')}, MAE={metric.get('mae', 'n/a')}"
            )
        path = self.plot_weather_case(
            x,
            y,
            target_index,
            predictions,
            "RNN / GRU / LSTM same-window forecast comparison",
            "weather_compare_live.png",
        )
        set_pixmap(self.weather_img, path, 880, 420)
        self.weather_info.setPlainText(
            f"输入来源: {self.weather_csv_path.name} 测试窗口 #{idx}\n"
            f"输入长度/预测长度: {base_ckpt['input_width']}h -> {base_ckpt['horizon']}h\n"
            + "\n".join(metric_lines)
        )

    def plot_weather_case(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_index: int,
        predictions: dict[str, np.ndarray],
        title: str,
        filename: str,
    ) -> Path:
        history = x[:, target_index].numpy()
        truth = y.numpy()
        hist_x = np.arange(-len(history), 0)
        future_x = np.arange(len(truth))
        path = TMP / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 4.1), dpi=140)
        plt.plot(hist_x, history, label="input history", color="#444444", linewidth=1.5)
        plt.plot(future_x, truth, label="true future", color="#111111", linewidth=2.2)
        for label, pred in predictions.items():
            plt.plot(future_x, pred, label=f"{label} prediction", linewidth=1.8)
        plt.axvline(-0.5, color="#999999", linestyle="--", linewidth=1)
        plt.title(title)
        plt.xlabel("hours relative to forecast start")
        plt.ylabel("normalized T (degC)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def upload_weather_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择Jena格式CSV", str(ROOT), "CSV (*.csv)")
        if not path:
            return
        self.weather_csv_path = Path(path)
        self.refresh_weather_limit()
        self.run_weather()

    def load_char_model(self) -> tuple[CharRNN, object, dict] | None:
        if self.char_model_cache is not None:
            return self.char_model_cache
        ckpt_path = OUT / "exp3_shakespeare" / "checkpoints" / "shakespeare_lstm.pt"
        if not ckpt_path.exists():
            self.char_info.setPlainText(f"缺少模型文件: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CharRNN(len(ckpt["vocab"]), ckpt["model"], int(ckpt["embed_dim"]), int(ckpt["hidden_size"]), int(ckpt["num_layers"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        vocab = type("Vocab", (), {})()
        vocab.chars = ckpt["vocab"]
        vocab.stoi = {ch: i for i, ch in enumerate(vocab.chars)}
        vocab.itos = {i: ch for ch, i in vocab.stoi.items()}
        self.char_model_cache = (model, vocab, ckpt)
        return self.char_model_cache

    def run_generation(self) -> None:
        loaded = self.load_char_model()
        if loaded is None:
            return
        model, vocab, ckpt = loaded
        seed = self.prompt.text() or "ROMEO:"
        length = int(self.length_spin.value())
        temperature = float(self.temp_spin.value())
        text = generate_text(model, vocab, seed, length, temperature, torch.device("cpu"))
        self.last_generated_text = text
        self.generated.setPlainText(text)
        result_path = OUT / "exp3_shakespeare" / "shakespeare_results.json"
        train_loss = "n/a"
        if result_path.exists():
            try:
                train_loss = str(json.loads(result_path.read_text(encoding="utf-8")).get("train_loss", "n/a"))
            except json.JSONDecodeError:
                train_loss = "n/a"
        self.char_info.setPlainText(
            f"模型: {ckpt['model'].upper()}    vocab={len(ckpt['vocab'])}    train_loss={train_loss}\n"
            f"生成参数: seed_len={len(seed)}    generate_length={length}    temperature={temperature:.2f}\n"
            f"checkpoint: {OUT / 'exp3_shakespeare' / 'checkpoints' / 'shakespeare_lstm.pt'}"
        )

    def run_generation_compare(self) -> None:
        loaded = self.load_char_model()
        if loaded is None:
            return
        model, vocab, ckpt = loaded
        seed = self.prompt.text() or "ROMEO:"
        length = int(self.length_spin.value())
        sections = []
        for temperature in [0.4, 0.8, 1.2]:
            text = generate_text(model, vocab, seed, length, temperature, torch.device("cpu"))
            sections.append(f"===== temperature={temperature:.1f} =====\n{text}")
        self.last_generated_text = "\n\n".join(sections)
        self.generated.setPlainText(self.last_generated_text)
        self.char_info.setPlainText(
            f"模型: {ckpt['model'].upper()}    同一开头多温度生成对比\n"
            f"生成参数: seed_len={len(seed)}    each_length={length}    temperatures=0.4/0.8/1.2\n"
            f"checkpoint: {OUT / 'exp3_shakespeare' / 'checkpoints' / 'shakespeare_lstm.pt'}"
        )

    def pick_random_text_seed(self) -> None:
        text_path = ROOT / "datasets" / "shakespeare.txt"
        if not text_path.exists():
            self.char_info.setPlainText(f"缺少语料文件: {text_path}")
            return
        text = text_path.read_text(encoding="utf-8")
        if len(text) < 80:
            self.prompt.setText(text[:30])
            return
        start = random.randrange(0, len(text) - 80)
        seed = text[start : start + 40].replace("\r", " ").replace("\n", " ")
        self.prompt.setText(" ".join(seed.split()))

    def export_generated_text(self) -> None:
        text = self.generated.toPlainText()
        if not text:
            self.char_info.setPlainText("当前还没有可导出的生成文本。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出生成文本", str(TMP / "shakespeare_generated_demo.txt"), "Text (*.txt)")
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")
        self.char_info.setPlainText(f"已导出: {path}")


class SummaryTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        text = QTextEdit(readOnly=True)
        parts = []
        for path in [
            OUT / "exp1" / "classification_results.csv",
            OUT / "exp2_seg" / "segmentation_results.csv",
            OUT / "exp2_sr" / "super_resolution_results.csv",
            OUT / "exp3_weather" / "weather_results.csv",
            OUT / "exp3_shakespeare" / "shakespeare_results.json",
        ]:
            parts.append(f"===== {path.relative_to(ROOT)} =====\n")
            parts.append(path.read_text(encoding="utf-8") if path.exists() else "missing\n")
            parts.append("\n")
        text.setPlainText("".join(parts))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("训练结果汇总"))
        layout.addWidget(text)


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("深度学习实验成果演示系统")
        self.resize(1120, 860)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { font-size: 13px; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #b9bec8;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                min-height: 28px;
                padding: 4px 10px;
            }
            QTableWidget {
                gridline-color: #d6dae2;
                selection-background-color: #dcecff;
            }
            """
        )
        tabs = QTabWidget()
        tabs.addTab(ClassificationDemo(), "实验1：图像分类")
        tabs.addTab(VisionDemo(), "实验2：视觉增强")
        tabs.addTab(SequenceDemo(), "实验3：序列建模")
        tabs.addTab(SummaryTab(), "训练产物")
        self.setCentralWidget(tabs)


def main() -> None:
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
