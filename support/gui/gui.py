from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


class ProcessPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.process: QProcess | None = None
        self.command_line = QLineEdit()
        self.command_line.setReadOnly(True)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.stop_button = QPushButton("停止当前任务")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("当前命令"))
        layout.addWidget(self.command_line)
        layout.addWidget(self.output, stretch=1)
        layout.addWidget(self.stop_button)

    def run(self, args: list[str]) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "任务正在运行", "请先停止或等待当前任务结束。")
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self.output.clear()
        self.command_line.setText(" ".join(args))
        self.process = QProcess(self)
        proc_env = self.process.processEnvironment()
        for key, value in env.items():
            proc_env.insert(key, value)
        self.process.setProcessEnvironment(proc_env)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProgram(args[0])
        self.process.setArguments(args[1:])
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.stop_button.setEnabled(True)
        self.output.append("[start] " + " ".join(args))
        self.process.start()

    def _read_stdout(self) -> None:
        if self.process:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.output.moveCursor(self.output.textCursor().End)
            self.output.insertPlainText(text)

    def _read_stderr(self) -> None:
        if self.process:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            self.output.moveCursor(self.output.textCursor().End)
            self.output.insertPlainText(text)

    def _finished(self, code: int) -> None:
        self.output.append(f"\n[finished] exit code={code}")
        self.stop_button.setEnabled(False)

    def stop(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()


class LabTab(QWidget):
    def __init__(self, title: str, description: str, image_paths: list[Path], buttons: list[tuple[str, list[str]]], process_panel: ProcessPanel) -> None:
        super().__init__()
        self.process_panel = process_panel
        root = QVBoxLayout(self)
        header = QLabel(f"<h2>{title}</h2><p>{description}</p>")
        header.setWordWrap(True)
        root.addWidget(header)

        btn_row = QHBoxLayout()
        for label, command in buttons:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, cmd=command: self.process_panel.run(cmd))
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        image_box = QGroupBox("实验结构与输入输出示意")
        image_layout = QGridLayout(image_box)
        for i, path in enumerate(image_paths):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if path.exists():
                pixmap = QPixmap(str(path))
                label.setPixmap(pixmap.scaled(420, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                label.setText(str(path))
            image_layout.addWidget(label, i // 2, i % 2)
        root.addWidget(image_box, stretch=1)


class Dashboard(QWidget):
    def __init__(self, process_panel: ProcessPanel) -> None:
        super().__init__()
        self.process_panel = process_panel
        self.status_view = QTextEdit()
        self.status_view.setReadOnly(True)
        self.status_view.setFont(QFont("Consolas", 10))
        layout = QVBoxLayout(self)
        intro = QLabel(
            "<h2>深度学习实验验收面板</h2>"
            "<p>这里把三个 PPT 实验封装为可点击的工程化入口：数据准备、完整真实训练、快速验收、结果查看。</p>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        refresh = QPushButton("刷新真实数据状态")
        refresh.clicked.connect(self.refresh_status)
        download = QPushButton("下载公开真实数据")
        download.clicked.connect(lambda: self.process_panel.run([PYTHON, "support/scripts/prepare_real_datasets.py", "--all-public"]))
        full = QPushButton("运行全部真实实验")
        full.clicked.connect(lambda: self.process_panel.run([PYTHON, "support/scripts/run_full_real_experiments.py", "--profile", "full"]))
        acceptance = QPushButton("运行真实数据验收版")
        acceptance.clicked.connect(
            lambda: self.process_panel.run([PYTHON, "support/scripts/run_full_real_experiments.py", "--profile", "acceptance"])
        )
        smoke = QPushButton("运行快速 smoke test")
        smoke.clicked.connect(lambda: self.process_panel.run([PYTHON, "support/scripts/run_smoke_tests.py"]))
        for btn in [refresh, download, acceptance, full, smoke]:
            row.addWidget(btn)
        layout.addLayout(row)
        layout.addWidget(self.status_view, stretch=1)

        result_row = QHBoxLayout()
        for label, folder in [
            ("打开 smoke 输出", ROOT / "outputs" / "smoke"),
            ("打开真实训练输出", ROOT / "outputs" / "full_real"),
            ("打开报告", ROOT / "reports"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, p=folder: QDesktopServices.openUrl(p.as_uri()))
            result_row.addWidget(btn)
        layout.addLayout(result_row)
        self.refresh_status()

    def refresh_status(self) -> None:
        import subprocess

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [PYTHON, "support/scripts/prepare_real_datasets.py", "--status"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            self.status_view.setPlainText(proc.stdout)
        else:
            self.status_view.setPlainText(proc.stdout + "\n" + proc.stderr)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("深度学习实验工程化验收 GUI")
        self.resize(1280, 820)
        self.process_panel = ProcessPanel()
        tabs = QTabWidget()
        tabs.addTab(Dashboard(self.process_panel), "总览")
        tabs.addTab(self.exp1_tab(), "实验1 图像分类")
        tabs.addTab(self.exp2_tab(), "实验2 分割/超分")
        tabs.addTab(self.exp3_tab(), "实验3 RNN")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tabs)
        splitter.addWidget(self.process_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

    def exp1_tab(self) -> QWidget:
        return LabTab(
            "实验 1：图像分类",
            "在 MNIST、Fashion MNIST、HWDB1 上训练 MLP、LeNet、AlexNet、GoogLeNet、ResNet，并对比参数量、网络深度、准确率和 batch size 影响。",
            [
                ROOT / "artifacts/ppt_media/实验1-图像分类(1)_image14.png",
                ROOT / "artifacts/ppt_media/实验1-图像分类(1)_image13.png",
                ROOT / "artifacts/ppt_media/实验1-图像分类(1)_image15.png",
                ROOT / "artifacts/ppt_media/实验1-图像分类(1)_image17.png",
            ],
            [
                (
                    "运行实验1真实训练",
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
                        "10",
                        "--batch-size",
                        "64",
                        "--batch-sizes",
                        "16",
                        "32",
                        "64",
                        "128",
                        "--output-dir",
                        "outputs/full_real/exp1",
                    ],
                ),
                (
                    "运行实验1快速验收",
                    [
                        PYTHON,
                        "-m",
                        "experiment1_image_classification.classification",
                        "--datasets",
                        "synthetic",
                        "--models",
                        "mlp",
                        "lenet",
                        "resnet",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "8",
                        "--output-dir",
                        "outputs/gui_acceptance/exp1",
                    ],
                ),
            ],
            self.process_panel,
        )

    def exp2_tab(self) -> QWidget:
        return LabTab(
            "实验 2：语义分割与图像超分",
            "UNet 对 MSRC-V2 做像素级类别预测并输出 mIoU；SRCNN 对 BSDS500 做 2 倍超分并输出 PSNR、SSIM 与 bicubic baseline。",
            [
                ROOT / "artifacts/ppt_media/实验2-图像分割-超分(1)_image7_white.png",
                ROOT / "artifacts/ppt_media/实验2-图像分割-超分(1)_image13.png",
                ROOT / "artifacts/ppt_media/实验2-图像分割-超分(1)_image14.png",
            ],
            [
                (
                    "运行语义分割真实训练",
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
                        "120",
                        "--batch-size",
                        "8",
                        "--base-channels",
                        "32",
                        "--augment",
                        "--samples-per-image",
                        "4",
                        "--output-dir",
                        "outputs/full_real/exp2_seg",
                    ],
                ),
                (
                    "运行超分真实训练",
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
                        "100",
                        "--batch-size",
                        "16",
                        "--patches-per-image",
                        "16",
                        "--augment",
                        "--output-dir",
                        "outputs/full_real/exp2_sr",
                    ],
                ),
                (
                    "运行实验2快速验收",
                    [PYTHON, "support/scripts/run_smoke_tests.py"],
                ),
            ],
            self.process_panel,
        )

    def exp3_tab(self) -> QWidget:
        return LabTab(
            "实验 3：循环神经网络",
            "RNN、GRU、LSTM 预测未来一周气温；字符级 LSTM 根据输入对话生成莎士比亚风格剧本。",
            [ROOT / "artifacts/ppt_media/实验1-图像分类(1)_image16.png"],
            [
                (
                    "运行气温预测真实训练",
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
                        "20",
                        "--batch-size",
                        "64",
                        "--output-dir",
                        "outputs/full_real/exp3_weather",
                    ],
                ),
                (
                    "运行剧本生成真实训练",
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
                        "20",
                        "--batch-size",
                        "64",
                        "--seed-text",
                        "ROMEO:",
                        "--generate-length",
                        "1000",
                        "--output-dir",
                        "outputs/full_real/exp3_shakespeare",
                    ],
                ),
                (
                    "运行实验3快速验收",
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
                        "--input-width",
                        "24",
                        "--horizon",
                        "12",
                        "--output-dir",
                        "outputs/gui_acceptance/exp3_weather",
                    ],
                ),
            ],
            self.process_panel,
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
