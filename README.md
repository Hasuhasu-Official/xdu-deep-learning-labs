# XDU 深度学习实验工程

本仓库整理了三个深度学习实验的可运行代码、训练脚本和图形化演示入口。目录按实验任务直接展开，便于查看每个实验的模型实现、训练命令和演示方式。

## 项目结构

```text
experiment1_image_classification/              # 实验 1：MNIST、Fashion-MNIST、HWDB1 图像分类
experiment2_segmentation_super_resolution/     # 实验 2：MSRC-V2 分割与 BSDS500 超分
experiment3_recurrent_neural_networks/         # 实验 3：Jena 气象预测与 Shakespeare 文本生成
support/
  gui/                                         # 训练控制界面与最终演示界面
  scripts/                                     # 数据准备、完整训练、质量复查脚本
  shared/                                      # 公共工具函数
docs/                                          # 数据集和模型产物说明
requirements.txt                               # Python 依赖
```

## 快速运行

在仓库根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

启动最终演示系统：

```powershell
python -m support.gui.demo_gui
```

启动训练控制界面：

```powershell
python -m support.gui.gui
```

如果从仓库根目录之外启动程序，请先把项目根目录加入 `PYTHONPATH`。

## 数据与模型产物

数据集不随代码仓库提交，可通过脚本准备：

```powershell
python support\scripts\prepare_real_datasets.py --all-public
python support\scripts\prepare_real_datasets.py --status
```

完整训练三个实验：

```powershell
python support\scripts\run_full_real_experiments.py --profile full
```

使用较短的真实数据流程生成可演示产物：

```powershell
python support\scripts\run_full_real_experiments.py --profile acceptance
```

复查已生成的模型产物：

```powershell
python support\scripts\audit_model_quality.py
```

## 实验 1：图像分类

```powershell
python -m experiment1_image_classification.classification `
  --datasets mnist fashion_mnist hwdb `
  --models mlp lenet alexnet googlenet resnet `
  --data-root datasets `
  --download `
  --image-size 64 `
  --epochs 10 `
  --batch-size 64 `
  --batch-sizes 16 32 64 128 `
  --output-dir outputs\exp1
```

输出：

- `outputs\exp1\classification_results.csv`
- `outputs\exp1\classification_results.json`
- `outputs\exp1\checkpoints\*.pt`

## 实验 2：语义分割与图像超分

语义分割：

```powershell
python -m experiment2_segmentation_super_resolution.segmentation_sr segmentation `
  --dataset msrc `
  --data-root datasets\msrc2_seg `
  --image-size 128 `
  --epochs 120 `
  --batch-size 8 `
  --base-channels 32 `
  --augment `
  --samples-per-image 4 `
  --output-dir outputs\exp2_seg
```

图像超分：

```powershell
python -m experiment2_segmentation_super_resolution.segmentation_sr super-resolution `
  --dataset bsds500 `
  --data-root datasets\bsds500 `
  --scale 4 `
  --image-size 96 `
  --epochs 100 `
  --batch-size 16 `
  --patches-per-image 16 `
  --augment `
  --output-dir outputs\exp2_sr
```

输出：

- `outputs\exp2_seg\segmentation_results.csv`
- `outputs\exp2_sr\super_resolution_results.csv`
- `outputs\exp2_seg\checkpoints\*.pt`
- `outputs\exp2_sr\checkpoints\*.pt`

## 实验 3：循环神经网络

Jena 气象预测：

```powershell
python -m experiment3_recurrent_neural_networks.sequence weather `
  --dataset jena `
  --csv-path datasets\jena_climate_2009_2016.csv `
  --models rnn gru lstm `
  --input-width 168 `
  --horizon 168 `
  --epochs 20 `
  --batch-size 64 `
  --output-dir outputs\exp3_weather
```

Shakespeare 字符级文本生成：

```powershell
python -m experiment3_recurrent_neural_networks.sequence shakespeare `
  --dataset shakespeare `
  --text-file datasets\shakespeare.txt `
  --rnn-type lstm `
  --seq-len 100 `
  --epochs 20 `
  --seed-text "ROMEO:" `
  --generate-length 1000 `
  --output-dir outputs\exp3_shakespeare
```

输出：

- `outputs\exp3_weather\weather_results.csv`
- `outputs\exp3_shakespeare\shakespeare_generated.txt`
- `outputs\exp3_weather\checkpoints\*.pt`
- `outputs\exp3_shakespeare\checkpoints\*.pt`
