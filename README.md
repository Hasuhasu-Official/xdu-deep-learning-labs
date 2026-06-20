# 深度学习实验实现

本目录已按 3 个 PPT 的要求整理为一个 PyTorch 工程，覆盖：

- 实验 1：MNIST、Fashion MNIST、HWDB1 的图像分类；模型包括 MLP、LeNet、AlexNet、GoogLeNet、ResNet，并输出模型性能、参数量、深度和 batch size 对比。
- 实验 2：MSRC-V2 语义分割，使用 UNet，评价 mIoU；BSDS500 x4 图像超分，使用 residual SRCNN，评价 PSNR、SSIM，并与 bicubic 对比。
- 实验 3：Jena 气温一周预测，使用 RNN、GRU、LSTM 对比；莎士比亚风格字符级剧本生成。

## GitHub 项目说明

本仓库面向课程验收整理，定位为代码仓库。仓库保留源码、训练脚本、GUI 演示入口和必要的复现实验说明；三份实验报告、真实数据集、模型 checkpoint、训练输出、原始 PPT 和本地 LaTeX/PDF 检查依赖不直接提交到 GitHub，原因是它们属于课程提交材料、可重建的大体积产物或本地辅助素材。

主要结构如下：

```text
src/dl_labs/      # 三个实验的模型、训练与 GUI 代码
  exp1_classification/  # 实验 1：图像分类
  exp2_vision/          # 实验 2：语义分割与图像超分
  exp3_sequence/        # 实验 3：循环神经网络
  apps/                 # 训练控制 GUI 与最终演示 GUI
  common/               # 公共工具函数
scripts/          # 数据准备、完整训练、质量复查和 smoke test 脚本
DATASETS.md       # 数据集来源、准备命令和本地目录约定
ARTIFACTS.md      # 训练产物、checkpoint 和审计输出说明
requirements.txt  # Python 依赖
```

重建顺序为：

1. 运行 `python scripts\prepare_real_datasets.py --all-public` 准备公开数据集。
2. 运行 `python scripts\run_full_real_experiments.py --profile acceptance` 或 `--profile full` 训练模型并生成产物。
3. 运行 `python -m dl_labs.demo_gui` 进入最终工程化演示界面。

## 环境

PowerShell 中先设置模块路径：

```powershell
$env:PYTHONPATH = "$PWD\src"
```

启动图形化验收界面：

```powershell
python -m dl_labs.gui
```

启动最终成果演示系统：

```powershell
python -m dl_labs.demo_gui
```

也可以直接运行文件：

```powershell
C:\Users\rytram\AppData\Local\Programs\Python\Python311\python.exe D:\code\深度学习-xdu\src\dl_labs\demo_gui.py
```

`demo_gui` 不只是启动训练命令；它会加载 `outputs\full_real` 下保存的模型 checkpoint 和真实测试样本，直接完成分类、分割、超分、气温预测和剧本生成案例推理。主界面默认进入实验 1 的可操作场景，结果汇总放在最后一个页签。

- 实验 1：按“输入来源、预处理预览、手写输入、五模型对比结果、运行记录”组织。使用者选择 MNIST、Fashion MNIST 或 HWDB1 后，可以抽取真实样例、上传图片或在画板手写，同一张输入会同时送入 MLP、LeNet、AlexNet、GoogLeNet、ResNet，并在表格中并排显示预测、置信度、Top-3、测试准确率和参数量。
- 实验 2：按“语义分割：场景理解”和“图像超分：清晰化”组织。分割页会对真实样例或上传图片输出 UNet 区域预测和类别占比；超分页会按 checkpoint 中的 x4 退化设置展示 bicubic、SRCNN、原图裁剪，并给出当前图像 PSNR/SSIM 对比。
- 实验 3：按“Jena 气温预测”和“Shakespeare 生成”组织。气温预测支持随机窗口、指定索引、最近窗口、三模型同窗对比、Jena 格式 CSV 上传预测；文本生成支持开头、长度、温度控制、多温度对比和导出。

快速验证全部入口：

```powershell
python scripts\run_smoke_tests.py
```

已验证输出位于 `outputs\smoke`。smoke test 使用合成小数据，只用于确认代码可运行，不代表真实数据集性能。

## 数据目录

真实数据按下面位置放置：

```text
datasets/
  HWDB1/
    train.txt              # 每行: relative/or/absolute/image_path label
    test.txt
    ...
  msrc2_seg/
    images/*.bmp
    gt/*_GT.bmp
    train.txt              # 可选；若不存在则读取 images 下全部 bmp
    val.txt                # 可选
  bsds500/
    trainval/*.jpg
    test/*.jpg
  jena_climate_2009_2016.csv
  shakespeare.txt
```

MNIST 与 Fashion MNIST 可通过 `--download` 自动下载到 `datasets`。

公开数据集准备：

```powershell
python scripts\prepare_real_datasets.py --all-public
```

该脚本会下载 Jena Climate、Tiny Shakespeare、BSDS500、MSRC-V2，并从公开 CASIA-HWDB PNG 镜像整理出课件要求的 10 类 HWDB1 兼容子集：`2382` 张训练图像、`601` 张测试图像。

查看真实数据状态：

```powershell
python scripts\prepare_real_datasets.py --status
```

运行三个实验的完整真实数据训练：

```powershell
python scripts\run_full_real_experiments.py --profile full
```

如只做验收流程验证但仍使用真实数据，可用质量验收配置：

```powershell
python scripts\run_full_real_experiments.py --profile acceptance
```

当前已经完成一次真实数据验收训练，并导出了模型与案例：

```text
outputs\full_real
  exp1\checkpoints\*.pt
  exp1\demo_cases\*.png
  exp2_seg\checkpoints\unet_msrc.pt
  exp2_sr\checkpoints\srcnn_bsds500.pt
  exp3_weather\checkpoints\weather_*.pt
  exp3_shakespeare\checkpoints\shakespeare_lstm.pt
```

重新审计当前全部正式 checkpoint：

```powershell
python scripts\audit_model_quality.py
```

审计输出位于 `outputs\model_audit\audit_summary.md`。

## 实验 1：图像分类

```powershell
python -m dl_labs.exp1_classification.classification `
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

结果：

- `outputs\exp1\classification_results.csv`
- `outputs\exp1\classification_results.json`

## 实验 2：语义分割与超分

语义分割：

```powershell
python -m dl_labs.exp2_vision.segmentation_sr segmentation `
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
python -m dl_labs.exp2_vision.segmentation_sr super-resolution `
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

结果：

- `outputs\exp2_seg\segmentation_results.csv`
- `outputs\exp2_sr\super_resolution_results.csv`

## 实验 3：循环神经网络

气温预测：

```powershell
python -m dl_labs.exp3_sequence.sequence weather `
  --dataset jena `
  --csv-path datasets\jena_climate_2009_2016.csv `
  --models rnn gru lstm `
  --input-width 168 `
  --horizon 168 `
  --epochs 20 `
  --batch-size 64 `
  --output-dir outputs\exp3_weather
```

莎士比亚剧本生成：

```powershell
python -m dl_labs.exp3_sequence.sequence shakespeare `
  --dataset shakespeare `
  --text-file datasets\shakespeare.txt `
  --rnn-type lstm `
  --seq-len 100 `
  --epochs 20 `
  --seed-text "ROMEO:" `
  --generate-length 1000 `
  --output-dir outputs\exp3_shakespeare
```

结果：

- `outputs\exp3_weather\weather_results.csv`
- `outputs\exp3_shakespeare\shakespeare_generated.txt`
