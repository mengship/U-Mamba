# U-Mamba 恒源云部署流程

下面这套流程面向恒源云的 GPU 训练环境，目标是把 U-Mamba 从代码、环境、数据、预处理到训练跑通。

## 1. 推荐目录规划

建议把代码、数据和结果分开挂载，避免实例重启后丢失训练产物：

```bash
/mnt/work/u-mamba/code
/mnt/work/u-mamba/data
/mnt/work/u-mamba/results
/mnt/work/u-mamba/logs
```

其中：
- `code` 放仓库代码
- `data` 放 `nnUNet_raw` 和 `nnUNet_preprocessed`
- `results` 放训练输出和模型文件
- `logs` 放终端日志或任务日志

## 2. 环境准备

默认以 Ubuntu 20.04 / 22.04、CUDA 11.8、单机多卡为前提。

```bash
conda create -n umamba python=3.10 -y
conda activate umamba
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d>=1.2.0 mamba-ssm --no-cache-dir
cd /mnt/work/u-mamba/code/U-Mamba/umamba
pip install -e .
```

如果恒源云镜像已经预装了兼容版本的 PyTorch，也可以只保留 `mamba-ssm` 和 `pip install -e .`。

## 3. 设置数据路径

U-Mamba 现在支持通过环境变量指定数据路径。建议把它们写进启动脚本或 `~/.bashrc`：

```bash
export nnUNet_raw=/mnt/work/u-mamba/data/nnUNet_raw
export nnUNet_preprocessed=/mnt/work/u-mamba/data/nnUNet_preprocessed
export nnUNet_results=/mnt/work/u-mamba/results
```

如果不设置，程序会回退到仓库里的 `data/` 目录。

## 4. 准备数据

把数据整理成 nnU-Net 结构，目录名使用仓库里已经约定的 Dataset 编号，例如：

```bash
/mnt/work/u-mamba/data/nnUNet_raw/Dataset701_AbdomenCT/
/mnt/work/u-mamba/data/nnUNet_raw/Dataset702_AbdomenMR/
/mnt/work/u-mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/
/mnt/work/u-mamba/data/nnUNet_raw/Dataset704_Endovis17/
```

每个数据集下至少要有：
- `imagesTr/`
- `labelsTr/`
- `dataset.json`

如果你是自己的数据集，先按 nnU-Net 的格式转换后再放入 `nnUNet_raw`。

### BraTS2018 的处理方式

BraTS2018 是 4 模态脑肿瘤 MRI 数据，通常每个病例都包含：

- `t1.nii.gz`
- `t1ce.nii.gz`
- `t2.nii.gz`
- `flair.nii.gz`
- `seg.nii.gz`

如果你拿到的是官方 BraTS2018 原始数据，常见目录结构会是按 `HGG/` 和 `LGG/` 分类的病例文件夹。处理时要做两件事：

1. 把四个模态重命名并拷贝到 nnU-Net 的 `imagesTr/`：
	- `case_t1.nii.gz` -> `case_0000.nii.gz`
	- `case_t1ce.nii.gz` -> `case_0001.nii.gz`
	- `case_t2.nii.gz` -> `case_0002.nii.gz`
	- `case_flair.nii.gz` -> `case_0003.nii.gz`
2. 把分割标签从 BraTS 的原始标注映射成连续标签：
	- `0 -> 0`
	- `1 -> 2`
	- `2 -> 1`
	- `4 -> 3`

这样生成的 `labelsTr/case.nii.gz` 才符合 nnU-Net / U-Mamba 的训练要求。

如果你想直接复用仓库里的转换逻辑，可以参考 [Dataset137_BraTS21.py](umamba/nnunetv2/dataset_conversion/Dataset137_BraTS21.py) 的写法，把数据源目录改成你的 BraTS2018 路径，再把输出 dataset 名称改成 `BraTS2018` 或你自己的任务名即可。

转换完成后，目标目录一般会长这样：

```bash
export nnUNet_raw=/mnt/work/u-mamba/data/nnUNet_raw

nnUNet_raw/
└── DatasetXXX_BraTS2018/
	 ├── imagesTr/
	 ├── labelsTr/
	 └── dataset.json
```

其中 `dataset.json` 要声明 4 个输入通道，标签建议写成：`background`、`whole tumor`、`tumor core`、`enhancing tumor`。

## 5. 预处理

先做 fingerprint 和 preprocessing，这一步会生成 plans 和预处理数据：

```bash
nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity
```

如果是自己的数据集，把 `701` 换成对应的 dataset id。

## 6. 开始训练

### 单卡训练

推荐先用 AMP 关闭版，稳定性更好：

```bash
nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP
```

如果想训练 2D：

```bash
nnUNetv2_train 701 2d all -tr nnUNetTrainerUMambaBot
```

### 多卡训练

训练脚本支持多 GPU，使用 `-num_gpus` 指定卡数，并用 `CUDA_VISIBLE_DEVICES` 控制实际使用的设备：

```bash
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP -num_gpus 2
```

### 断点续训

如果任务中断，可以继续上次训练：

```bash
nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP --c
```

### 只做验证

训练完成后只跑验证：

```bash
nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP --val
```

如果想用最佳权重验证：

```bash
nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP --val_best
```

## 7. 推理

训练完成后，使用测试集或待预测数据做推理：

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 701 -c 3d_fullres -f all -tr nnUNetTrainerUMambaEncNoAMP --disable_tta
```

## 8. 恒源云上建议的运行方式

建议把完整流程写成一个启动脚本，避免手动输入出错。下面是一个最小可用模板：

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate umamba

export nnUNet_raw=/mnt/work/u-mamba/data/nnUNet_raw
export nnUNet_preprocessed=/mnt/work/u-mamba/data/nnUNet_preprocessed
export nnUNet_results=/mnt/work/u-mamba/results
export CUDA_VISIBLE_DEVICES=0,1

cd /mnt/work/u-mamba/code/U-Mamba/umamba

nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity
nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerUMambaEncNoAMP -num_gpus 2
```

如果恒源云支持任务守护或后台运行，优先把日志重定向到 `logs/`，并保留 checkpoint 到独立磁盘。

## 9. 最小排错清单

- `mamba_ssm` 导入失败：检查 PyTorch 和 CUDA 版本是否匹配
- 找不到数据：检查 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 是否都指向正确路径
- 训练 NaN：优先切到 `nnUNetTrainerUMambaEncNoAMP`
- 预处理失败：先确认 `dataset.json` 和文件命名是否符合 nnU-Net 规范

如果你要，我可以继续把这套流程再整理成一份“恒源云一键启动脚本 + 提交任务模板”。
