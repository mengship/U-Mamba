# Claude 可视化分割图生成提示词

请你在当前 U-Mamba 项目中，帮我生成 BraTS2020 脑肿瘤分割结果的论文可视化图。注意：本任务只做可视化，不训练模型，不修改网络结构，不重新预测。

## 背景

我正在写一篇关于 U-Mamba 改进方法的小论文。当前最终方法为：

`nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs`

需要将其与基线模型进行可视化对比。主要对比对象包括：

1. `nnUNetTrainer`
2. `nnUNetTrainerUMambaEnc_150epochs`
3. `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs`

如果某个模型的预测目录不存在，请跳过该模型，并在输出中说明。

## 数据路径

请优先使用以下路径：

```bash
IMAGE_DIR=/hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr
GT_DIR=/hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations
RESULTS_ROOT=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020
```

预测结果路径格式通常为：

```bash
$RESULTS_ROOT/TRAINER__nnUNetPlans__3d_fullres/fold_0/validation
```

例如：

```bash
/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_0/validation
/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0/validation
```

## 可视化目标

请生成一张适合放入中文小论文的分割结果对比图，推荐布局如下：

```text
Case | FLAIR | Ground Truth | nnU-Net | U-Mamba | Ours
```

或者如果 `nnU-Net` 预测结果不存在，则使用：

```text
Case | FLAIR | Ground Truth | U-Mamba | Ours
```

要求：

1. 每一行展示一个病例。
2. 建议选择 3 个病例，最多不要超过 4 个病例。
3. 每个病例选择肿瘤区域面积最大的 axial slice。
4. 底图优先使用 FLAIR 模态，即 nnU-Net 格式下的 `_0003.nii.gz`。
5. 将 Ground Truth 和预测结果以半透明彩色 mask 叠加在 FLAIR 图像上。
6. 需要保留清晰的列标题和病例名称。
7. 输出高分辨率图片，建议 `dpi=300`。
8. 输出文件建议保存为：

```bash
/hy-tmp/brats_visualization_fold0.png
/hy-tmp/brats_visualization_fold0.pdf
```

## 标签约定

当前数据已经转换为 nnU-Net BraTS 标签格式：

```text
0: background
1: edema / WT outer region
2: tumor core related region
3: enhancing tumor
```

可视化颜色建议：

```text
label 1: green
label 2: yellow
label 3: red
```

如果发现标签仍是 BraTS 原始格式 `0/1/2/4`，请先转换为 nnU-Net 格式再画图：

```text
BraTS 2 -> nnU-Net 1
BraTS 1 -> nnU-Net 2
BraTS 4 -> nnU-Net 3
```

## 选病例原则

请不要只随机选病例。请优先选择更适合论文展示的病例：

1. 肿瘤区域不能太小，图像缩小后仍能看清。
2. Ground Truth 中 WT、TC、ET 至少有两个区域较明显。
3. U-Mamba 与 Ours 之间最好能看出边界或小区域差异。
4. 避免选择预测全对或全错的极端病例。

可以先自动扫描 fold_0 validation 中共有病例，计算每个病例 GT 的肿瘤体素数，优先从肿瘤区域较明显的病例中挑选 3 个。也可以额外根据 HD95 或 Dice 差异挑选 Ours 表现更稳定的病例。

## 实现要求

请生成或修改一个独立脚本，例如：

```bash
umamba/0607/visualize_brats_predictions.py
```

脚本需要支持以下参数：

```bash
--image-dir
--gt-dir
--pred NAME=DIR
--cases
--num-cases
--modality
--axis
--out
```

脚本运行示例：

```bash
python umamba/0607/visualize_brats_predictions.py \
  --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred "nnU-Net=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
  --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --num-cases 3 \
  --out /hy-tmp/brats_visualization_fold0.png
```

如果 `nnUNetTrainer` 预测目录不存在，请自动跳过这一列，并改用：

```bash
python umamba/0607/visualize_brats_predictions.py \
  --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --num-cases 3 \
  --out /hy-tmp/brats_visualization_fold0.png
```

## 画图风格要求

1. 不要使用太花哨的背景和装饰。
2. 图像之间留白要小，整体紧凑。
3. 列标题使用英文或中英文均可，推荐：

```text
FLAIR, GT, nnU-Net, U-Mamba, Ours
```

4. 如果使用图例，放在图下方，说明：

```text
Green: edema / WT outer region
Yellow: tumor core
Red: enhancing tumor
```

5. 图像导出后，请检查是否存在以下问题：

```text
- 图像是否过暗
- mask 是否和底图错位
- 每列标题是否清晰
- 病例名是否太长影响排版
- 输出 png 是否能正常打开
```

## 最终输出

请完成以下内容：

1. 生成可视化脚本。
2. 运行脚本生成图片。
3. 输出实际使用的病例 ID 和 slice 编号。
4. 输出图片保存路径。
5. 如果某个模型预测路径不存在，请说明缺失路径，不要中断全部流程。
6. 给出一段可直接写入论文“可视化分析”小节的文字。

论文文字风格示例：

```text
为直观比较不同模型的脑肿瘤分割效果，本文选取 BraTS2020 验证集中 3 个代表性病例进行可视化分析，结果如图 X 所示。可以看出，U-Mamba 能够较好地捕获肿瘤整体区域，但在局部边界和小体积增强区域处仍存在轮廓不连续或轻微漏分现象。相比之下，本文方法生成的分割结果在肿瘤边缘处与真实标签更加接近，尤其在肿瘤核心区和增强肿瘤区的局部结构上保持了更好的连续性。该现象与 HD95 指标下降的定量结果一致，说明本文方法有助于降低边界定位误差并提升脑肿瘤精细分割效果。
```
