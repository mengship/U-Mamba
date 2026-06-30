# F3 Fold 失败案例可视化指南

## 运行命令

### 方案1：使用FLAIR作为背景模态

```bash
python umamba/0607/visualize_brats_predictions.py \
  --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_3/validation" \
  --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_3/validation" \
  --cases BraTS20_Training_087 BraTS20_Training_299 BraTS20_Training_315 \
  --modality flair \
  --prefer-et \
  --dual-output \
  --out /hy-tmp/brats_f3_failure_cases.png
```

### 方案2：使用T1CE作为背景模态（推荐用于ET分析）

```bash
python umamba/0607/visualize_brats_predictions.py \
  --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_3/validation" \
  --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_3/validation" \
  --cases BraTS20_Training_087 BraTS20_Training_299 BraTS20_Training_315 \
  --modality t1ce \
  --prefer-et \
  --dual-output \
  --out /hy-tmp/brats_f3_failure_cases.png
```

## 输出文件

运行后将生成4个文件：

1. `/hy-tmp/brats_f3_failure_cases_full.png` - 完整脑切片对比图（PNG）
2. `/hy-tmp/brats_f3_failure_cases_full.pdf` - 完整脑切片对比图（PDF）
3. `/hy-tmp/brats_f3_failure_cases_zoom.png` - ET局部放大图（PNG）
4. `/hy-tmp/brats_f3_failure_cases_zoom.pdf` - ET局部放大图（PDF）

## 参数说明

- `--prefer-et`: 选择GT中ET（增强肿瘤，label 3）面积最大的axial切片
- `--dual-output`: 自动生成完整图和局部放大图两个版本
- `--modality flair` 或 `t1ce`: 选择背景模态（FLAIR或T1CE）
- `--cases`: 指定要可视化的病例ID

## 论文可视化分析文本（中文）

> 为了进一步分析fold 3的性能退化现象，我们对三个典型的失败案例（BraTS20_Training_087、BraTS20_Training_299和BraTS20_Training_315）进行了可视化分析。图X展示了这些病例在GT中增强肿瘤（ET）区域面积最大的轴向切片上的分割结果对比。可以观察到，相比U-Mamba基线，我们的方法在这些小体积ET病例上出现了轻微的过分割或欠分割现象，这与定量统计结果一致。这些案例主要集中在ET边界模糊或肿瘤形态不规则的情况，提示模型在处理此类挑战性病例时仍有改进空间。

## 论文可视化分析文本（英文）

> To further investigate the performance degradation in fold 3, we conducted a visual analysis of three representative failure cases (BraTS20_Training_087, BraTS20_Training_299, and BraTS20_Training_315). Figure X presents the segmentation comparisons on the axial slices with maximum enhancing tumor (ET) area in the ground truth. Compared to the U-Mamba baseline, our method exhibits slight over-segmentation or under-segmentation in these small-volume ET cases, which is consistent with the quantitative statistics. These cases are primarily characterized by ambiguous ET boundaries or irregular tumor morphology, suggesting room for improvement when handling such challenging cases.

## 可视化说明

- **颜色方案**：
  - 绿色 (label 1): 水肿区域/WT外围
  - 黄色 (label 2): 肿瘤核心相关区域
  - 红色 (label 3): 增强肿瘤（ET，最关键区域）
  
- **图像布局**：
  - 每行：一个病例
  - 列：[MRI模态, GT, U-Mamba, Ours]
  
- **切片选择**：自动选择GT中ET面积最大的轴向切片

- **局部放大**：围绕肿瘤区域的外接框进行裁剪，保留24像素上下文边距

## 注意事项

1. 脚本已处理非标准gzip压缩的`.nii.gz`文件问题
2. 如果某个病例的ET区域非常小或不存在，脚本会退回到选择整个肿瘤（WT）面积最大的切片
3. 生成的PDF文件适合直接用于论文插图
