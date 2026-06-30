# C类补实验与统计计划

本文档对应`审稿意见整改矩阵.md`中的C1-C8，目标是把“需要补实验或统计”的意见拆成可执行任务。当前主方法仍为：

`nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs`

## 一、当前已有结果

已完成五折：

| 方法 | trainer | 已有fold | 用途 |
| ---- | ------- | -------- | ---- |
| U-Mamba基线 | `nnUNetTrainerUMambaEnc_150epochs` | f0-f4 | 主横向基线、显著性检验基线 |
| 本文方法 | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs` | f0-f4 | 最终方法 |

已完成f0消融：

| 消融项 | trainer | 已有fold | 说明 |
| ------ | ------- | -------- | ---- |
| Encoder ETSM only | `nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs` | f0 | 解码器不使用ETSM |
| Stage-aware decoder | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs` | f0 | 解码器低分辨率阶段使用ETSM，不启用SkipCalibration |
| Stage-aware decoder + SkipCalibration | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs` | f0-f4 | 当前主方法 |
| Full model | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full_150epochs` | f0 | 含BoundaryAttention，效果不作为主线 |

## 二、C1：五折消融实验

审稿意见：消融实验只有f0，缺少五折平均，统计可靠性不足。

优先级：高。

建议做法：先补齐“主线消融”的五折，即U-Mamba、EncoderOnly、StageAwareDecoder、StageAwareDecoder+SkipCalibration。BoundaryAttention因f0效果较差，不进入主线消融；如果导师要求，可作为附录或补充说明。

需要补跑：

```bash
for FOLD in 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs
done
```

跑完后收集Dice：

```bash
python umamba/0607/collect_rthd_results.py \
  --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --folds 0 1 2 3 4 \
  --trainers \
    nnUNetTrainerUMambaEnc_150epochs \
    nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
  --csv /hy-tmp/c1_five_fold_ablation_dice.csv
```

HD95需要对每个trainer/fold分别计算，推荐输出命名：

`/hy-tmp/hd95_json/{TRAINER}_fold{FOLD}_hd95.json`

```bash
python umamba/0607/compute_brats_hd95.py \
  --pred-dir /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/TRAINER__nnUNetPlans__3d_fullres/fold_FOLD/validation \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --json /hy-tmp/hd95_json/TRAINER_foldFOLD_hd95.json \
  --csv /hy-tmp/hd95_csv/TRAINER_foldFOLD_hd95.csv
```

论文中使用：将原“f0消融表”改为“五折均值±标准差消融表”；如果只来得及补部分折，必须在表题中标明“已完成fold”。

## 三、C2：阶段感知解码对照

审稿意见：缺少全解码层ETSM、仅低分辨率ETSM、无解码ETSM的对照。

优先级：高，但需要先评估显存。

已具备训练器：

| 对照 | trainer | 状态 |
| ---- | ------- | ---- |
| 无解码ETSM | `nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs` | 已有f0 |
| 仅低分辨率解码ETSM | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs` | 已有f0 |
| 全解码层ETSM | `nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs` | 已新增，待跑 |

先跑f0筛查：

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
```

如果显存和结果正常，再补f1-f4：

```bash
for FOLD in 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
done
```

论文中使用：单独形成“解码ETSM部署位置消融表”，比较Dice、HD95、参数量、推理时间和峰值显存。重点说明若全解码层ETSM未带来更好Dice/HD95或显存更高，则支持阶段感知部署策略。

## 四、C3：跨视图交互门控消融

审稿意见：ETSM内部`cross-view gating`缺少独立量化验证。

优先级：中高。

新增训练器：

`nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs`

该对照保持Encoder ETSM、StageAwareDecoder和SkipCalibration不变，仅关闭编码器与解码器ETSM中的`cross_view_interaction`。

先跑f0筛查：

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs
```

若f0差异明显或导师要求补足统计，再跑f1-f4：

```bash
for FOLD in 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs
done
```

论文中使用：作为ETSM内部消融，比较`w/o cross-view gating`和本文方法。若差异很小，需要如实说明跨视图门控贡献有限，主要收益来自三视图降维与SkipCalibration。

## 五、C4：传统Attention U-Net门控对照

审稿意见：跳跃标定未与经典0-1注意力门控对照，无法证明本文[-1,1]残差双向门控更合适。

优先级：中。

新增训练器：

`nnUNetTrainerUMambaEncRTHD_AttentionSkipGate_150epochs`

该对照保持ETSM与阶段感知解码策略不变，仅将SkipCalibration替换为Attention U-Net风格0-1单向注意力门控。

先跑f0筛查：

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_AttentionSkipGate_150epochs
```

若结果接近或优于本文方法，再补五折；若明显较差，可作为f0机制对照，是否扩展五折由导师决定。

论文中使用：作为“跳跃连接标定机制对照”。如果本文门控在HD95上更优，可强调残差双向调节避免传统0-1门控过度抑制细节；如果不优，应调整表述，只保留为轻量替代方案。

## 六、C5：统一环境横向基线

审稿意见：nnU-Net、SegMamba等横向对比需要统一环境复现。

优先级：高，但成本较大。

建议分两级处理：

1. 必做：补跑`nnUNetTrainer`五折，作为最公平的CNN强基线。
2. 尽量做：补跑SegMamba。如果本仓库没有SegMamba实现，需要单独引入官方代码或说明“由于框架和实现差异，文献结果仅作参考”。

nnU-Net命令：

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainer
done
```

收集结果时加入：

```bash
python umamba/0607/collect_rthd_results.py \
  --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --folds 0 1 2 3 4 \
  --trainers \
    nnUNetTrainer \
    nnUNetTrainerUMambaEnc_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
  --csv /hy-tmp/c5_fair_baselines_dice.csv
```

论文中使用：横向对比表分成“统一环境复现结果”和“文献参考结果”两组，不把不同来源结果混在一起直接排名。

## 七、C6：f3折退化病例统计

审稿意见：f3折退化不能只写“存在波动”，需要统计病例特征。

优先级：高，可先用已有f3预测结果完成。

推荐统计内容：

- f3每个病例WT/TC/ET的GT体素量。
- U-Mamba与本文方法每病例Dice差值、HD95差值。
- ET体积较小、ET预测为空、HD95异常大的病例。
- 本文方法明显退化的top病例，用于可视化或局部放大。

统计脚本：

```bash
python umamba/0607/analyze_c_revisions.py \
  --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --trainers \
    nnUNetTrainerUMambaEnc_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
  --folds 3 \
  --hd95-root /hy-tmp/hd95_json \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --csv /hy-tmp/c6_f3_case_stats.csv \
  --json /hy-tmp/c6_f3_case_stats_summary.json
```

注意：`--pred-root`支持nnU-Net默认目录`{trainer}__nnUNetPlans__3d_fullres/fold_N/validation`，也兼容简化目录`{trainer}/fold_N/validation`。

论文中使用：在五折结果分析后增加一段“f3退化病例分析”，避免只做主观猜测。

## 八、C7：统计显著性检验

审稿意见：五折提升是否稳定，建议配对显著性检验。

优先级：中，依赖每病例Dice/HD95。

已有脚本可做配对t-test和Wilcoxon检验：

```bash
python umamba/0607/analyze_c_revisions.py \
  --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --trainers \
    nnUNetTrainerUMambaEnc_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
  --folds 0 1 2 3 4 \
  --hd95-root /hy-tmp/hd95_json \
  --csv /hy-tmp/c7_case_level_metrics.csv \
  --json /hy-tmp/c7_significance_summary.json
```

论文中使用：如果p值显著，可写“配对检验显示HD95改善具有统计意义”；如果不显著，应改为“整体趋势改善，但不同折间仍存在波动”。

## 九、C8：表格均值与标准差核对

审稿意见：平均值和标准差需要核对。

优先级：中。

使用同一个脚本：

```bash
python umamba/0607/analyze_c_revisions.py \
  --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
  --trainers \
    nnUNetTrainerUMambaEnc_150epochs \
    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
  --folds 0 1 2 3 4 \
  --hd95-root /hy-tmp/hd95_json \
  --csv /hy-tmp/c8_case_level_metrics.csv \
  --json /hy-tmp/c8_mean_std_summary.json
```

论文中使用：将五折表格从单一均值扩展为`mean±std`；若版面有限，主表保留均值，在正文中补充标准差范围。

## 十、建议执行顺序

1. 先完成C6、C7、C8统计：不需要重新训练，能最快补强论文分析。
2. 跑C2的`FullDecoderETSM` f0，确认显存和结果是否可用。
3. 跑C3的`NoCrossViewGate` f0和C4的`AttentionSkipGate` f0，先判断是否值得扩展五折。
4. 补C1的EncoderOnly、StageAwareDecoder f1-f4，形成五折主线消融。
5. 补C5的nnU-Net五折；SegMamba视代码迁移成本和导师要求决定。
6. 根据新结果更新论文表格、结果分析和`审稿意见回复说明.md`。
