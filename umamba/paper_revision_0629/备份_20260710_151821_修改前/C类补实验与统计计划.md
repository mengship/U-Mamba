# C类补实验与统计计划

本文档对应`审稿意见整改矩阵.md`中的C1-C8，目标是把“需要补实验或统计”的意见拆成可执行任务。当前主方法仍为：

`nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs`

## 一、当前已有结果

已完成五折：

| 方法 | trainer | 已有fold | 用途 |
| ---- | ------- | -------- | ---- |
| U-Mamba基线 | `nnUNetTrainerUMambaEnc_150epochs` | f0-f4 | 主横向基线、显著性检验基线 |
| Encoder ETSM only | `nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs` | f0-f4 | C1主线消融 |
| Stage-aware decoder | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs` | f0-f4 | C1主线消融 |
| 本文方法 | `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs` | f0-f4 | 最终方法 |

已完成f0消融：

| 消融项 | trainer | 已有fold | 说明 |
| ------ | ------- | -------- | ---- |
| Partial decoder ETSM | `nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs` | f0 | 解码器stage 0/1/2使用ETSM，作为阶段范围筛查 |
| No cross-view gate | `nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs` | f0 | 关闭跨视图交互门控，结果反向，待决策 |
| Attention Skip Gate | `nnUNetTrainerUMambaEncRTHD_AttentionSkipGate_150epochs` | f0 | Attention U-Net式0-1门控机制对照 |
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
| 部分解码层ETSM | `nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs` | f0已完成，Mean Dice=88.73% |
| 全解码层ETSM | `nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs` | f0筛查OOM，24GB显存不可行 |

先跑f0筛查：

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
```

f0筛查结果：

- 运行命令：`nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs`
- 报错位置：`UMambaEnc_RTHD.py`解码器阶段，`rthd_modules.py`中残差相加前后显存继续增长。
- 显存情况：RTX3090 24GB上训练进程已占用约23.19GB，仅剩约502MB，继续分配512MB时触发`torch.cuda.OutOfMemoryError`。
- 结论：全解码层ETSM在当前128×128×128 patch和24GB显存条件下不可训练，不建议继续补f1-f4。
- 论文中可用表述：全解码层引入ETSM会在高分辨率解码阶段带来过高显存开销，验证了阶段感知解码仅在低分辨率阶段部署ETSM的必要性。

若后续导师坚持测试全解码层ETSM，需要更高显存GPU、降低patch size或让Claude重写轻量版本。当前小论文主线不建议为该不可行对照继续投入训练资源。

如果显存和结果正常，再补f1-f4（当前已判定不执行）：

```bash
for FOLD in 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
done
```

论文中使用：不再将全解码层ETSM作为完整数值消融表项，可在阶段感知解码分析中补充“全解码层ETSM在24GB GPU上触发OOM，说明高分辨率解码阶段引入状态空间模块具有明显显存开销”，用于支持当前低分辨率阶段部署策略。

补充可训练对照：

- `nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs`为较温和的部分解码层ETSM版本，在解码器stage 0/1/2使用ETSM，而当前`StageAwareDecoder`仅在stage 0/1使用ETSM。
- f0训练结果：`Mean Validation Dice=0.8872545937003561`，即约88.73%。
- f0 HD95结果：WT=5.028944，TC=4.270064，ET=3.447150，Mean=4.248719；ET统计中finite=71、nan=1、inf=2，与U-Mamba和最终方法的ET计数口径一致。
- 与已有f0结果对比：
  - `+ETSM+Stage`：88.47%。
  - `PartialDecoderETSM`：88.73%，Mean HD95=4.249。
  - `+ETSM+Stage+Skip`（最终方法）：88.86%，Mean HD95=3.444。
- 当前结论：增加一个中等分辨率解码ETSM阶段后，Mean Dice较`+ETSM+Stage`有所提升，Mean HD95也由4.308下降至4.249，但仍明显低于最终方法的HD95表现；说明扩大解码ETSM部署范围存在一定收益，但SkipCalibration对边界误差改善仍有更关键的补充作用。
- 待补：若要写入论文表格，需要进一步收集该trainer的WT/TC/ET Dice、峰值显存和推理时间，避免只用Mean Dice和HD95下结论。

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

f0筛查结果：

- 训练已完成，`Mean Validation Dice=0.8889759482835508`，即约88.90%。
- 与最终方法f0的88.86%相比，NoCrossViewGate的Mean Dice略高，说明不能简单声称`cross-view gating`带来Dice提升。
- f0 HD95结果：WT=3.925076，TC=2.787099，ET=1.662399，Mean=2.791524；ET统计中finite=71、nan=1、inf=2，与最终方法计数口径一致。
- 与最终方法f0的Mean HD95=3.444相比，NoCrossViewGate的HD95更低，说明在f0上关闭跨视图交互门控反而取得更好的边界距离表现。
- 当前判断：`cross-view gating`在当前实现中不能作为稳定增益模块来强调。至少在f0上，它可能引入额外扰动或过度调节，导致边界误差变大。论文中应谨慎处理该模块：若保留当前最终方法，需要弱化“跨视图门控贡献”的表述；若将NoCrossViewGate作为新候选方法，则需要进一步补五折验证。
- 待补：需要计算该trainer的WT/TC/ET Dice、峰值显存和推理时间；若准备将其替换为最终方法，还需补f1-f4和显著性统计。

若f0差异明显或导师要求补足统计，再跑f1-f4：

```bash
for FOLD in 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs
done
```

论文中使用：作为ETSM内部消融，比较`w/o cross-view gating`和本文方法。当前f0结果显示关闭cross-view gating后Dice和HD95均更优，因此不应再将cross-view gating表述为明确贡献点。若后续不扩展五折，可在论文中将其作为“交互门控在当前实现下未带来稳定收益”的负向消融结果；若要调整最终方法，则需补NoCrossViewGate五折。

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

f0筛查结果：

- 训练命令：`nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_AttentionSkipGate_150epochs`
- Mean Validation Dice=0.8845835022375329，即约88.46%。
- HD95结果：WT=5.627576，TC=4.599872，ET=3.596005，Mean=4.607818；ET统计中finite=71、nan=1、inf=2。
- 与最终方法f0对比：最终方法Mean Dice=88.86%，Mean HD95=3.444；Attention U-Net式0-1门控在Dice和HD95上均较差。
- 当前结论：传统0-1单向门控可能对浅层有效细节产生过度抑制；本文残差式双向语义标定在第0折机制对照中表现更优。该结果已写入修订稿消融实验段落。

## 六、C5：统一环境横向基线

审稿意见：nnU-Net、SegMamba等横向对比需要统一环境复现。

优先级：高，但成本较大。

建议分两级处理：

1. 必做：补跑`nnUNetTrainer`五折，作为最公平的CNN强基线。
2. 尽量做：补跑SegMamba。SegMamba采用官方公开实现，因其不是nnU-Net trainer，复现时对齐BraTS2020五折划分、训练轮数和Dice/HD95统计口径，而不强行声称完全相同训练框架。

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

当前进展：

- nnU-Net：`nnUNetTrainer_150epochs`五折已完成Dice和HD95统计。五折Mean Dice为85.49±2.23%，Mean HD95为4.042±0.463；各区域Dice为WT=91.28±0.79%、TC=87.23±1.69%、ET=77.98±4.44%，各区域HD95为WT=4.456±0.467、TC=4.252±0.471、ET=3.419±0.822。
- nnU-Net五折逐折结果：f0 Dice=88.43%、HD95=4.313；f1 Dice=86.06%、HD95=3.716；f2 Dice=86.24%、HD95=3.707；f3 Dice=82.57%、HD95=4.732；f4 Dice=84.16%、HD95=3.742。
- 与本文方法五折均值对比：本文方法Mean Dice=85.53±2.46%，Mean HD95=4.077±1.055；nnU-Net 150epochs的Mean Dice略低于本文方法，但Mean HD95略低于本文方法。因此正式论文中应表述为本文方法与nnU-Net强基线性能接近，在Dice上略高，但不宜声称在HD95上全面优于nnU-Net。
- SegMamba：已下载官方代码并完成环境验证，`python 0_inference.py`可正常前向输出`torch.Size([1, 4, 128, 128, 128])`。
- SegMamba已完成fold0-f4的150 epochs训练、预测和统一口径评估。预测NIfTI初始保存为`(155,240,240)`，而GT为`(240,240,155)`；经单病例测试，正确轴变换为`np.transpose(pred, (2, 1, 0))`，各fold修正后保存到`prediction_results/segmamba_brats2020_fold*_fixed`。
- SegMamba fold0 Dice：WT=91.99%，TC=88.43%，ET=82.75%，Mean=87.72%；HD95：WT=4.677，TC=4.815，ET=3.275，Mean=4.256。
- SegMamba fold1 Dice：WT=91.42%，TC=85.78%，ET=78.03%，Mean=85.08%；HD95：WT=3.413，TC=5.106，ET=3.475，Mean=3.998。
- SegMamba fold2 Dice：WT=91.67%，TC=83.99%，ET=76.82%，Mean=84.16%；HD95：WT=5.790，TC=5.578，ET=4.355，Mean=5.241。
- SegMamba fold3 Dice：WT=90.85%，TC=85.19%，ET=70.45%，Mean=82.17%；HD95：WT=5.062，TC=5.975，ET=6.765，Mean=5.934。
- SegMamba fold4 Dice：WT=90.20%，TC=82.84%，ET=73.80%，Mean=82.28%；HD95：WT=4.443，TC=5.211，ET=3.721，Mean=4.458。fold4验证集为73例，符合369例五折划分。
- SegMamba五折结果：Dice WT=91.23±0.71%、TC=85.25±2.11%、ET=76.37±4.62%、Mean=84.28±2.29%；HD95 WT=4.677±0.872、TC=5.337±0.449、ET=4.318±1.427、Mean=4.777±0.796。
- 与本文方法五折均值对比：本文方法Mean Dice=85.53±2.46%，Mean HD95=4.077±1.055；SegMamba Mean Dice=84.28±2.29%，Mean HD95=4.777±0.796。正式论文中可表述为本文方法相较SegMamba取得更高Dice和更低HD95，但仍需说明SegMamba采用官方独立实现，公平性主要体现在相同BraTS2020五折划分、训练轮数和评价口径。

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

## 十、当前完成情况与后续顺序

1. C1、C5、C6、C7和C8已完成，并已将关键结果写入修订稿和审稿回复说明。
2. C2已完成第0折资源可行性筛查：全解码层ETSM触发显存不足，PartialDecoderETSM仅作为阶段感知策略的补充证据记录。
3. C3已完成第0折反向筛查：`NoCrossViewGate`第0折结果优于当前最终方法，后续需由导师决定是否扩展f1-f4，或仅作为负向筛查结果保留。
4. C4已完成第0折机制对照，并已写入修订稿消融分析段落。
5. 后续优先级转为论文收口：数值一致性终检、图表格式统一、参考文献格式核对和正式审稿回复整理。

## 十一、后续处理TODO

当前依据：
- `审稿意见整改矩阵.md`中的C1-C8
- `C类补实验与统计计划.md`
- 已生成的`c6_f3_case_stats.csv`
- 已生成的`c6_f3_case_stats_summary.json`

当前结论：
- C6已得到f3逐病例Dice、HD95和体积统计。
- f3中Ours相较U-Mamba平均Dice下降约0.61个百分点，按WT、TC和ET区域均值计算的Mean HD95升高约1.08。
- f3退化主要不是整体普遍退化，而是少数极小ET病例和异常病例拉低均值，尤其`BraTS20_Training_087.nii.gz`。
- 已将C6统计结论整理为修订稿表4，用于说明f3折退化主要集中在`0<ET<1000`的小体积ET病例。

## 一、优先处理C6：f3折退化分析

- [x] 确认HD95结果文件是否已经存在并被统计脚本读入。
  - 需要至少包含两个文件：
    - `nnUNetTrainerUMambaEnc_150epochs_fold3_hd95.json`
    - `nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs_fold3_hd95.json`
  - 推荐放置目录：`umamba/paper_revision_0629/hd95_json/`或云端`/hy-tmp/hd95_json/`。

- [x] 如果HD95文件不存在，先重新计算f3的HD95。
  - U-Mamba：
    ```bash
    python umamba/0607/compute_brats_hd95.py \
      --pred-dir /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_3/validation \
      --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
      --json /hy-tmp/hd95_json/nnUNetTrainerUMambaEnc_150epochs_fold3_hd95.json \
      --csv /hy-tmp/hd95_csv/nnUNetTrainerUMambaEnc_150epochs_fold3_hd95.csv
    ```
  - Ours：
    ```bash
    python umamba/0607/compute_brats_hd95.py \
      --pred-dir /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_3/validation \
      --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
      --json /hy-tmp/hd95_json/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs_fold3_hd95.json \
      --csv /hy-tmp/hd95_csv/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs_fold3_hd95.csv
    ```

- [x] 重新运行C6统计，让HD95进入`c6_f3_case_stats.csv`。
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

- [x] 把云端更新后的两个C6结果文件下载/复制到：
  - `umamba/paper_revision_0629/c6_f3_case_stats.csv`
  - `umamba/paper_revision_0629/c6_f3_case_stats_summary.json`

- [x] 基于完整C6结果整理f3退化结论。
  - 必写点：
    - f3中Ours相较U-Mamba平均Dice下降约0.61个百分点。
    - f3中Ours相较U-Mamba平均HD95升高约1.08，边界定位在该折也存在退化。
    - 74例中Ours有42例优于U-Mamba、32例低于U-Mamba，说明不是普遍退化。
    - `0<ET<1000`体素的小ET病例退化最明显。
    - `BraTS20_Training_087.nii.gz`为关键异常样本，Ours几乎未预测出ET。
    - `BraTS20_Training_087.nii.gz`的mean HD95由9.24升至94.95，是f3 HD95均值升高的主要异常来源。

- [x] 将`BraTS20_Training_087.nii.gz`、`BraTS20_Training_299.nii.gz`、`BraTS20_Training_315.nii.gz`作为f3失败案例候选，用于可视化或局部放大图。
  - 目的：让小论文读者直观看到f3退化不是抽象数字，而是集中出现在小体积ET或预测漏检场景。
  - 建议图中至少包含：原始MRI切片、GT、U-Mamba预测、Ours预测。
  - 如果版面允许，可增加局部放大框，突出ET区域或边界漏分区域。
  - 这一步可和D类可视化合并执行，但在C6分析中需要明确引用该失败案例图。
  - 当前处理：失败案例候选已根据`c6_f3_case_stats.csv`确认，生成图片本身放到D类可视化任务中执行。

  | 病例 | GT ET体素数 | 选择原因 | U-Mamba Mean Dice/HD95 | Ours Mean Dice/HD95 | 可视化重点 |
  | ---- | ----------: | -------- | --------------------- | ------------------- | ---------- |
  | `BraTS20_Training_087.nii.gz` | 406 | 关键异常样本，Ours几乎未预测出ET | 0.5072 / 9.238 | 0.0000 / 94.947 | 小体积ET漏检、边界距离异常增大 |
  | `BraTS20_Training_299.nii.gz` | 659 | 小体积ET退化较明显，ET Dice下降 | 0.8115 / 6.098 | 0.7234 / 6.872 | ET区域漏分或响应不足 |
  | `BraTS20_Training_315.nii.gz` | 109 | 极小ET病例，ET Dice下降但HD95未恶化 | 0.8816 / 1.412 | 0.8420 / 1.244 | 小目标Dice敏感性与局部预测差异 |

  推荐图形方案：
  - 主图：每行一个病例，列为`FLAIR/T1ce`、`GT`、`U-Mamba`、`Ours`。
  - 局部放大：围绕ET区域或GT肿瘤外接框裁剪，突出红色ET区域。
  - 切片选择：优先选择GT中ET面积最大的axial slice；如果ET极小导致难以观察，可同时输出一版肿瘤整体最大slice和一版ET最大slice。
  - 论文用途：作为图5或图4的补充子图，用于解释f3中小体积ET病例造成的性能波动。

  Claude可视化任务提示词：

  ```text
  请在当前U-Mamba项目中补充f3失败案例局部可视化图。本任务只做可视化，不训练模型，不修改网络结构，不重新预测。

  背景：
  论文修订中已经通过C6统计发现，f3折性能退化主要集中在小体积ET病例。需要生成局部放大图，帮助解释BraTS20_Training_087、BraTS20_Training_299、BraTS20_Training_315三个病例中Ours相对U-Mamba的退化现象。

  使用路径：
  IMAGE_DIR=/hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr
  GT_DIR=/hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations
  UMAMBA_PRED=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_3/validation
  OURS_PRED=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_3/validation

  指定病例：
  BraTS20_Training_087
  BraTS20_Training_299
  BraTS20_Training_315

  输出目标：
  1. 生成完整脑切片对比图：/hy-tmp/brats_f3_failure_cases_full.png和.pdf
  2. 生成ET局部放大图：/hy-tmp/brats_f3_failure_cases_zoom.png和.pdf
  3. 每行一个病例，列为FLAIR或T1ce、GT、U-Mamba、Ours。
  4. 优先选择GT中ET面积最大的axial slice；如果该slice脑部显示不完整或ET太小，再输出肿瘤整体最大slice作为备选。
  5. mask颜色使用：label1绿色，label2黄色，label3红色；红色ET区域需要最醒目。
  6. 局部放大图要围绕GT肿瘤或ET区域外接框裁剪，并保留适当上下文。
  7. 需要处理imagesTr中`.nii.gz`可能不是标准gzip压缩的问题，不能因为nibabel报“not a gzip file”而失败。
  8. 图中不要写大段解释文字，只保留列名、病例ID和必要图例。

  请优先复用或修复umamba/0607/visualize_brats_predictions.py。如果现有脚本不方便，请新建独立脚本，但不要影响训练代码。
  最后输出可直接运行的命令、生成的文件路径，以及一段可写入论文“可视化分析”的简短说明。
  ```

- [x] 将f3退化分析写入修订稿。
  - 目标文件：`基于三视图状态空间建模的脑肿瘤MRI分割_修订稿.md`
  - 建议位置：五折交叉验证结果分析之后。
  - 写法要求：承认波动，避免强行解释成优势；强调小体积ET和固定阶段策略的局限。
  - 已写入：修订稿表4及其后续分析段落。

## 二、处理C7：统计显著性检验

- [x] 确认五折HD95 JSON是否齐全。
  - 每个fold都需要U-Mamba和Ours各一个HD95 JSON。
  - 命名格式推荐：`{TRAINER}_fold{FOLD}_hd95.json`。
  - 当前核查：新版`c7_case_level_metrics.csv`中HD95已覆盖f0-f4，五折均可用于配对统计。

- [x] 运行五折配对统计。
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

- [x] 下载/复制C7结果到`paper_revision_0629`目录。
  - `c7_case_level_metrics.csv`
  - `c7_significance_summary.json`

- [x] 根据p值决定论文表述。
  - 若HD95显著：写“配对检验显示HD95改善具有统计意义”。
  - 若不显著：写“整体趋势改善，但不同病例和不同折间仍存在波动”。
  - 不要在p值不支持时写“显著提升”。
  - 已写入修订稿：五折369个配对病例的Mean Dice统计检验。Ours在227例优于U-Mamba、142例低于U-Mamba，Mean Dice平均差值为0.0028，中位数差值为0.0016；Wilcoxon p=0.00014，配对t检验p=0.2179。因此论文表述为“病例层面小幅正向偏移、均值差异未达显著水平”，不写显著均值提升。
  - 已写入修订稿：Mean HD95平均降低0.5733，配对t检验p=0.1984，Wilcoxon p=0.1077。因此论文表述为“HD95整体下降趋势”，不写显著边界误差降低。

## 三、处理C8：均值与标准差核对

- [x] 使用C7输出或重新运行C8统计。
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

- [x] 核对论文中五折表格的均值。
  - U-Mamba平均Dice：目前记录为85.27%。
  - Ours平均Dice：目前记录为85.53%。
  - U-Mamba平均HD95：目前记录为4.606。
  - Ours平均HD95：目前记录为4.077。
  - 核对结果：表3均值与当前论文记录一致。
  - 按表3各fold汇总值计算的标准差：
    - U-Mamba Mean Dice：2.13个百分点。
    - Ours Mean Dice：2.46个百分点。
    - U-Mamba Mean HD95：0.556。
    - Ours Mean HD95：1.055。
  - 说明：`c7_significance_summary.json`中的`fold_mean_std`为逐病例指标聚合口径，和表3的区域/折汇总展示口径略有差异；论文正文优先使用表3口径补充标准差，避免主表与正文口径混用。

- [x] 决定表格是否改成`mean±std`。
  - 如果版面允许，主表推荐写`mean±std`。
  - 如果版面紧张，正文中补充标准差，表格保留均值。
  - 当前决策：表3保留逐fold和平均值展示，不改成`mean±std`；在表后正文补充标准差说明，已写入修订稿。

## 四、处理C1：五折主线消融

- [x] 补跑EncoderOnly的f1-f4。
  ```bash
  for FOLD in 1 2 3 4; do
    nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs
  done
  ```

- [x] 补跑StageAwareDecoder的f1-f4。
  ```bash
  for FOLD in 1 2 3 4; do
    nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs
  done
  ```

- [x] 收集C1五折Dice和HD95。

- [x] 更新论文消融表。
  - 已将修订稿表5由“第0折消融实验结果”更新为“五折主线消融实验结果”。
  - 五折主线消融结果如下：

  | 方法 | Mean Dice/% | Mean HD95 |
  | ---- | ----------: | --------: |
  | U-Mamba | 85.27±2.13 | 4.606±0.556 |
  | +ETSM | 85.21±2.02 | 4.263±0.992 |
  | +ETSM+Stage | 85.23±2.20 | 4.603±1.198 |
  | +ETSM+Stage+Skip | 85.53±2.46 | 4.077±1.055 |

  - 论文分析已改为五折口径：单独引入ETSM和阶段感知解码时增益并不单调，加入SkipCalibration后取得最高Mean Dice和最低Mean HD95。

## 五、处理C2-C4：新增消融对照

注意：涉及代码训练器入口的实现和修复，后续默认生成Claude提示词，让Claude完成代码检查或重写。

- [x] C2：全解码层ETSM对照。
  - 先跑f0：
    ```bash
    nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
    ```
  - 若显存不够或结果异常，暂不扩展五折，在论文中只保留阶段感知策略的理论解释。
  - 已完成f0筛查：RTX3090 24GB上触发CUDA OOM，当前不扩展五折；该结果可作为阶段感知解码设计的资源开销依据。
  - 补充结果：`PartialDecoderETSM_150epochs` f0可训练，Mean Validation Dice=0.8872545937003561，Mean HD95=4.248719；待补区域Dice/复杂度后决定是否写入论文消融表。

- [ ] C3：`w/o cross-view gating`对照。
  - 先跑f0：
    ```bash
    nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs
    ```
  - 若差异明显，再考虑扩展五折。
  - 已完成f0训练：Mean Validation Dice=0.8889759482835508，Mean HD95=2.791524，均优于当前最终方法f0。需要决定是否扩展f1-f4，或仅作为负向消融说明cross-view gating贡献有限。

- [x] C4：Attention U-Net 0-1门控对照。
  - 先让Claude检查/重写该训练器逻辑。
  - 再跑f0：
    ```bash
    nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_AttentionSkipGate_150epochs
    ```
  - 已完成f0筛查：Mean Dice=88.46%，Mean HD95=4.608，均劣于最终方法f0的88.86%和3.444。
  - 已写入修订稿消融实验段落，作为跳跃连接标定机制对照。

## 六、处理C5：公平横向基线

- [x] 完成nnU-Net五折训练并收集Dice/HD95结果。
  ```bash
  for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainer
  done
  ```

- [ ] 补充或核对nnU-Net的参数量、推理时间、显存。

- [x] 补齐SegMamba f2-f4。
  - f0已完成：Mean Dice=87.72%，Mean HD95=4.256。
  - f1已完成：Mean Dice=85.08%，Mean HD95=3.998。
  - f2已完成：Mean Dice=84.16%，Mean HD95=5.241。
  - f3已完成：Mean Dice=82.17%，Mean HD95=5.934。
  - f4已完成：Mean Dice=82.28%，Mean HD95=4.458。
  - 五折均值：Mean Dice=84.28±2.29%，Mean HD95=4.777±0.796。
  - 每折预测后需先执行`np.transpose(pred, (2, 1, 0))`轴顺序修正，再计算Dice和HD95。
  - 已具备写入正式横向对比表的条件；表述需强调SegMamba为官方独立实现，本文对齐数据划分、训练轮数和评价口径。

## 七、处理D类可视化联动

- [x] 根据C6结果选择失败案例。
  - 首选：`BraTS20_Training_087.nii.gz`
  - 候选：`BraTS20_Training_299.nii.gz`
  - 候选：`BraTS20_Training_315.nii.gz`
  - 已完成：上述三个病例已确认为f3失败案例可视化候选，具体理由见C6部分表格。

- [ ] 根据C6结果选择改善案例。
  - 候选：`BraTS20_Training_260.nii.gz`
  - 候选：`BraTS20_Training_177.nii.gz`
  - 候选：`BraTS20_Training_246.nii.gz`

- [x] 生成局部放大图或边界对比图。
  - 目的：解释HD95改善和f3小ET失败场景。
  - 代码任务优先交给Claude实现。
  - 已完成：`brats_f3_failure_cases_full.png`和`brats_f3_failure_cases_zoom.png`已放入`umamba/paper_revision_0629/paper_assets/`，并写入修订稿图5和图6。

## 八、论文与回复文件同步

- [x] 更新`基于三视图状态空间建模的脑肿瘤MRI分割_修订稿.md`。
  - 补f3统计分析。
  - 补显著性检验或谨慎说明。
  - 更新消融表/五折表。

- [x] 更新`审稿意见回复说明.md`。
  - C6已从“未完成/部分完成”改为“已补充病例统计与失败案例可视化”。
  - C7/C8根据输出结果更新。

- [x] 更新`审稿意见整改矩阵.md`。
  - C6已补齐HD95并写入论文，状态标记为“已完成”。
  - C7/C8已运行并写入论文，状态标记为“已完成”。
  - C1和C5已有真实训练及统计结果，状态标记为“已完成”。
