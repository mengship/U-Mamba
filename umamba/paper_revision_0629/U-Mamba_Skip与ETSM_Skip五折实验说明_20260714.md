# U-Mamba+Skip 与 ETSM+Skip 五折实验说明

## 1. 实验目的

本轮补充两个严格单变量对照，用于拆分语义引导跳跃连接特征标定（Skip Calibration）的独立作用：

| 实验 | 编码器 | 解码器 | Skip Calibration | 其他增强模块 |
| --- | --- | --- | --- | --- |
| U-Mamba | 原始MambaLayer | 原始卷积 | 关闭 | 关闭 |
| U-Mamba+Skip | 原始MambaLayer | 原始卷积 | 开启，阶段[0,1] | 关闭 |
| ETSM | 编码器ETSM | 原始卷积 | 关闭 | 关闭 |
| ETSM+Skip | 编码器ETSM | 原始卷积 | 开启，阶段[0,1] | 关闭 |

其中，U-Mamba+Skip仅相较U-Mamba增加Skip Calibration；ETSM+Skip仅相较现有EncoderOnly ETSM增加Skip Calibration。两组均不使用解码器ETSM、边界注意力或频率细化模块。

## 2. 五折训练命令

### 2.1 U-Mamba+Skip

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEnc_SkipCalibration_150epochs
done
```

### 2.2 ETSM+Skip

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 705 3d_fullres ${FOLD} -tr nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration_150epochs
done
```

如需断点续训，在对应命令末尾增加`--c`。正式五折实验不应混用从其他Trainer生成的检查点。

## 3. 曲线与结果归档

nnU-Net会在每个fold的结果目录中自动生成`progress.png`。两组实验完成后，每折至少保留：

- `progress.png`：训练损失、验证损失和伪Dice曲线；
- `checkpoint_final.pth`与`checkpoint_best.pth`；
- `training_log_*.txt`；
- `validation/`目录及其评价结果；
- `summary.json`（若验证流程已生成）。

结果目录名称应分别包含完整Trainer名：

- `nnUNetTrainerUMambaEnc_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0`至`fold_4`；
- `nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0`至`fold_4`。

建议将五折曲线按`U-Mamba+Skip_f0.png`至`f4.png`、`ETSM+Skip_f0.png`至`f4.png`重命名后备份，但原始结果目录仍完整保留，避免丢失可追溯信息。

在当前Dataset705结果目录命名不变的前提下，可在云GPU上执行：

```bash
RESULTS_ROOT="${nnUNet_results}/Dataset705_BraTS2020"
CURVE_DIR="/hy-tmp/ablation_curves_20260714"
mkdir -p "${CURVE_DIR}"

for FOLD in 0 1 2 3 4; do
  cp "${RESULTS_ROOT}/nnUNetTrainerUMambaEnc_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_${FOLD}/progress.png" \
    "${CURVE_DIR}/U-Mamba+Skip_f${FOLD}.png"
  cp "${RESULTS_ROOT}/nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_${FOLD}/progress.png" \
    "${CURVE_DIR}/ETSM+Skip_f${FOLD}.png"
done
```

`progress.png`反映训练损失、验证损失、伪Dice、每轮耗时和学习率变化，用于检查收敛和异常波动；论文中的最终Dice与HD95仍应从五折验证结果统一计算，不能用伪Dice曲线直接替代。

## 4. 结果进入论文的条件

在两组五折全部完成并按统一Dice、HD95口径汇总前，不修改修订稿中的消融表和结论。完成后优先比较：

1. U-Mamba+Skip相对U-Mamba的五折均值、标准差和逐折变化；
2. ETSM+Skip相对ETSM的五折均值、标准差和逐折变化；
3. Skip在无ETSM与有ETSM条件下的增益方向是否一致；
4. 若增益方向不一致，则按模块交互解释，不表述为Skip的稳定独立增益。
