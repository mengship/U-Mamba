# UD-Mamba 借鉴思路

## 一、总体判断

UD-Mamba 的主要价值不在于整网结构，而在于它提出了一种面向 Mamba 扫描的 uncertainty-driven 思路：

```text
根据像素/patch 特征的不确定性重新组织扫描顺序
+ 使用多方向扫描输出的一致性约束辅助训练
```

这和当前 RTHD 的三视图 Mamba 建模天然相关。RTHD 已经有 axial / coronal / sagittal 三视图分解，UD-Mamba 可以启发我们进一步做：

```text
uncertainty-guided RTHD refinement
tri-view consistency loss
uncertainty-guided view fusion
```

不建议直接把 UD-Mamba 网络迁移到当前 BraTS 3D nnU-Net 框架。它是 2D Mamba-Unet 代码，更适合作为局部思想参考。

## 二、UD-Mamba 本地代码位置

本地代码路径：

```text
/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/UD-mamba
```

关键文件：

```text
code/networks/mamba_sys.py
code/networks/vision_mamba.py
code/train_fully_supervised_2D_UD-Mamba.py
```

核心模块：

```text
SS2D
VSSBlock
VSSLayer_up
VSSM
```

核心训练入口：

```text
outputs, loss_out = model(volume_batch)
loss = 0.5 * (loss_dice + loss_ce) + 0.3 * loss_out
```

注意：代码里 `loss_out` 在网络 forward 中看起来是 list，实际使用时需要确认是否应先求和或取均值。

## 三、核心机制 1：不确定性排序扫描

来源位置：

```text
UD-mamba/code/networks/mamba_sys.py
sort_by_entropy()
```

代码逻辑大致是：

```text
x: B, C, H, W
M = x.view(B, C, H * W)
score = mean(abs(M - mean(M, dim=channel)), dim=channel)
sorted_indices = sort(score, descending=True)
sorted_x = gather(M, sorted_indices)
```

虽然函数名叫 `sort_by_entropy`，但它实际计算的不是概率熵，而是通道维度上的 feature dispersion / mean absolute deviation。

可以理解为：

```text
通道响应越不一致的位置，越可能是不确定区域或边界区域。
```

UD-Mamba 随后对排序后的 token 做 Mamba 扫描，再恢复原始空间顺序：

```text
sorted feature
  -> multi-direction selective scan
  -> restore original token order
```

### 对 RTHD 的启发

RTHD 当前的三视图建模主要关注如何把 3D 分解为 2D 视图并重建。UD-Mamba 启发我们进一步考虑：

```text
哪些位置更值得优先建模？
哪些区域的跨视图表示更不稳定？
```

可迁移方向：

```text
uncertainty map = feature dispersion over channels
uncertainty map guides RTHD view fusion or decoder refinement
```

## 四、核心机制 2：多方向扫描一致性约束

来源位置：

```text
UD-mamba/code/networks/mamba_sys.py
cosine_similarity_loss()
SS2D.forward_corev0()
```

UD-Mamba 在 2D 中构造四个方向扫描输出：

```text
y1: normal scan
y2: reverse scan
y3: transpose scan
y4: reverse transpose scan
```

然后使用 cosine similarity 构造一致性约束：

```text
loss = 1 - mean(cos_sim(y1, y3), cos_sim(y2, y4))
```

这个 loss 只在 decoder 侧触发。代码里用 `position > 50` 判断是否返回额外 loss，而 decoder 构建时传入 `position=self.position+100`。

### 对 RTHD 的启发

RTHD 天然有三视图输出：

```text
axial
coronal
sagittal
```

因此可以把 UD-Mamba 的方向一致性约束改造成：

```text
tri-view consistency loss
```

推荐形式：

```text
loss_view =
  1 - cosine(axial_feature, coronal_feature)
+ 1 - cosine(axial_feature, sagittal_feature)
+ 1 - cosine(coronal_feature, sagittal_feature)
```

或者更稳定的概率分布形式：

```text
loss_view =
  KL(softmax(view_i), softmax(view_j))
```

优点：

- 训练期增强三视图一致性。
- 推理期不增加参数量。
- 推理期不增加 FLOPs。
- 和 RTHD 的三视图设计高度贴合。

## 五、核心机制 3：Decoder-only Uncertainty Constraint

来源位置：

```text
UD-mamba/code/networks/mamba_sys.py
VSSLayer_up
VSSM.forward_up_features()
```

UD-Mamba 的额外一致性 loss 主要放在 decoder up layer，而不是 encoder。

这点非常适合当前 RTHD 的 stage-aware decoder 叙事：

```text
Encoder RTHD: global tri-view representation learning
Decoder RTHD: uncertainty-guided structure recovery
High-resolution decoder: local boundary refinement
```

建议迁移方式：

```text
只在 RTHD decoder D4 / D3 加 uncertainty-guided consistency loss
不要一开始放到所有 encoder/decoder stage
```

原因：

- D4 / D3 分辨率较低，计算和显存更可控。
- decoder 负责结构恢复，更容易解释 uncertainty refinement。
- 可以避免对 encoder 表示学习造成过强约束。

## 六、推荐迁移方案

### 方案 A：RTHD Tri-view Consistency Loss

这是最推荐的第一版。

思路：

在 RTHD block 内部暴露三个视图的重建前或重建后特征：

```text
axial_feature
coronal_feature
sagittal_feature
```

训练时计算：

```text
loss_tri_view_consistency
```

总损失：

```text
loss = nnUNet_loss + lambda_cons * loss_tri_view_consistency
```

推荐初始权重：

```text
lambda_cons = 0.05 or 0.1
```

放置位置：

```text
Decoder RTHD stages D4 / D3
```

预期收益：

- 改善跨视图重建一致性。
- 对边界和小病灶可能有帮助。
- 不增加推理复杂度。

### 方案 B：Uncertainty-guided View Gate

思路：

用 feature dispersion 生成 uncertainty map，然后调节三视图融合权重。

流程：

```text
x_3d
  -> channel dispersion uncertainty map
  -> axial / coronal / sagittal view features
  -> view weights conditioned on uncertainty
  -> gated reconstruction
```

可以和当前 `TriViewReconstruction(mode='gated')` 对齐。

推荐形式：

```text
gate_logits = Conv3d(concat(axial_3d, coronal_3d, sagittal_3d, uncertainty_map))
view_gates = softmax(gate_logits, dim=view)
```

优点：

- 比 token sorting 更符合 3D 空间结构。
- 可解释为 uncertain region adaptive view fusion。

风险：

- 会增加少量参数。
- 需要确认不会和已有 gated reconstruction 重复。

### 方案 C：Uncertainty-guided RTHD Refinement

思路：

只在高不确定区域增强 RTHD 输出：

```text
uncertainty = channel_dispersion(x)
rthd_out = RTHD(x)
out = x + scale * uncertainty * (rthd_out - x)
```

优点：

- 让 RTHD 更关注边界/难分区域。
- 计算图简单。
- 可以作为 decoder refinement 插件。

风险：

- 如果 uncertainty map 不稳定，可能放大噪声。
- 需要小权重初始化，例如 `scale=0.1`。

### 方案 D：3D Token Sorting

不建议作为第一版。

虽然可以把 UD-Mamba 的排序扩展到 3D：

```text
x: B, C, D, H, W
score: B, D*H*W
sort tokens by score
scan sorted sequence
restore original order
```

但风险较高：

- 破坏 3D 空间邻接关系。
- 排序和 gather 对大体积特征开销不低。
- 和 RTHD 的三视图空间建模主线可能冲突。

如果要做，只建议在最低分辨率 bottleneck 或 D4 stage 做消融。

## 七、不建议直接照搬的点

### 1. 不建议整网迁移

UD-Mamba 是 2D patch-based Mamba-Unet，当前任务是 BraTS 3D nnU-Net / U-Mamba 框架。整网替换会导致变量过多，也会削弱 RTHD 主线。

### 2. 不建议直接使用二分类标签处理

UD-Mamba 训练脚本中有：

```text
label_batch[label_batch != 0] = 1
```

这会把所有非背景合成一个类别，不适合 BraTS 的 WT / TC / ET 区域评价。

### 3. 不建议照搬 token sorting 到高分辨率 3D stage

高分辨率 3D token 数量大，排序开销和空间结构破坏都明显。更推荐使用 uncertainty map 做 soft gating。

### 4. 注意 loss_out 的实现细节

UD-Mamba 代码中网络返回的 `out_loss` 是 list，但训练脚本直接使用：

```text
loss = 0.5 * (loss_dice + loss_ce) + 0.3 * loss_out
```

迁移时应明确写成：

```text
loss_aux = sum(losses) / max(len(losses), 1)
loss = base_loss + lambda_aux * loss_aux
```

## 八、推荐实验优先级

| 优先级 | 改进点 | 放置位置 | 推理成本 | 风险 |
|---|---|---|---|---|
| 高 | Tri-view Consistency Loss | Decoder D4/D3 RTHD | 无增加 | 低 |
| 高 | Uncertainty-guided View Gate | RTHD reconstruction | 少量增加 | 中 |
| 中 | Uncertainty-guided RTHD Refinement | Decoder refinement | 少量增加 | 中 |
| 低 | 3D Token Sorting | Bottleneck / D4 | 中等增加 | 高 |

## 九、建议第一个实现版本

建议命名：

```text
nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_UncertaintyConsistency_150epochs
```

包含：

```text
Encoder: existing RTHD
Decoder D4/D3: existing RTHD refinement
Auxiliary training loss: tri-view consistency loss
Skip gate / boundary head: 暂时关闭或沿用已有最佳配置
```

第一轮消融：

```text
RTHD StageAwareDecoder
RTHD StageAwareDecoder + TriViewConsistency
RTHD StageAwareDecoder + TriViewConsistency + SkipCalibration
```

评价指标：

```text
Dice WT / TC / ET
HD95 WT / TC / ET
Params
FLOPs
Inference time
```

重点观察：

```text
是否改善 HD95
是否改善 ET 小区域
是否在不增加推理 FLOPs 的情况下提升稳定性
```

## 十、论文叙事草稿

英文：

```text
Inspired by uncertainty-driven scanning in UD-Mamba, we introduce a decoder-side tri-view consistency constraint for RTHD. Instead of explicitly sorting 3D tokens, which may disrupt volumetric spatial continuity, we estimate uncertainty from feature dispersion and encourage consistent representations among axial, coronal, and sagittal RTHD views. This auxiliary constraint improves structure recovery during training without introducing additional inference cost.
```

中文：

```text
受 UD-Mamba 中不确定性驱动扫描思想启发，本文在 RTHD 解码阶段引入三视图一致性约束。不同于直接对 3D token 进行排序，本文利用特征离散度刻画局部不确定性，并约束轴状位、冠状位和矢状位视图表示保持一致，从而在不增加推理复杂度的情况下增强解码阶段的结构恢复能力。
```
