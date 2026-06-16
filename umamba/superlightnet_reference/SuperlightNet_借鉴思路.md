# SuperlightNet 借鉴思路

## 一、总体判断

SuperlightNet 不建议整网替换当前 U-Mamba / RTHD 主体。它更适合作为轻量化 decoder 和 skip fusion 的设计参考。

当前 RTHD 的优势在于三视图全局空间建模，尤其适合编码器中高层和低分辨率 decoder stage。SuperlightNet 的优势在于用很低成本完成上采样、skip 融合和局部结构恢复。因此更合理的借鉴方式是：

```text
Encoder / low-resolution decoder: RTHD for global 3D structure modeling
High-resolution decoder / skip path: Superlight-style lightweight refinement
```

也就是把方法叙事从“RTHD 编码器模块”扩展成：

```text
RTHD global modeling + Superlight local reconstruction + calibrated skip fusion
```

## 二、SuperlightNet 中值得关注的模块

### 1. Learnable Res Skip UpRepr

来源位置：

```text
umamba/instructions/superlightnet.py
class Learnable_Res_Skip_UpRepr4
```

核心流程：

```text
decoder feature
  -> 1x1 conv channel projection
  -> linear upsample
  -> add learnable_channel_scale * skip
  -> InstanceNorm
  -> grouped/depthwise local refinement
  -> residual output with learnable residual scale
```

关键设计：

- 使用相加而不是 concat，减少融合后的通道膨胀。
- skip 分支有逐通道可学习缩放 `group_skip_scale`。
- decoder 残差分支有全局可学习缩放 `group_res_scale`。
- 局部细化使用 grouped/depthwise convolution，计算量低。

适合迁移到当前 RTHD 的位置：

```text
UMambaEnc_RTHD.py
UNetResDecoder_RTHD.forward()
```

当前 decoder 是：

```text
x_up = upsample(lres_input)
skip = skip_gate(skip, x_up)
x = concat(x_up, skip)
x = BasicResBlock / RTHDBlock
```

可以新增一个轻量变体：

```text
x = x_up + skip * channel_skip_scale
x = local_refine(x)
x = x + residual_scale * x
```

优先建议只用于高分辨率 decoder stage，例如 D2 / D1。

### 2. Half-channel Processing

来源位置：

```text
umamba/instructions/superlightnet.py
class THPAEncFR3
```

SuperlightNet 的做法是把输入通道切成两半：

```text
x_main, x_residual = chunk(x, 2)
x_main = spatial attention / view modeling
out = MLP(concat(x_main, x_residual))
```

这个思想很适合改造成 `HalfChannelRTHDBlock`：

```text
x_rthd, x_res = chunk(x, 2)
x_rthd = RTHD(x_rthd)
out = 1x1 MLP(concat(x_rthd, x_res))
```

潜在收益：

- 降低 RTHD 计算和显存。
- 保留一半原始局部表示，减少三视图投影/重建带来的细节损失。
- 论文叙事可以写成 partial-channel global modeling。

建议优先放置：

```text
Decoder D3 / D4 RTHD refinement
```

不建议一开始替换所有 encoder RTHD，否则变量太多，消融不清楚。

### 3. Grouped Multi-axis Hadamard Product Attention

来源位置：

```text
umamba/instructions/superlightnet.py
class Grouped_multi_axis_Hadamard_Product_Attention
```

它把通道分成四组：

- 一组做 XY 平面调制。
- 一组做 ZX 方向调制。
- 一组做 ZY 方向调制。
- 一组做 depthwise local convolution。

这个设计和 RTHD 的三视图思想有相似性，但它更像轻量 attention / gate，不是 SSM。

可参考但不建议直接照搬。原因：

- 当前 RTHD 已经有 axial / coronal / sagittal 三视图建模。
- 直接再加多轴 attention 容易和 RTHD 功能重叠。
- 更适合转化为 skip 或 decoder 的轻量门控模块，而不是替代 RTHD 主体。

### 4. NormDownsample

来源位置：

```text
umamba/instructions/superlightnet.py
class NormDownsample
```

流程很简单：

```text
InstanceNorm3d -> Conv3d(kernel=2, stride=2)
```

这个可以作为下采样稳定性的小实验，但优先级较低。当前 nnU-Net / U-Mamba 的下采样逻辑已经比较成熟，贸然替换可能影响 baseline 公平性。

## 三、不建议直接借鉴的点

### 1. forward 中随机选择方向

SuperlightNet 的 `THPAEncFR3` 在 forward 时随机选择一个方向：

```text
random_direction = torch.randint(0, 3, (1,)).item()
```

不建议直接用于当前医学分割主实验。原因：

- 训练和推理行为可能不稳定。
- 可复现性变差。
- BraTS 这类任务对 3D 空间一致性要求高，随机单方向建模可能损伤结构连续性。

如果要参考，建议改成消融实验里的 `view dropout` 或 `stochastic view sampling`，而不是主方法。

### 2. 整体替换为 SuperlightNet

不建议把当前 U-Mamba / RTHD 主干替换成 SuperlightNet。这样会让论文主线偏离 RTHD，而且难以说明提升来自 RTHD 还是整网重构。

更好的方式是只借鉴局部模块：

```text
skip calibration
high-resolution decoder refinement
half-channel RTHD
```

## 四、推荐实验路线

### 实验 A：Superlight Skip Scale

目标：

在当前 `SemanticSkipFusionGate3d` 基础上加入逐通道 skip scale。

当前形式：

```text
gate = spatial_gate(concat(skip, decoder))
refined_skip = skip + gate_scale * skip * gate
```

建议形式：

```text
refined_skip = skip * channel_scale + gate_scale * skip * gate
```

预期收益：

- 成本极低。
- 改善 skip 中噪声特征直接传入 decoder 的问题。
- 和当前 SkipCalibration 命名高度一致。

建议命名：

```text
ChannelSemanticSkipFusionGate3d
SuperlightSkipCalibration3d
```

### 实验 B：High-resolution Superlight Decoder Refinement

目标：

低分辨率 decoder stage 保留 RTHD，高分辨率 decoder stage 使用 grouped/depthwise local refinement。

推荐结构：

```text
D4 / D3: RTHD refinement
D2 / D1: Superlight local refinement
```

预期收益：

- 保留 RTHD 的全局结构建模能力。
- 用低成本局部细化改善边界和小病灶。
- 有利于 HD95。
- 参数量和 FLOPs 增加可控。

建议命名：

```text
StageAwareDecoder_SuperlightRefine
```

### 实验 C：HalfChannel RTHD

目标：

只对一半通道做 RTHD，另一半通道作为局部残差保留。

推荐结构：

```text
HalfChannelRTHDBlock:
  x_rthd, x_res = split(x)
  x_rthd = RTHDBlock(x_rthd)
  out = channel_mlp(concat(x_rthd, x_res))
```

优先放置：

```text
decoder RTHD stages: D4 / D3
```

预期收益：

- 降低复杂度。
- 减少 RTHD 对局部细节的过度重构。
- 强化轻量化论文叙事。

风险：

- 如果通道过少，RTHD 表达能力可能下降。
- 需要和完整 RTHD 做 Dice / HD95 / Params / FLOPs 对比。

## 五、建议优先级

| 优先级 | 改进点 | 放置位置 | 预期收益 | 风险 |
|---|---|---|---|---|
| 高 | Superlight Skip Scale | SkipCalibration | 低成本改善 skip 融合 | 很低 |
| 高 | High-resolution Superlight Refinement | Decoder D2/D1 | 改善边界和 HD95 | 低 |
| 中 | HalfChannel RTHD | Decoder D4/D3 | 降低复杂度，增强轻量化 | 中 |
| 低 | NormDownsample | Encoder downsample | 可能稳定训练 | 会影响 baseline 公平性 |
| 低 | Stochastic View Sampling | RTHD view path | 有创新感 | 稳定性和复现性风险 |

## 六、推荐第一个实现版本

建议先实现一个最稳、变量最少的版本：

```text
nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SuperlightSkipRefine_150epochs
```

包含：

```text
Encoder: existing RTHD
Decoder D4/D3: existing stage-aware RTHD
Skip: SemanticSkipFusionGate3d + channel skip scale
Decoder D2/D1: Superlight-style grouped/depthwise local refinement
Boundary/Frequency head: 暂时关闭
```

这样第一轮消融可以对比：

```text
Baseline UMambaEnc
RTHD EncoderOnly
RTHD StageAwareDecoder
RTHD StageAwareDecoder + SkipCalibration
RTHD StageAwareDecoder + SuperlightSkipRefine
```

评价指标：

```text
Dice WT / TC / ET
HD95 WT / TC / ET
Params
FLOPs
Inference time
```

## 七、论文叙事草稿

可以把这部分写成：

```text
Although RTHD effectively captures long-range 3D dependencies through tri-view state space modeling, directly applying such global modeling to all decoder stages is computationally unnecessary and may be suboptimal for high-resolution boundary recovery. Inspired by lightweight reconstruction designs, we introduce a Superlight-style decoder refinement strategy, where low-resolution decoder stages employ RTHD for global structure recovery, while high-resolution stages adopt learnable skip scaling and grouped local refinement for efficient detail restoration.
```

中文表述：

```text
虽然 RTHD 能够通过三视图状态空间建模有效捕获长程 3D 依赖，但在所有解码阶段都使用全局建模并不必要，且高分辨率阶段更关注边界和局部细节恢复。因此，本文借鉴轻量化重建思想，在低分辨率解码阶段保留 RTHD 进行全局结构恢复，在高分辨率阶段引入可学习跳跃缩放和分组局部细化，以较低计算代价提升边界恢复能力。
```
