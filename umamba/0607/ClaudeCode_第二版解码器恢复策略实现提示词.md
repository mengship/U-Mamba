# Claude Code 第二版实现提示词：RTHD 解码器结构恢复策略

你现在在一个本地代码仓库中工作，需要帮助我基于现有 `RTHD` 代码实现第二版增强。  
请直接修改代码并自行完成必要检查。  
不要只给建议，要实际改代码。


## 一、背景

当前项目是基于 U-Mamba / nnU-Net 的 3D 脑肿瘤分割项目。  
已有 `RTHD (Recursive Tri-view Hierarchical Decomposition)` 模块，核心思想是：

`3D feature -> tri-view projection -> 2D VMamba scanning -> 3D reconstruction`

第一版增强已经实现：

- `gated reconstruction`
- `minimal cross-view interaction`

核心文件：

- `umamba/nnunetv2/nets/rthd_modules.py`
- `umamba/nnunetv2/nets/UMambaEnc_RTHD.py`

现有代码中已经有：

- `TriViewProjection`
- `TriViewReconstruction`
- `TriViewVMambaBlock`
- `RTHDBlock`
- `UNetResDecoder_RTHD`
- `decoder_rthd_mode = "none" / "partial" / "full"`
- `rthd_stages_decoder`
- `rthd_config_encoder`
- `rthd_config_decoder`

当前问题：

虽然已有 RTHD，但主要创新仍偏向编码器。  
作为毕业论文中的一个完整章节，内容还不够厚。  
这次希望把第一章方法补强为：

`Encoder RTHD + Stage-aware Decoder RTHD Refinement + Semantic Skip Fusion + Boundary-aware Output`

重点是让方法从“只改编码器”升级为：

`编码器全局建模 + 解码器结构恢复 + 跳跃连接选择性融合 + 输出边界增强`


## 二、这次实现的总目标

实现一个低风险、可消融、默认兼容的解码器结构恢复增强版本：

1. `Stage-aware RTHD Decoder`
2. `Semantic-guided Skip Fusion Gate`
3. `Boundary-aware Segmentation Attention`
4. 可选：`High-low Frequency Structure Recovery`

注意：

- 不要引入 foundation model
- 不要引入缺失模态训练
- 不要改 nnU-Net 主训练流程
- 不要改默认 loss
- 不要默认返回额外 boundary map
- 不要破坏原有 trainer 和原有配置


## 三、必须遵守的兼容性要求

所有新增功能默认关闭。

如果用户不传新参数，当前模型行为应尽量保持不变。

新增参数建议默认：

```python
use_skip_fusion_gate: bool = False
skip_gate_stages: List[int] = None
use_boundary_attention_head: bool = False
use_frequency_refinement: bool = False
frequency_refinement_stages: List[int] = None
```

保留已有参数：

```python
use_rthd_decoder
decoder_rthd_mode
rthd_stages_decoder
rthd_config
rthd_config_encoder
rthd_config_decoder
```

已有 `decoder_rthd_mode="none" / "partial" / "full"` 不能破坏。  
已有 `rthd_config_encoder` / `rthd_config_decoder` 的回退逻辑不能破坏。  
已有 `deep_supervision=True/False` 输出格式不能破坏。


## 四、实现任务 A：Semantic-guided Skip Fusion Gate

### 4.1 目标

当前 `UNetResDecoder_RTHD.forward` 中是：

```python
x = self.upsample_layers[s](lres_input)
x = torch.cat((x, skips[-(s+2)]), 1)
x = self.stages[s](x)
```

这会直接 concat decoder feature 和 encoder skip feature。  
希望改成可选的语义引导 skip gate：

```python
x = self.upsample_layers[s](lres_input)
skip = skips[-(s+2)]
if use_skip_fusion_gate and s in skip_gate_stages:
    skip = self.skip_gates[s](skip, x)
x = torch.cat((x, skip), 1)
x = self.stages[s](x)
```

注意这里不是循环：

- `self.skip_gates[s](skip, x)` 内部可以临时 concat `skip` 和 `x` 来生成 gate
- 这个临时 concat 只服务于 gate generator，不作为最终 decoder 输入
- 外部的 `torch.cat((x, skip), 1)` 才是最终送入 decoder stage 的正式融合

推荐等价流程：

```python
x_up = self.upsample_layers[s](lres_input)
skip = skips[-(s + 2)]

if self.use_skip_fusion_gate and s in self.skip_gate_stages:
    skip = self.skip_gates[s](skip, x_up)  # internal temporary concat only for gate

x = torch.cat((x_up, skip), dim=1)         # final decoder fusion
x = self.stages[s](x)
```


### 4.2 建议新增模块

建议在 `umamba/nnunetv2/nets/rthd_modules.py` 中新增：

```python
class SemanticSkipFusionGate3d(nn.Module):
    ...
```

输入：

```python
skip:    (B, C, D, H, W)
decoder: (B, C, D, H, W)
```

输出：

```python
refined_skip: (B, C, D, H, W)
```

推荐实现：

1. 拼接 `skip` 和 `decoder`: `(B, 2C, D, H, W)`
2. 使用轻量卷积生成 gate:

```python
nn.Conv3d(2 * dim, hidden_dim, kernel_size=1, bias=False)
norm / activation
nn.Conv3d(hidden_dim, dim, kernel_size=1, bias=True)
sigmoid
```

3. 残差式门控：

```python
refined_skip = skip + skip * gate
```

注意：

- 如果 skip 和 decoder 空间尺寸不一致，请用 `F.interpolate` 将 decoder 对齐到 skip 尺寸
- hidden_dim 可设置为 `max(dim // reduction, 16)`
- 新增参数 `skip_gate_reduction: int = 4`


### 4.3 在解码器中接入

修改 `UNetResDecoder_RTHD`：

- `__init__` 新增参数：

```python
use_skip_fusion_gate: bool = False
skip_gate_stages: List[int] = None
skip_gate_reduction: int = 4
```

- 建立 `self.skip_gates = nn.ModuleList(...)`
- stage 不启用 gate 时使用 `nn.Identity()` 或 `None`
- forward 中在 concat 前调用

默认：

```python
if skip_gate_stages is None:
    skip_gate_stages = [0, 1]
```

但只有 `use_skip_fusion_gate=True` 时才启用。


## 五、实现任务 B：Boundary-aware Segmentation Attention

### 5.1 目标

希望在输出端增加边界感知增强，但第一版不要改 loss，也不要返回额外 boundary map。

也就是说，这次先实现：

`boundary attention`

不是：

`boundary supervised branch`


### 5.2 建议新增模块

建议在 `umamba/nnunetv2/nets/rthd_modules.py` 中新增：

```python
class BoundaryAttentionHead3d(nn.Module):
    ...
```

输入：

```python
x: (B, C, D, H, W)
```

输出：

```python
refined_x: (B, C, D, H, W)
```

推荐实现：

```python
edge = depthwise Conv3d / Conv3d 生成边界响应
attn = sigmoid(Conv3d(edge))
refined_x = x + x * attn
```

建议结构：

```python
nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
norm / activation
nn.Conv3d(dim, 1, kernel_size=1, bias=True)
sigmoid
```

注意：

- 输出 attention channel 可以是 1，广播到 C
- 不要改变 segmentation head 输出格式
- 不要返回 boundary map
- 之后如果要做 boundary loss，再单独开一个任务


### 5.3 在解码器中接入

修改 `UNetResDecoder_RTHD`：

- `__init__` 新增参数：

```python
use_boundary_attention_head: bool = False
```

- 建立：

```python
self.boundary_attention_heads = nn.ModuleList(...)
```

或者只对最后一个 decoder stage 建立一个：

```python
self.final_boundary_attention = BoundaryAttentionHead3d(final_dim)
```

推荐最小实现：

- 只在最终输出前使用
- 即 `s == len(self.stages) - 1` 时，在 `seg_layers[-1]` 前调用

伪代码：

```python
if self.use_boundary_attention_head and s == len(self.stages) - 1:
    x = self.final_boundary_attention(x)
```

注意 deep supervision：

- `deep_supervision=True` 时，不要在每个 supervision head 都强行加边界增强
- 第一版只给最终最高分辨率输出前加
- 保持输出 list 顺序不变


## 六、实现任务 C：High-low Frequency Structure Recovery（可选但推荐）

### 6.1 目标

在解码器结构恢复阶段加入轻量高低频细节恢复。  
不要使用复杂 FFT/DCT，先用平均池化近似低频。


### 6.2 建议新增模块

建议在 `umamba/nnunetv2/nets/rthd_modules.py` 中新增：

```python
class HighLowFrequencyRefinement3d(nn.Module):
    ...
```

输入输出：

```python
x -> refined_x
```

推荐实现：

```python
low = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
high = x - low
gate = sigmoid(conv(high))
out = x + high * gate
```

可加一个轻量卷积：

```python
refined_high = depthwise_conv(high)
gate = sigmoid(pointwise_conv(refined_high))
```

参数：

```python
use_frequency_refinement: bool = False
frequency_refinement_stages: List[int] = None
```

默认：

```python
if frequency_refinement_stages is None:
    frequency_refinement_stages = [0, 1]
```

但只有 `use_frequency_refinement=True` 时才启用。


### 6.3 在解码器中接入

推荐在 `self.stages[s](x)` 之后调用：

```python
x = self.stages[s](x)
if use_frequency_refinement and s in frequency_refinement_stages:
    x = self.frequency_refiners[s](x)
```

这样不会影响 concat 之前的通道逻辑。


## 七、实现任务 D：网络级参数透传

需要把新增参数从顶层一路传到 `UNetResDecoder_RTHD`。

修改 `UMambaEnc_RTHD.__init__` 新增参数：

```python
use_skip_fusion_gate: bool = False
skip_gate_stages: List[int] = None
skip_gate_reduction: int = 4
use_boundary_attention_head: bool = False
use_frequency_refinement: bool = False
frequency_refinement_stages: List[int] = None
```

修改 `get_umamba_enc_rthd_3d_from_plans(...)` 新增同名参数，并放入 `kwargs['UMambaEnc_RTHD']`。

传给 `UNetResDecoder_RTHD`。

注意：

- 如果 `decoder_rthd_mode == "none"`，skip gate 和 boundary attention 是否启用需要谨慎。
- 推荐：只在使用 `UNetResDecoder_RTHD` 时支持这些增强。
- 如果用户设置了 `decoder_rthd_mode="none"` 但 `use_skip_fusion_gate=True`，可以打印 warning 并忽略，或者直接仍使用增强版 decoder 但 `rthd_stages_decoder=[]`。为了最小改动，建议打印 warning 并忽略。


## 八、实现任务 E：新增推荐 Trainer

新增一个 trainer 文件：

`umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder.py`

类名：

```python
class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder(nnUNetTrainer):
```

作用：

启用第二版推荐配置。

推荐配置：

```python
model = get_umamba_enc_rthd_3d_from_plans(
    plans_manager,
    dataset_json,
    configuration_manager,
    num_input_channels,
    deep_supervision=enable_deep_supervision,
    rthd_config_encoder={
        "view_mode": "tri",
        "share_weights": True,
        "scan_mode": "omni",
        "use_local_window": True,
        "window_size": 8,
        "reconstruction_mode": "gated",
        "cross_view_interaction": True,
        "interaction_mode": "post",
        "interaction_type": "gate",
    },
    rthd_config_decoder={
        "view_mode": "tri",
        "share_weights": True,
        "scan_mode": "omni",
        "use_local_window": False,
        "window_size": 8,
        "reconstruction_mode": "gated",
        "cross_view_interaction": True,
        "interaction_mode": "post",
        "interaction_type": "gate",
    },
    use_rthd_decoder=True,
    decoder_rthd_mode="partial",
    rthd_stages_decoder=[0, 1],
    use_skip_fusion_gate=True,
    skip_gate_stages=[0, 1],
    skip_gate_reduction=4,
    use_boundary_attention_head=True,
    use_frequency_refinement=True,
    frequency_refinement_stages=[0, 1],
)
```

说明：

- `decoder_rthd_mode="partial"` 是阶段感知部署核心
- 只在 D4/D3 使用 RTHD refinement
- 高分辨率 D2/D1 保持卷积
- skip gate 和 frequency refinement 也只默认作用于 D4/D3
- boundary attention 只作用于最终输出前

新增一个 350 epochs 包装类可选：

`nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_350epochs_patience50.py`

可以参考已有：

- `nnUNetTrainerUMambaEncRTHD_350epochs_patience50.py`
- `nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation.py`


## 九、测试要求

请新增或更新测试脚本：

`umamba/docs0602/script/test_stage_aware_decoder_strategy.py`

测试内容：

1. `SemanticSkipFusionGate3d`
   - 输入 skip/decoder shape 一致
   - 输入 skip/decoder shape 不一致时能自动对齐
   - 输出 shape 等于 skip shape

2. `BoundaryAttentionHead3d`
   - 输入输出 shape 一致

3. `HighLowFrequencyRefinement3d`
   - 输入输出 shape 一致

4. `UMambaEnc_RTHD`
   - `decoder_rthd_mode="partial"`
   - `use_skip_fusion_gate=True`
   - `use_boundary_attention_head=True`
   - `use_frequency_refinement=True`
   - 小尺寸随机输入 forward 成功
   - `deep_supervision=False` 输出是 tensor
   - `deep_supervision=True` 输出是 list

也请尽量跑已有测试：

- `umamba/docs0602/script/test_rthd_v1_enhancements.py`
- `umamba/docs0602/script/test_stage2_partial_decoder.py`

如果本地缺依赖导致不能完整运行，请在最终报告里明确说明是哪一个依赖缺失，而不是静默跳过。


## 十、实现细节注意事项

### 10.1 shape

所有新增模块都要假设输入是 3D 特征：

```python
(B, C, D, H, W)
```

不要写死 D/H/W。  
不要假设 patch size 固定。


### 10.2 nnU-Net 输出兼容

不要改变 decoder 的返回格式：

- `deep_supervision=False`: 返回 tensor
- `deep_supervision=True`: 返回 list

不要返回：

```python
(seg, boundary)
```

这会破坏现有 trainer。


### 10.3 compute_conv_feature_map_size

如果新增模块被加入 `self.stages` 这种会参与 `compute_conv_feature_map_size` 的路径，请确认不会导致：

```python
AttributeError: 'Sequential' object has no attribute 'compute_conv_feature_map_size'
```

如果必要，可以：

- 不把新增模块塞进 `self.stages`，而是单独用 `ModuleList`
- 或在 decoder 的 `compute_conv_feature_map_size` 中保守处理新增模块的额外开销

优先推荐单独 `ModuleList`，这样最稳。


### 10.4 代码风格

- 保持现有代码风格
- 不做大重构
- 不改无关文件
- 新增类写简短 docstring
- 新增参数在 docstring 中说明
- 新增 trainer 的 print 信息要清楚列出启用模块


## 十一、最终交付

请完成后给出：

1. 修改了哪些文件
2. 新增了哪些类和参数
3. 新 trainer 怎么运行
4. 测试结果
5. 如果有测试未跑成功，说明原因

目标最终模型名称建议：

`RTHD-StageAwareDecoder`

论文中可描述为：

`本文在解码阶段设计阶段感知的结构恢复策略，通过低分辨率 RTHD refinement 建模全局结构、语义引导跳连融合恢复局部细节，并结合边界注意力增强肿瘤轮廓质量。`
