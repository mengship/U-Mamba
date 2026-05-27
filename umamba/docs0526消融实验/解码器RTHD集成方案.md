# 解码器 RTHD 集成方案

## 🎯 目标

将编码器的 RTHD（三视图递归分解）逻辑同步到解码器，实现完整的编码器-解码器 RTHD 架构。

---

## 📊 当前状态分析

### 编码器（已修改）

```python
class ResidualMambaEncoder_RTHD:
    - Stage 0-4: 可选使用 RTHDBlock 或 MambaLayer
    - 支持消融实验配置（rthd_config）
    - 参数：
      - view_mode: 'tri' / 'single'
      - share_weights: True / False
      - scan_mode: 'omni' / 'standard'
      - use_local_window: True / False
```

### 解码器（未修改）

```python
class UNetResDecoder:
    - 使用原始的 BasicResBlock + BasicBlockD
    - 没有 Mamba 或 RTHD
    - 只有卷积 + 归一化 + 激活
```

**问题**：编码器使用了先进的 RTHD 机制，但解码器还是传统卷积，不对称。

---

## 🔧 设计方案

### 方案 1: 完全对称（推荐）

**思路**：解码器也使用 RTHD，与编码器完全对称

```python
class UNetResDecoder_RTHD:
    def __init__(self, encoder, ..., use_rthd=True, rthd_stages=None, rthd_config=None):
        # 解码器的每个 stage 也可以选择使用 RTHD
        for s in range(n_stages_decoder):
            if s in rthd_stages:
                # 使用 RTHDBlock
                self.stages[s] = nn.Sequential(
                    BasicResBlock(...),  # 初始融合 skip connection
                    RTHDBlock(...),      # RTHD 处理
                    *[BasicBlockD(...) for _ in range(n_conv-1)]
                )
            else:
                # 使用原始卷积
                self.stages[s] = nn.Sequential(
                    BasicResBlock(...),
                    *[BasicBlockD(...) for _ in range(n_conv-1)]
                )
```

**优势**：
- ✅ 编码器-解码器完全对称
- ✅ 解码器也能享受 RTHD 的优势
- ✅ 可以进行解码器的消融实验

**劣势**：
- ❌ 计算量增加
- ❌ 实现复杂度较高

---

### 方案 2: 部分集成

**思路**：只在解码器的浅层（大特征图）使用 RTHD

```python
class UNetResDecoder_RTHD:
    # 解码器 Stage 0-1: 使用 RTHD（特征图较大）
    # 解码器 Stage 2-3: 使用原始卷积（特征图较小）
    
    rthd_stages_decoder = [0, 1]  # 只在前两个 stage 使用
```

**优势**：
- ✅ 平衡性能和计算量
- ✅ 在大特征图上使用 RTHD 效果更好
- ✅ 实现相对简单

**劣势**：
- ⚖️ 不完全对称

---

### 方案 3: 轻量级集成

**思路**：解码器使用简化版的 RTHD（只用 MambaLayer，不用完整的 RTHDBlock）

```python
class UNetResDecoder_Mamba:
    # 在解码器中插入 MambaLayer
    for s in range(n_stages_decoder):
        self.stages[s] = nn.Sequential(
            BasicResBlock(...),
            MambaLayer(...),  # 简单的 Mamba，不是 RTHD
            *[BasicBlockD(...) for _ in range(n_conv-1)]
        )
```

**优势**：
- ✅ 实现简单
- ✅ 计算量适中
- ✅ 仍然比纯卷积强

**劣势**：
- ❌ 不是真正的 RTHD
- ❌ 与编码器不对称

---

## 💡 推荐方案：方案 1（完全对称）

### 理由

1. **理论完整性**：编码器-解码器对称是 U-Net 的核心思想
2. **性能最优**：解码器也能享受 RTHD 的三视图融合优势
3. **消融实验价值**：可以对比"仅编码器 RTHD" vs "编码器+解码器 RTHD"
4. **论文价值**：完整的 RTHD 架构更有说服力

---

## 🔨 实现细节

### 1. 创建 UNetResDecoder_RTHD

```python
class UNetResDecoder_RTHD(nn.Module):
    """
    集成 RTHD 的解码器
    与编码器对称，在指定的 stage 使用 RTHDBlock
    """
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 use_rthd: bool = True,  # 新增
                 rthd_stages: List[int] = None,  # 新增
                 rthd_config: dict = None,  # 新增
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        # 决定解码器哪些 stage 使用 RTHD
        if rthd_stages is None:
            # 默认：与编码器对称
            # 如果编码器 stage 0,1,2 用 RTHD，解码器也在对应位置用
            rthd_stages = encoder.rthd_stages if hasattr(encoder, 'rthd_stages') else []
        
        self.use_rthd = use_rthd
        self.rthd_stages = rthd_stages
        self.rthd_config = rthd_config or {}
        
        print(f"Decoder RTHD enabled for stages: {rthd_stages}")
        
        stages = []
        upsample_layers = []
        seg_layers = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            
            # 上采样层
            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))
            
            # 决定这个 stage 是否使用 RTHD
            decoder_stage_idx = s - 1  # 解码器 stage 索引
            use_rthd_this_stage = use_rthd and (decoder_stage_idx in rthd_stages)
            
            if use_rthd_this_stage:
                # 使用 RTHD
                stage_blocks = [
                    BasicResBlock(
                        conv_op = encoder.conv_op,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                        input_channels = 2 * input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        padding=encoder.conv_pad_sizes[-(s + 1)],
                        stride=1,
                        use_1x1conv=True
                    ),
                    RTHDBlock(
                        dim=input_features_skip,
                        **self.rthd_config  # 传递消融实验配置
                    ),
                    *[
                        BasicBlockD(
                            conv_op = encoder.conv_op,
                            input_channels = input_features_skip,
                            output_channels = input_features_skip,
                            kernel_size = encoder.kernel_sizes[-(s + 1)],
                            stride = 1,
                            conv_bias = encoder.conv_bias,
                            norm_op = encoder.norm_op,
                            norm_op_kwargs = encoder.norm_op_kwargs,
                            nonlin = encoder.nonlin,
                            nonlin_kwargs = encoder.nonlin_kwargs,
                        ) for _ in range(n_conv_per_stage[s-1] - 2)  # -2 因为已经有 BasicResBlock 和 RTHDBlock
                    ]
                ]
            else:
                # 使用原始卷积
                stage_blocks = [
                    BasicResBlock(
                        conv_op = encoder.conv_op,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                        input_channels = 2 * input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        padding=encoder.conv_pad_sizes[-(s + 1)],
                        stride=1,
                        use_1x1conv=True
                    ),
                    *[
                        BasicBlockD(
                            conv_op = encoder.conv_op,
                            input_channels = input_features_skip,
                            output_channels = input_features_skip,
                            kernel_size = encoder.kernel_sizes[-(s + 1)],
                            stride = 1,
                            conv_bias = encoder.conv_bias,
                            norm_op = encoder.norm_op,
                            norm_op_kwargs = encoder.norm_op_kwargs,
                            nonlin = encoder.nonlin,
                            nonlin_kwargs = encoder.nonlin_kwargs,
                        ) for _ in range(n_conv_per_stage[s-1] - 1)
                    ]
                ]
            
            stages.append(nn.Sequential(*stage_blocks))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        
        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
    
    def forward(self, skips):
        # 与原始解码器相同
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        
        seg_outputs = seg_outputs[::-1]
        
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
```

### 2. 修改 UMambaEnc_RTHD

```python
class UMambaEnc_RTHD(nn.Module):
    def __init__(self, ..., 
                 use_rthd_decoder: bool = True,  # 新增
                 rthd_stages_decoder: List[int] = None,  # 新增
                 ):
        # ... 编码器初始化 ...
        
        # 创建解码器
        if use_rthd_decoder:
            self.decoder = UNetResDecoder_RTHD(
                self.encoder,
                num_classes,
                n_conv_per_stage_decoder,
                deep_supervision,
                use_rthd=True,
                rthd_stages=rthd_stages_decoder,
                rthd_config=rthd_config,
            )
        else:
            # 使用原始解码器
            self.decoder = UNetResDecoder(
                self.encoder,
                num_classes,
                n_conv_per_stage_decoder,
                deep_supervision,
            )
```

---

## 🧪 新的消融实验

### 实验 #9: 仅编码器 RTHD（当前）

```python
rthd_config = {
    'view_mode': 'tri',
    'share_weights': True,
    'scan_mode': 'omni',
    'use_local_window': True,
}

use_rthd_encoder = True
use_rthd_decoder = False  # 解码器不用 RTHD
```

**目的**：当前的基线

### 实验 #10: 编码器+解码器 RTHD（新增）

```python
rthd_config = {
    'view_mode': 'tri',
    'share_weights': True,
    'scan_mode': 'omni',
    'use_local_window': True,
}

use_rthd_encoder = True
use_rthd_decoder = True  # 解码器也用 RTHD
rthd_stages_decoder = [0, 1, 2]  # 解码器前3个stage
```

**目的**：验证解码器 RTHD 的价值

**预期**：
- 性能提升 1-2%
- 计算量增加约 50%（解码器也用 RTHD）

---

## 📊 对比表格

| 配置 | 编码器 | 解码器 | 预期性能 | 计算量 |
|-----|-------|--------|---------|--------|
| **原始 U-Mamba** | MambaLayer | 卷积 | 基线 | 1× |
| **RTHD-Enc** | RTHD | 卷积 | 基线+3% | 1.2× |
| **RTHD-Full** | RTHD | RTHD | 基线+4~5% | 1.5× |

---

## 🎯 实现优先级

### 短期（当前论文）

1. **先完成现有的 6 个消融实验**（仅编码器 RTHD）
2. **在论文的"讨论"中提到解码器 RTHD 的可能性**

### 中期（论文修订）

1. **实现 UNetResDecoder_RTHD**
2. **运行实验 #10（编码器+解码器 RTHD）**
3. **如果性能提升明显**：
   - 在修订版中加入这个实验
   - 更新论文的架构图

### 长期（后续研究）

1. **探索不对称的 RTHD 配置**：
   - 编码器用全向扫描，解码器用定向扫描
   - 编码器用固定窗口，解码器用滑动窗口
2. **探索解码器专用的视图融合策略**

---

## 💡 关键考虑

### 1. 解码器的特殊性

**与编码器的区别**：
- 解码器有 skip connection（跳跃连接）
- 解码器是上采样过程
- 解码器的特征图逐渐变大

**RTHD 在解码器中的作用**：
- 融合来自编码器的 skip features
- 在上采样后的大特征图上进行三视图建模
- 恢复空间细节

### 2. 计算量权衡

**解码器 RTHD 的计算开销**：
```
假设编码器 5 个 stage，解码器 4 个 stage
如果都用 RTHD：
- 编码器：5 个 RTHD block
- 解码器：4 个 RTHD block
- 总计：9 个 RTHD block

相比仅编码器 RTHD：
- 计算量增加约 80%（4/5）
```

**优化策略**：
- 只在解码器的前 2-3 个 stage 使用 RTHD
- 解码器使用更轻量的 RTHD 配置（如更小的 window_size）

### 3. 对称性的价值

**U-Net 的核心思想**：编码器-解码器对称

**RTHD 的对称性**：
- 编码器：3D → 2D 视图 → 特征提取 → 融合回 3D
- 解码器：3D → 2D 视图 → 特征细化 → 融合回 3D

**论文叙述**：
> "为了保持 U-Net 的对称性，我们在解码器中也采用了 RTHD 机制。解码器的 RTHD 不仅融合了来自编码器的 skip features，还通过三视图分解在上采样后的大特征图上进行精细的空间建模。"

---

## 📝 实现步骤

### Step 1: 创建 UNetResDecoder_RTHD 类

在 `UMambaEnc_RTHD.py` 中添加新的解码器类

### Step 2: 修改 UMambaEnc_RTHD 类

添加 `use_rthd_decoder` 和 `rthd_stages_decoder` 参数

### Step 3: 更新 get_umamba_enc_rthd_3d_from_plans

支持解码器 RTHD 的配置

### Step 4: 创建新的 Trainer

```python
class nnUNetTrainerUMambaEncRTHD_Ablation10_FullRTHD(nnUNetTrainer):
    """
    消融实验 #10: 编码器+解码器完整 RTHD
    """
    @staticmethod
    def build_network_architecture(...):
        rthd_config = {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
            'use_local_window': True,
        }
        
        return get_umamba_enc_rthd_3d_from_plans(
            ...,
            use_rthd=True,
            rthd_stages=[0, 1, 2, 3, 4],
            rthd_config=rthd_config,
            use_rthd_decoder=True,  # 新增
            rthd_stages_decoder=[0, 1, 2],  # 新增
        )
```

### Step 5: 测试和验证

```bash
# 测试新的解码器
python test_decoder_rthd.py

# 运行训练
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_Ablation10_FullRTHD
```

---

**创建时间**: 2026-05-27  
**作者**: Claude (Kiro)  
**目的**: 将编码器的 RTHD 逻辑同步到解码器
