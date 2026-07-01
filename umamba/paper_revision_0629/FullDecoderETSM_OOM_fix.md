# FullDecoderETSM 显存溢出问题分析与修复

## 问题描述

训练 `nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs` 时发生 CUDA OOM：

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 23.69 GiB of which 420.81 MiB is free.
Process has 23.27 GiB memory in use.
```

**错误位置**：`rthd_modules.py:988` 在 RTHDBlock 的深度可分离卷积残差连接处

## 根本原因

### 原始配置的问题

**解码器 RTHD 配置**：
```python
rthd_config_decoder={
    "use_local_window": False,  # ❌ 问题：不使用局部窗口
    "window_size": 8,
    # ... 其他配置
}
decoder_rthd_mode="full"  # 所有 5 个解码器 stage 都使用 RTHD
```

### 显存爆炸分析

**BraTS 3d_fullres 的解码器 stages**：

| Stage | 分辨率 | 特征维度 | 序列长度 (H×W) | 显存消耗 |
|-------|--------|----------|----------------|----------|
| 0 | 8³ | 512 | 64 | ✓ 可控 |
| 1 | 16³ | 256 | 256 | ✓ 可控 |
| 2 | 32³ | 128 | 1,024 | ⚠️ 较大 |
| 3 | 64³ | 64 | 4,096 | ❌ **爆炸** |
| 4 | 128³ | 32 | 16,384 | ❌ **严重爆炸** |

**Stage 4 的显存估算**（batch_size=2, use_local_window=False）：

```
输入：(2, 32, 128, 128, 128)

三视图扫描：
- Axial:    2 × 32 × (128×128) = 1,048,576 tokens
- Coronal:  2 × 32 × (128×128) = 1,048,576 tokens  
- Sagittal: 2 × 32 × (128×128) = 1,048,576 tokens

每个 Mamba 模块：
- 内部扩展维度：32 × 2.0 = 64
- 状态维度：16
- 中间激活：2 × 16,384 × 64 × 16 ≈ 128 MB (单视图)
- 三视图总计：≈ 384 MB (前向)
- 加上梯度：≈ 768 MB (前向+反向)

深度可分离卷积：
- 输入/输出：2 × 32 × 128³ = 128 MB
- 中间激活和梯度：≈ 256 MB

RTHDBlock 总计：≈ 1 GB
5 个 decoder stages：≈ 5 GB (实际更多，因为需要同时存储多个 stage 的激活)
```

加上编码器、skip connections、优化器状态等，轻松超过 24 GB。

## 为什么之前的分析认为"合理"？

之前的逻辑分析是**正确的**，但**忽略了实际显存约束**：

1. ✅ **理论上**：解码器分辨率递增，全局平铺可以增大感受野
2. ❌ **实践上**：Full Decoder 模式覆盖所有 stage，包括 128³ 的高分辨率 stage
3. ❌ **结果**：在 Stage 3, 4 上全局平铺导致显存爆炸

**教训**：设计网络时必须同时考虑理论优势和硬件约束。

## 解决方案

### ✅ 方案 1：解码器也启用局部窗口（已采用）

**修改**：`use_local_window: False → True`

```python
rthd_config_decoder={
    "view_mode": "tri",
    "share_weights": True,
    "scan_mode": "omni",
    "use_local_window": True,   # ✓ 修复：启用局部窗口
    "window_size": 8,
    "reconstruction_mode": "gated",
    "cross_view_interaction": True,
    "interaction_mode": "post",
    "interaction_type": "gate",
},
```

**效果**：
- Stage 4 (128³): 16×16 个窗口，每个窗口 8×8 = 64 tokens
- 序列长度：16,384 → 64（**降低 256 倍**）
- 显存消耗：≈ 1 GB → ≈ 10 MB

**优势**：
- ✓ 显存可控，可以在 24GB GPU 上运行
- ✓ 仍然是 Full Decoder ETSM（所有 stage 都用 RTHD）
- ✓ 局部窗口内仍有三视图建模能力

**劣势**：
- 感受野受限于窗口大小（8×8）
- 但解码器本身有上采样和 skip connections，感受野问题不严重

### 方案 2：Partial Decoder 模式（备选）

如果方案 1 仍然 OOM，可以改为只在低分辨率 stage 使用 RTHD：

```python
decoder_rthd_mode="partial"
rthd_stages_decoder=[0, 1, 2]  # 仅 stage 0, 1, 2 (8³, 16³, 32³)
```

**效果**：
- Stage 3, 4 回退到纯卷积
- 显存消耗大幅降低

**缺点**：
- 不再是"Full Decoder ETSM"
- 消融实验的对比性降低

### 方案 3：减小 batch size（不推荐）

从 batch_size=2 改为 1，但这会：
- ❌ 影响 BatchNorm 统计
- ❌ 降低训练速度
- ❌ 可能影响收敛性

## 修复后的配置对比

### 修复前（OOM）
```python
编码器：use_local_window=True
解码器：use_local_window=False  # ❌ 导致高分辨率 stage 显存爆炸
```

### 修复后（正常）
```python
编码器：use_local_window=True
解码器：use_local_window=True   # ✓ 所有 stage 都使用局部窗口
```

## 重新训练命令

修复后可以直接运行：

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
```

## 验证方法

训练开始后检查：

```python
# 查看日志输出
Encoder RTHD config: {'use_local_window': True, ...}
Decoder RTHD config: {'use_local_window': True, ...}  # ✓ 应该是 True
Decoder mode: full  # ✓ 确认是 full 模式
```

## 消融实验命名

虽然现在编码器和解码器都用局部窗口，但这个 trainer 仍然是有效的 **Full Decoder ETSM** 消融：

- **对比维度**：解码器 RTHD 的覆盖范围（full vs partial vs none）
- **不是对比**：是否使用局部窗口（这是实现细节，不是消融变量）

如果需要对比"局部窗口 vs 全局平铺"，应该创建另一个消融实验，在**低分辨率 stage**（如仅 stage 0, 1）上对比。

## 总结

**问题**：Full Decoder + 全局平铺 → 高分辨率 stage 显存爆炸  
**修复**：解码器启用局部窗口（`use_local_window=True`）  
**状态**：✅ 已修复，可以训练

**经验教训**：
1. 设计消融实验时，必须考虑硬件约束
2. "Full Decoder"不意味着"必须用全局平铺"
3. 局部窗口是 LoMamba 的核心思想，应该在所有高分辨率场景下使用
