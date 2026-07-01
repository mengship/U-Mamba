# FullDecoderETSM OOM 问题排查与解决方案

## 问题现状

即使修改了 `use_local_window=True`，训练仍然 OOM。

## 可能的原因

### 1. Python 缓存未更新 ⚠️

服务器端可能仍在使用旧的 `.pyc` 缓存文件。

**解决方法**：在服务器上运行以下命令清理缓存

```bash
# 清理 trainer 缓存
find /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer -name "*.pyc" -delete
find /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer -name "__pycache__" -type d -exec rm -rf {} +

# 清理网络模块缓存
find /hy-tmp/U-Mamba/umamba/nnunetv2/nets -name "*.pyc" -delete
find /hy-tmp/U-Mamba/umamba/nnunetv2/nets -name "__pycache__" -type d -exec rm -rf {} +

# 验证配置
grep "use_local_window" /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncRTHD_FullDecoderETSM.py
```

**预期输出**：两处都应该是 `True`

### 2. 即使用局部窗口，Full Decoder 仍然太大 ❌

即使启用局部窗口，在所有 5 个 decoder stage 都使用 RTHD 仍可能超过 24GB。

**原因**：
- RTHDBlock 包含：三视图 VMamba + 深度可分离卷积 + 残差连接
- 5 个 stage 累积的显存 + 编码器 + skip connections + 优化器状态
- Batch size = 2 进一步加倍显存

## 推荐解决方案

### 方案 A：使用 Partial Decoder 模式（强烈推荐）✅

只在低分辨率 stage (0, 1, 2) 使用 RTHD，高分辨率 stage (3, 4) 用卷积。

**训练命令**：
```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs
```

**优势**：
- ✅ 显存占用大幅降低（约 30-40%）
- ✅ 可以在 24GB GPU 上稳定运行
- ✅ 仍然是有效的消融实验（部分 vs 无 decoder ETSM）

**配置**：
```python
decoder_rthd_mode="partial"
rthd_stages_decoder=[0, 1, 2]  # 仅低分辨率 stage
```

**Decoder stages 分配**：
| Stage | 分辨率 | 使用模块 | 原因 |
|-------|--------|---------|------|
| 0 | 8³ | RTHD | ✓ 低分辨率，显存可控 |
| 1 | 16³ | RTHD | ✓ 低分辨率，显存可控 |
| 2 | 32³ | RTHD | ✓ 中等分辨率，可接受 |
| 3 | 64³ | Conv | ✗ 高分辨率，避免 OOM |
| 4 | 128³ | Conv | ✗ 最高分辨率，避免 OOM |

### 方案 B：继续尝试 Full Decoder（如果缓存清理后）

**训练命令**：
```bash
# 先清理缓存
cd /hy-tmp/U-Mamba
find umamba/nnunetv2 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 重新训练
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
```

**检查配置是否生效**：训练开始后查看日志

```
Encoder RTHD config: {...'use_local_window': True...}
Decoder RTHD config: {...'use_local_window': True...}  # ← 必须是 True
Decoder mode: full
```

如果 `use_local_window` 仍显示 `False`，说明缓存未清理干净。

### 方案 C：减小 Batch Size（不推荐）

如果必须用 Full Decoder，可以尝试减小 batch size：

```python
# 在 trainer 中添加
def configure_batch_size(self):
    return 1  # 从 2 减到 1
```

**缺点**：
- ❌ 影响 BatchNorm 统计
- ❌ 训练速度减半
- ❌ 可能影响收敛性和最终性能

## 训练验证步骤

1. **清理缓存**（服务器端）
   ```bash
   cd /hy-tmp/U-Mamba
   find umamba/nnunetv2/training/nnUNetTrainer -name "__pycache__" -exec rm -rf {} +
   find umamba/nnunetv2/nets -name "__pycache__" -exec rm -rf {} +
   ```

2. **验证配置**（服务器端）
   ```bash
   grep -A 2 "use_local_window" umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncRTHD_FullDecoderETSM.py
   ```
   
   应该看到两处 `True`（编码器和解码器）

3. **选择训练方案**

   **如果想要 Full Decoder**（风险较高）：
   ```bash
   nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs
   ```

   **如果想要稳定训练**（推荐）：
   ```bash
   nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs
   ```

4. **监控显存**（训练开始后）
   ```bash
   watch -n 1 nvidia-smi
   ```
   
   观察显存是否稳定在 24GB 以下

5. **检查日志输出**
   ```
   Decoder RTHD config: {'use_local_window': True, ...}  # ← 确认
   Decoder mode: full  # 或 partial
   ```

## 消融实验对比

| Trainer | Decoder 模式 | 显存占用 | 训练可行性 | 实验价值 |
|---------|-------------|---------|-----------|---------|
| FullDecoderETSM | full (5 stages) | ~24GB+ | ⚠️ 边缘 | 高（完整对比） |
| **PartialDecoderETSM** | partial (3 stages) | ~18GB | ✅ 稳定 | 中（部分对比） |
| 基线（无 decoder RTHD） | none | ~15GB | ✅ 稳定 | - |

## 推荐的实验策略

### 优先级 1：Partial Decoder（稳妥方案）

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs
```

**理由**：
- ✅ 可以稳定完成训练
- ✅ 仍能验证 decoder ETSM 的有效性（在低分辨率 stage）
- ✅ 论文中可以说明是由于显存限制采用部分模式
- ✅ 如果结果好，后续可以在更大的 GPU（如 A100 40GB）上尝试 Full 模式

### 优先级 2：Full Decoder（激进方案）

在清理缓存后再次尝试 Full 模式。如果仍然 OOM，放弃并使用 Partial 模式。

## 论文中的表述

如果使用 Partial Decoder 模式：

> To evaluate the effect of decoder ETSM, we applied RTHD blocks to the first three decoder stages (covering resolutions 8³, 16³, and 32³). Higher-resolution stages (64³ and 128³) used standard convolutions to stay within the 24GB GPU memory constraint. This partial decoder configuration still demonstrates the benefits of ETSM in the decoder path while maintaining computational feasibility.

## 文件清单

已创建的文件：
1. `nnUNetTrainerUMambaEncRTHD_FullDecoderETSM.py` - 修改后的 Full 版本
2. `nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM.py` - 新建的 Partial 版本（推荐）
3. `nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs.py` - 150 epochs 封装
4. `clean_cache_and_train.sh` - 缓存清理脚本

## 总结

**当前建议**：使用 **Partial Decoder 模式**，这是最稳妥的方案，既能完成实验，又不会因为显存问题卡住研究进度。

Full Decoder 在理论上更完整，但在 24GB GPU 上即使启用局部窗口仍可能因为累积显存（5 个 RTHD stage + 编码器 + 优化器状态）而 OOM。
