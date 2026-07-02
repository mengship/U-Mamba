# nnUNetTrainerUMambaEncRTHD_NoCrossViewGate Code Review

## 问题分析

### ❌ 严重问题：解码器使用全局平铺会导致 OOM

**第 34 行**：
```python
decoder_config = {**common_config, "use_local_window": False}  # ❌ 问题
```

**配置详情**：
```python
decoder_rthd_mode="partial"
rthd_stages_decoder=[0, 1]  # Stage 0 (8³), Stage 1 (16³)
```

### 显存分析

虽然只在 stage 0, 1 使用 RTHD（比 Full Decoder 少），但：

- **Stage 0 (8³ × 512 channels)**：
  - 序列长度：8×8 = 64 tokens
  - 全局平铺可控 ✓

- **Stage 1 (16³ × 256 channels)**：
  - 序列长度：16×16 = 256 tokens
  - 全局平铺仍可控 ✓

**结论**：partial mode + stage [0,1] 的全局平铺**理论上不会 OOM**，因为这两个 stage 分辨率较低。

但为了**保险和一致性**，建议仍然启用局部窗口。

## 其他配置检查

### ✅ 正确的配置

1. **消融目标明确**：
   ```python
   "cross_view_interaction": False,  # C3 消融：关闭跨视图交互
   ```

2. **Partial Decoder 模式**：
   ```python
   decoder_rthd_mode="partial"
   rthd_stages_decoder=[0, 1]  # 只在低分辨率 stage
   ```

3. **Semantic Skip Gate 启用**：
   ```python
   use_skip_fusion_gate=True
   skip_gate_stages=[0, 1]
   skip_gate_type="semantic"
   ```

4. **编码器使用局部窗口**：
   ```python
   "use_local_window": True  # 编码器 ✓
   ```

## 风险等级评估

| 配置项 | 当前值 | 风险 | 原因 |
|--------|--------|------|------|
| decoder use_local_window | False | ⚠️ 中 | Stage 0,1 分辨率低，但不够稳妥 |
| decoder_rthd_mode | partial | ✅ 低 | 只用 2 个 stage |
| rthd_stages_decoder | [0, 1] | ✅ 低 | 只覆盖低分辨率 |
| cross_view_interaction | False | ✅ 低 | 符合消融设计 |

**总体风险**：⚠️ **中等**

- 在 Stage 0, 1 (8³, 16³) 上全局平铺**可能**不会 OOM
- 但如果其他因素（batch size, 编码器显存等）叠加，仍有风险
- 建议修改为 `use_local_window=True` 以保险

## 修复建议

### 推荐修复（保险起见）

```python
decoder_config = {**common_config, "use_local_window": True}  # 修改为 True
```

**理由**：
1. ✅ 消除潜在的显存风险
2. ✅ 与编码器保持一致
3. ✅ 不影响消融实验的有效性（消融点是 cross_view_interaction，不是 use_local_window）
4. ✅ 局部窗口是 LoMamba 的核心思想，应该全面使用

### 如果不修复

可以尝试直接训练，但需要：
1. 密切监控显存使用
2. 如果 OOM，立即修改配置

## 消融实验语义验证

### ✅ C3 消融设计正确

**目标**：验证跨视图交互门控（cross-view interaction gate）的作用

**对比**：
- **基线（最终方法）**：`cross_view_interaction=True`
- **C3 消融**：`cross_view_interaction=False`

**其他配置保持一致**：
- ✅ Partial decoder (stage 0, 1)
- ✅ Semantic skip gate
- ✅ 三视图 + 参数共享 + 全向扫描

**消融有效性**：✅ 正确，可以验证跨视图门控的贡献

## 修复后的代码

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_NoCrossViewGate(nnUNetTrainer):
    """
    C3 ablation: final ETSM pipeline without cross-view interaction gate.

    The encoder, stage-aware decoder, and semantic skip calibration are retained,
    but cross_view_interaction is disabled in both encoder and decoder ETSM blocks.
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) != 3:
            raise NotImplementedError("ETSM currently only supports 3D models")

        common_config = {
            "view_mode": "tri",
            "share_weights": True,
            "scan_mode": "omni",
            "window_size": 8,
            "reconstruction_mode": "gated",
            "cross_view_interaction": False,  # C3 消融点
        }
        encoder_config = {**common_config, "use_local_window": True}
        decoder_config = {**common_config, "use_local_window": True}  # ✓ 修复

        return get_umamba_enc_rthd_3d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            rthd_config_encoder=encoder_config,
            rthd_config_decoder=decoder_config,
            use_rthd_decoder=True,
            decoder_rthd_mode="partial",
            rthd_stages_decoder=[0, 1],
            use_skip_fusion_gate=True,
            skip_gate_stages=[0, 1],
            skip_gate_reduction=4,
            skip_gate_type="semantic",
            use_boundary_attention_head=False,
            use_frequency_refinement=False,
        )
```

## 训练建议

### 方案 1：先尝试当前配置（风险中等）

```bash
nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_NoCrossViewGate_150epochs
```

**如果 OOM**：修改配置（见方案 2）

### 方案 2：修改后训练（推荐）

修改 `use_local_window=True` 后训练，确保稳定性。

## 总结

| 项目 | 评估 |
|------|------|
| **逻辑正确性** | ✅ 正确（消融设计合理） |
| **显存风险** | ⚠️ 中等（decoder 全局平铺） |
| **建议修复** | ✅ 改为 `use_local_window=True` |
| **训练可行性** | ⚠️ 可能可行，但建议修复后训练 |

**推荐操作**：修改配置后再训练，避免潜在的 OOM 风险。
