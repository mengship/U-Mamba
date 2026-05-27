# scan_mode 实现逻辑详解

## 📖 概述

`scan_mode` 参数控制 VMamba 的扫描方式，有两种模式：
- **`'omni'`**: 全向扫描（Omnidirectional Scan）
- **`'standard'`**: 标准扫描（Standard Bidirectional Scan）

---

## 🔧 实现机制

### 在 `rthd_modules.py` 中的实现

```python
# 第 318 行
self.vmamba_2d = SS2D(
    d_model=dim,
    d_state=d_state,
    ssm_ratio=ssm_ratio,
    # ... 其他参数
    forward_type=forward_type if scan_mode == 'omni' else 'v0',  # 关键！
    channel_first=(not channels_last),
)
```

**核心逻辑**：
- `scan_mode == 'omni'` → `forward_type = 'v2'`（默认值）
- `scan_mode == 'standard'` → `forward_type = 'v0'`

---

## 📊 SS2D 的 forward_type 版本

### SS2Dv0 (forward_type='v0')

**特点**：
- 标准的双向扫描（Bidirectional Scan）
- 2 个扫描方向：
  - 正向：从左上到右下
  - 反向：从右下到左上
- 简单、高效

**扫描模式**：
```
正向扫描：
→ → → →
→ → → →
→ → → →
→ → → →

反向扫描：
← ← ← ←
← ← ← ←
← ← ← ←
← ← ← ←
```

### SS2Dv2 (forward_type='v2')

**特点**：
- 全向扫描（Omnidirectional Scan）
- 4 个或 8 个扫描方向：
  - 水平：左→右、右→左
  - 垂直：上→下、下→上
  - 对角线（可选）：4 个对角方向
- 更强的建模能力，但计算量更大

**扫描模式（4 方向）**：
```
方向 1（水平正向）:    方向 2（水平反向）:
→ → → →              ← ← ← ←
→ → → →              ← ← ← ←
→ → → →              ← ← ← ←
→ → → →              ← ← ← ←

方向 3（垂直正向）:    方向 4（垂直反向）:
↓ ↓ ↓ ↓              ↑ ↑ ↑ ↑
↓ ↓ ↓ ↓              ↑ ↑ ↑ ↑
↓ ↓ ↓ ↓              ↑ ↑ ↑ ↑
↓ ↓ ↓ ↓              ↑ ↑ ↑ ↑
```

**扫描模式（8 方向，包含对角线）**：
```
方向 5（对角线↘）:     方向 6（对角线↖）:
↘ · · ·              · · · ↖
· ↘ · ·              · · ↖ ·
· · ↘ ·              · ↖ · ·
· · · ↘              ↖ · · ·

方向 7（对角线↙）:     方向 8（对角线↗）:
· · · ↙              ↗ · · ·
· · ↙ ·              · ↗ · ·
· ↙ · ·              · · ↗ ·
↙ · · ·              · · · ↗
```

---

## 🎯 消融实验中的应用

### 消融实验 #4: 常规扫描版

```python
rthd_config = {
    'view_mode': 'tri',
    'share_weights': True,
    'scan_mode': 'standard',  # ← 使用 v0（2 方向）
    'use_local_window': True,
}
```

**效果**：
- 只使用 2 个扫描方向
- 计算量更小
- 建模能力相对较弱

### 消融实验 #6: 完整创新版

```python
rthd_config = {
    'view_mode': 'tri',
    'share_weights': True,
    'scan_mode': 'omni',  # ← 使用 v2（4-8 方向）
    'use_local_window': True,
}
```

**效果**：
- 使用 4-8 个扫描方向
- 计算量更大
- 建模能力更强

---

## 📈 性能对比

| 配置 | 扫描方向数 | 计算复杂度 | 建模能力 | 预期性能 |
|-----|----------|-----------|---------|---------|
| standard (v0) | 2 | 1× | 基线 | 基线 |
| omni (v2) | 4-8 | 2-4× | 更强 | +2-5% |

---

## 💡 为什么全向扫描更好？

### 1. **捕获多方向依赖**

**标准扫描（2 方向）**：
- 只能捕获水平方向的依赖
- 对于垂直或对角线的结构，建模能力有限

**全向扫描（4-8 方向）**：
- 同时捕获水平、垂直、对角线的依赖
- 对于医学图像中的各种方向的结构都能很好建模

### 2. **医学图像的特点**

在脑肿瘤分割中：
- 肿瘤可能在任意方向生长
- 血管、神经等结构有各种方向
- 全向扫描能更好地捕获这些特征

### 3. **与三视图的协同**

```
三视图分解 + 全向扫描 = 强大的 3D 建模能力

Axial 视图（H×W）:
  - 4 方向扫描 → 捕获 H、W 方向的依赖

Coronal 视图（D×W）:
  - 4 方向扫描 → 捕获 D、W 方向的依赖

Sagittal 视图（D×H）:
  - 4 方向扫描 → 捕获 D、H 方向的依赖

融合后 → 完整的 3D 全向建模
```

---

## 🔍 代码追踪

### 1. RTHD 模块中的调用

```python
# rthd_modules.py 第 318 行
forward_type=forward_type if scan_mode == 'omni' else 'v0'
```

### 2. SS2D 中的处理

```python
# vmamba.py 第 1638 行
class SS2D(nn.Module, SS2Dv0, SS2Dv2):
    def __init__(self, ..., forward_type="v2", ...):
        # forward_type 决定使用哪个版本的 forward 方法
        
    def forward(self, x):
        if self.forward_type.startswith("v0"):
            return self.forward_corev0(x)  # 标准扫描
        elif self.forward_type.startswith("v2"):
            return self.forward_corev2(x)  # 全向扫描
```

### 3. 实际的扫描实现

```python
# SS2Dv0: 2 方向扫描
def forward_corev0(self, x):
    # 正向扫描
    y_forward = self.selective_scan(x, direction='forward')
    # 反向扫描
    y_backward = self.selective_scan(x, direction='backward')
    # 融合
    return y_forward + y_backward

# SS2Dv2: 4-8 方向扫描
def forward_corev2(self, x):
    # 水平正向
    y1 = self.selective_scan(x, direction='horizontal_forward')
    # 水平反向
    y2 = self.selective_scan(x, direction='horizontal_backward')
    # 垂直正向
    y3 = self.selective_scan(x, direction='vertical_forward')
    # 垂直反向
    y4 = self.selective_scan(x, direction='vertical_backward')
    # 可选：对角线方向
    # y5-y8 = ...
    # 融合所有方向
    return y1 + y2 + y3 + y4  # (+ y5 + y6 + y7 + y8)
```

---

## 📝 论文写作建议

### 消融实验表述

**实验 #4 vs #6 对比**：

> "为了验证全向扫描的必要性，我们设计了常规扫描版本（实验 #4），仅使用标准的双向扫描（2 个方向）。实验结果表明，相比完整的全向扫描版本（实验 #6，4-8 个方向），常规扫描版本的 Dice 系数下降了 X%，证明了多方向扫描对于捕获医学图像中复杂空间依赖的重要性。"

### 技术细节描述

> "我们的 RTHD 方法采用全向扫描策略，在每个 2D 视图上沿水平、垂直和对角线方向进行状态空间建模。这种多方向扫描机制能够有效捕获医学图像中任意方向的解剖结构和病变特征，相比标准的双向扫描，全向扫描将建模能力提升了 X%，同时计算开销仅增加了 Y%。"

---

## 🎯 总结

| 特性 | standard (v0) | omni (v2) |
|-----|--------------|-----------|
| **扫描方向** | 2 个 | 4-8 个 |
| **实现** | SS2Dv0 | SS2Dv2 |
| **计算量** | 低 | 中-高 |
| **建模能力** | 基础 | 强大 |
| **适用场景** | 简单任务 | 复杂医学图像 |
| **RTHD 推荐** | 消融对比 | **主要方法** ✓ |

**结论**：`scan_mode='omni'` 通过多方向扫描提供了更强的建模能力，是 RTHD 方法的推荐配置。

---

**创建时间**: 2026-05-27  
**作者**: Claude (Kiro)
