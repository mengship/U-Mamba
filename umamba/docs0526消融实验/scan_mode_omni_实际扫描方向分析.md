# scan_mode='omni' 实际扫描方向分析报告

## 执行摘要

**结论：scan_mode='omni' 实际执行 4 个扫描方向，而非 8 个。**

---

## 1. 代码追踪路径

### 1.1 参数传递链路

```
rthd_modules.py (TriViewVMambaBlock.__init__)
  ↓ scan_mode='omni'
  ↓ forward_type='v02' (当 scan_mode=='omni')
  ↓
vmamba.py (SS2D.__init__)
  ↓ forward_type='v02'
  ↓
vmamba.py (SS2Dv2.__initv2__)
  ↓ self.k_group = 4
  ↓ self.forward_core = FORWARD_TYPES['v02']
  ↓
vmamba.py (forward_corev2)
  ↓ scan_mode='cross2d' (默认)
  ↓ _scan_mode = 0
  ↓
vmamba.py (cross_scan_fwd with scans=0)
```

**关键代码位置：**
- [rthd_modules.py:342](../nnunetv2/nets/rthd_modules.py#L342): `forward_type=forward_type if scan_mode == 'omni' else 'v0'`
- [vmamba.py:1415](../instructions/vmamba.py#L1415): `self.k_group = 4`
- [vmamba.py:1520](../instructions/vmamba.py#L1520): `_scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None)`
- [vmamba.py:1536](../instructions/vmamba.py#L1536): `xs = cross_scan_fn(x, ..., scans=_scan_mode, ...)`

---

## 2. 扫描方向详解

### 2.1 scan_mode='omni' 的实际实现

当 `scan_mode='omni'` 时：
- 使用 `forward_type='v02'`（默认值）
- 调用 `SS2Dv2.__initv2__()` 初始化
- 设置 `self.k_group = 4`（**硬编码**）
- 在 `forward_corev2()` 中，默认 `scan_mode='cross2d'`，映射为 `_scan_mode=0`
- 调用 `cross_scan_fwd(x, scans=0)`

### 2.2 cross_scan_fwd(scans=0) 的 4 个扫描方向

**代码位置：[vmamba.py:44-48](../instructions/vmamba.py#L44-L48)**

```python
if scans == 0:
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)                          # 方向 0
    y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 方向 1
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])   # 方向 2, 3
```

**4 个扫描方向的具体含义：**

| 方向索引 | 扫描路径 | 代码实现 | 说明 |
|---------|---------|---------|------|
| **0** | **从左到右，从上到下** | `x.flatten(2, 3)` | 按行优先顺序展平 (H×W) |
| **1** | **从上到下，从左到右** | `x.transpose(2,3).flatten(2, 3)` | 转置后展平 (W×H) |
| **2** | **从右到左，从下到上** | `flip(方向0, dims=[-1])` | 方向 0 的反向 |
| **3** | **从下到上，从右到左** | `flip(方向1, dims=[-1])` | 方向 1 的反向 |

**可视化示例（4×4 特征图）：**

```
方向 0 (→↓):          方向 1 (↓→):          方向 2 (←↑):          方向 3 (↑←):
1  2  3  4           1  5  9  13          16 15 14 13          16 12  8  4
5  6  7  8           2  6  10 14          12 11 10  9          15 11  7  3
9  10 11 12          3  7  11 15           8  7  6  5          14 10  6  2
13 14 15 16          4  8  12 16           4  3  2  1          13  9  5  1
```

---

## 3. 对比：scan_mode='standard' (v0)

### 3.1 standard 模式的实现

当 `scan_mode='standard'` 时：
- 使用 `forward_type='v0'`
- 调用 `SS2Dv0.__initv0__()` 初始化
- 同样设置 `k_group = 4`（**硬编码**）
- 在 `forwardv0()` 中，手动构造 4 个扫描方向

**代码位置：[vmamba.py:1324-1325](../instructions/vmamba.py#L1324-L1325)**

```python
x_hwwh = torch.stack([x.view(B, -1, L), 
                      torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], 
                     dim=1).view(B, 2, -1, L)
xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
```

**结论：standard 模式也是 4 个扫描方向，与 omni 模式完全相同。**

---

## 4. 为什么不是 8 个方向？

### 4.1 k_group 的硬编码

在所有版本中，`k_group` 都被硬编码为 4：

- [vmamba.py:1267](../instructions/vmamba.py#L1267): `k_group = 4` (SS2Dv0)
- [vmamba.py:1415](../instructions/vmamba.py#L1415): `self.k_group = 4` (SS2Dv2)

### 4.2 cross_scan_fwd 的设计

`cross_scan_fwd` 函数支持多种扫描模式，但都是 4 个方向：

| scans 参数 | 扫描方向数 | 说明 |
|-----------|-----------|------|
| `scans=0` | **4** | cross2d：2个正向 + 2个反向 |
| `scans=1` | 4 | unidi：4个相同方向（退化） |
| `scans=2` | 4 | bidi：2个正向 + 2个反向 |
| `scans=3` | **4** | 4个旋转方向（0°, 90°, 180°, 270°） |

**代码位置：[vmamba.py:54-59](../instructions/vmamba.py#L54-L59)**

```python
elif scans == 3:
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)
    y[:, 1, :, :] = torch.rot90(x, 1, dims=(2, 3)).flatten(2, 3)  # 旋转90°
    y[:, 2, :, :] = torch.rot90(x, 2, dims=(2, 3)).flatten(2, 3)  # 旋转180°
    y[:, 3, :, :] = torch.rot90(x, 3, dims=(2, 3)).flatten(2, 3)  # 旋转270°
```

---

## 5. 潜在的混淆来源

### 5.1 论文描述 vs 代码实现

可能的混淆点：
1. **论文可能描述了 8 个方向**（包括对角线扫描），但代码实际只实现了 4 个方向
2. **"omni-directional"（全向）** 这个术语可能让人误以为是 8 个方向
3. **三视图 × 4 方向 = 12 次扫描**，可能被误解为单视图的 8 方向

### 5.2 实际的扫描次数

在 U-Mamba 的 RTHD 模块中：
- **单视图（axial/coronal/sagittal）**：4 个扫描方向
- **三视图模式**：3 个视图 × 4 个方向 = **12 次扫描**
- **参数共享模式**：12 次扫描共享同一套权重
- **独立参数模式**：12 次扫描使用 3 套独立权重

---

## 6. 代码一致性检查

### 6.1 ✅ 无 Bug 发现

经过详细检查，代码实现**逻辑一致**，未发现以下问题：
- ❌ 注释与代码不一致
- ❌ 方向重复
- ❌ 方向遗漏
- ❌ 参数传递错误

### 6.2 关键验证点

| 验证项 | 代码位置 | 结果 |
|-------|---------|------|
| k_group 初始化 | vmamba.py:1415 | ✅ 正确设置为 4 |
| scan_mode 映射 | vmamba.py:1520 | ✅ cross2d → 0 |
| cross_scan 调用 | vmamba.py:1536 | ✅ scans=0 |
| 方向数量 | vmamba.py:45 | ✅ (B, 4, C, H*W) |
| 方向定义 | vmamba.py:46-48 | ✅ 4 个不同方向 |
| merge 逻辑 | vmamba.py:91-93 | ✅ 正确合并 4 个方向 |

---

## 7. 最终结论

### 7.1 明确答案

**问题 1：scan_mode='omni' 是 4 个还是 8 个扫描方向？**
- **答案：4 个扫描方向**

**问题 2：分别是哪几个方向？**
- **方向 0**：从左到右，从上到下（行优先）
- **方向 1**：从上到下，从左到右（列优先）
- **方向 2**：从右到左，从下到上（方向 0 反向）
- **方向 3**：从下到上，从右到左（方向 1 反向）

**问题 3：关键代码证明**
- [vmamba.py:1415](../instructions/vmamba.py#L1415): `self.k_group = 4` （硬编码 4 个方向）
- [vmamba.py:44-48](../instructions/vmamba.py#L44-L48): `cross_scan_fwd(scans=0)` 的 4 个方向实现
- [vmamba.py:1536](../instructions/vmamba.py#L1536): 实际调用 `cross_scan_fn(..., scans=0)`

**问题 4：是否存在 Bug？**
- **否**，代码实现逻辑正确，无 Bug

---

## 8. 补充说明

### 8.1 为什么叫 "omni-directional"？

虽然只有 4 个方向，但这 4 个方向覆盖了：
- 2 个正交方向（水平、垂直）
- 每个方向的双向扫描（正向、反向）

这种设计能够捕获 2D 特征图的**全局上下文依赖**，因此称为"全向扫描"。

### 8.2 与 Vision Mamba 的关系

这个实现来自 **VMamba (Vision Mamba)** 论文：
- 论文中明确使用 **4 个扫描方向**
- 代码实现与论文描述一致
- U-Mamba 直接复用了 VMamba 的 SS2D 模块

---

## 9. 参考代码位置汇总

| 文件 | 行号 | 内容 |
|-----|------|------|
| rthd_modules.py | 342 | `forward_type=forward_type if scan_mode == 'omni' else 'v0'` |
| vmamba.py | 1415 | `self.k_group = 4` |
| vmamba.py | 1520 | `_scan_mode = dict(cross2d=0, ...)` |
| vmamba.py | 1536 | `xs = cross_scan_fn(x, ..., scans=_scan_mode)` |
| vmamba.py | 44-48 | `cross_scan_fwd(scans=0)` 的 4 个方向实现 |
| vmamba.py | 91-93 | `cross_merge_fwd(scans=0)` 的 4 个方向合并 |
| vmamba.py | 1324-1325 | `forwardv0()` 的 4 个方向构造（standard 模式） |

---

**报告生成时间：** 2026-05-30  
**分析者：** Claude (Kiro)  
**代码版本：** U-Mamba (基于 VMamba v2)
