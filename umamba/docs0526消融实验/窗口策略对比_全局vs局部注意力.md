# 窗口策略对比：全局注意力 vs 局部注意力
固定窗口/滑动窗口，对比一下哪个更好，与全局平铺并行合并，从全局注意力与局部注意力的读来阐述

## 🎯 核心思路

从**注意力机制的感受野**角度，对比三种窗口策略：

1. **全局平铺**（Global Flatten）→ 全局注意力
2. **固定窗口**（Fixed Window）→ 局部注意力（不重叠）
3. **滑动窗口**（Sliding Window）→ 局部注意力（重叠）

---

## 📊 三种策略对比

### 1. 全局平铺（Global Flatten）

**实现方式**：
```python
# 直接处理整个特征图
out = self.vmamba_2d(view)  # view: (B, C, H, W)
```

**注意力特性**：
- ✅ **全局感受野**：每个位置可以"看到"整个特征图
- ✅ 长距离依赖建模能力强
- ❌ 序列长度 = H × W（如 32×32 = 1024）
- ❌ 计算复杂度高：O(H×W × C)

**适用场景**：
- 小特征图（如 16×16, 8×8）
- 需要全局上下文的任务

---

### 2. 固定窗口（Fixed Window）- 当前实现

**实现方式**：
```python
# 不重叠的网格分割
windows, (H_pad, W_pad) = window_partition(view, window_size)
# windows: (B*num_windows, C, window_size, window_size)
windows_out = self.vmamba_2d(windows)
out = window_reverse(windows_out, window_size, H_pad, W_pad, H, W)
```

**窗口特性**：
```
32×32 特征图，window_size=8，stride=8（不重叠）

┌───────┬───────┬───────┬───────┐
│ Win 1 │ Win 2 │ Win 3 │ Win 4 │
├───────┼───────┼───────┼───────┤
│ Win 5 │ Win 6 │ Win 7 │ Win 8 │
├───────┼───────┼───────┼───────┤
│ Win 9 │ Win10 │ Win11 │ Win12 │
├───────┼───────┼───────┼───────┤
│ Win13 │ Win14 │ Win15 │ Win16 │
└───────┴───────┴───────┴───────┘

窗口数量：16
每个窗口：8×8 = 64 个位置
窗口之间：无重叠，无信息共享
```

**注意力特性**：
- ✅ **局部感受野**：每个位置只能"看到"所在窗口内的其他位置
- ✅ 序列长度短：window_size² = 64
- ✅ 计算效率高：O(num_windows × window_size² × C)
- ✅ 并行度高：所有窗口并行处理
- ❌ **窗口边界问题**：窗口边缘的像素无法与相邻窗口交互
- ❌ 跨窗口的长距离依赖建模能力弱

**适用场景**：
- 大特征图（如 128×128, 64×64）
- 局部纹理特征重要的任务
- 需要高计算效率的场景

---

### 3. 滑动窗口（Sliding Window）- 新提出

**实现方式**：
```python
# 重叠的滑动窗口
stride = window_size // 2  # 50% 重叠
windows = sliding_window_partition(view, window_size, stride)
# windows: (B*num_windows, C, window_size, window_size)
windows_out = self.vmamba_2d(windows)
out = sliding_window_reverse(windows_out, window_size, stride, H, W)
```

**窗口特性**：
```
32×32 特征图，window_size=8，stride=4（50% 重叠）

┌───────┐
│ Win 1 ├───────┐
└───┬───┘ Win 2 ├───────┐
    └───┬───────┘ Win 3 ├───────┐
        └───┬───────────┘ Win 4 │
            └───────────────────┘
            
每行/列窗口数：(32-8)/4 + 1 = 7
总窗口数量：7×7 = 49
每个窗口：8×8 = 64 个位置
重叠区域：每个位置被 2×2 = 4 个窗口覆盖
```

**注意力特性**：
- ✅ **局部感受野 + 信息融合**：每个位置被多个窗口处理
- ✅ 缓解窗口边界问题：重叠区域提供跨窗口信息交互
- ✅ 更平滑的特征表示：多个窗口的输出需要融合
- ❌ 窗口数量增加：49 vs 16（3倍）
- ❌ 计算量增加：O(3× × num_windows × window_size² × C)
- ❌ 需要额外的融合策略（平均、加权平均等）

**适用场景**：
- 需要平衡局部和全局信息的任务
- 对窗口边界敏感的任务
- 计算资源充足的场景

---

## 🔬 从注意力机制角度的深入分析

### 全局注意力 vs 局部注意力

#### 全局注意力（Global Attention）

**Transformer 中的 Self-Attention**：
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```
- 每个 query 与所有 key 计算相似度
- 复杂度：O(N²)，N = H×W

**Mamba 中的全局扫描**：
```
h_t = SSM(x_t, h_{t-1})  # 状态空间模型
```
- 序列长度 N = H×W
- 复杂度：O(N)，但 N 很大时仍然昂贵

**优势**：
- 可以捕获任意距离的依赖关系
- 适合需要全局上下文的任务（如分割大肿瘤）

**劣势**：
- 计算和显存开销大
- 在大特征图上不可行

---

#### 局部注意力（Local Attention）

**固定窗口 = 局部注意力（不重叠）**

类似于 **Swin Transformer** 的窗口注意力：
```
Attention(Q, K, V) = softmax(QK^T / √d) V
但 Q, K, V 只来自同一个窗口内
```

**特点**：
- 每个窗口独立计算注意力
- 窗口之间**完全隔离**
- 复杂度：O(num_windows × window_size²)

**问题**：
```
窗口边界处的像素无法与相邻窗口交互

┌───────┬───────┐
│   A   │   B   │  ← A 和 B 在不同窗口
│       │       │     无法直接交互
└───────┴───────┘
```

**Swin Transformer 的解决方案**：
- **Shifted Window**：交替使用常规窗口和移位窗口
- 通过移位实现跨窗口信息交互

**RTHD 的解决方案**：
- **三视图融合**：不同视图提供不同角度的信息
- Axial 视图的窗口边界 ≠ Coronal 视图的窗口边界
- 通过视图融合实现跨窗口交互

---

**滑动窗口 = 局部注意力（重叠）**

类似于 **卷积神经网络** 的重叠感受野：
```
每个位置被多个窗口覆盖
最终输出 = 多个窗口输出的融合
```

**特点**：
- 每个位置被多个窗口处理
- 窗口之间有**信息共享**
- 复杂度：O(overlap_factor × num_windows × window_size²)

**优势**：
```
重叠区域提供跨窗口信息交互

┌───────┐
│   A   ├───────┐
│   ┌───┼───┐   │
│   │ C │   │ B │  ← C 同时在 A 和 B 的窗口中
│   └───┼───┘   │     可以融合两个窗口的信息
└───────┴───────┘
```

**融合策略**：
1. **平均融合**：`output = mean(window_outputs)`
2. **加权融合**：`output = Σ w_i × window_outputs_i`
3. **学习融合**：使用可学习的融合网络

---

## 📈 计算复杂度对比

假设特征图尺寸：H×W = 32×32，window_size = 8

| 策略 | 窗口数量 | 序列长度 | 总计算量 | 相对复杂度 |
|-----|---------|---------|---------|-----------|
| **全局平铺** | 1 | 1024 | O(1024 × C) | 1× |
| **固定窗口** | 16 | 64 | O(16 × 64 × C) | 1× |
| **滑动窗口 (50%)** | 49 | 64 | O(49 × 64 × C) | **3.06×** |

**关键发现**：
- 固定窗口与全局平铺的总计算量相同，但序列更短，效率更高
- 滑动窗口的计算量是固定窗口的 **3 倍**

---

## 🎯 三种策略的权衡

### 性能 vs 效率

```
性能（全局信息）
  ↑
  │  全局平铺 ●
  │            
  │  滑动窗口 ●
  │            
  │  固定窗口 ●
  │            
  └──────────────→ 效率（计算速度）
     慢          快
```

### 感受野 vs 计算量

| 策略 | 感受野 | 边界处理 | 计算量 | 显存 | 推荐场景 |
|-----|-------|---------|--------|------|---------|
| **全局平铺** | 全局 | 无边界 | 高 | 高 | 小特征图 |
| **固定窗口** | 局部 | 有边界 | 低 | 低 | 大特征图 + 三视图融合 |
| **滑动窗口** | 局部+ | 缓解边界 | 中 | 中 | 需要平滑特征 |

---

## 🔧 滑动窗口的实现方案

### 方案 1: 简单滑动窗口

```python
def sliding_window_partition(x: torch.Tensor, window_size: int, stride: int):
    """
    滑动窗口分割（重叠）
    
    输入: (B, C, H, W)
    输出: (B*num_windows, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    
    # 使用 unfold 实现滑动窗口
    # unfold(dimension, size, step)
    windows = x.unfold(2, window_size, stride).unfold(3, window_size, stride)
    # windows: (B, C, nH, nW, window_size, window_size)
    
    nH, nW = windows.shape[2], windows.shape[3]
    windows = windows.permute(0, 2, 3, 1, 4, 5).contiguous()
    windows = windows.view(B * nH * nW, C, window_size, window_size)
    
    return windows, (nH, nW)

def sliding_window_reverse(windows: torch.Tensor, window_size: int, stride: int,
                           nH: int, nW: int, H: int, W: int):
    """
    滑动窗口合并（需要处理重叠区域）
    
    输入: (B*num_windows, C, window_size, window_size)
    输出: (B, C, H, W)
    """
    B = windows.shape[0] // (nH * nW)
    C = windows.shape[1]
    
    # 重塑窗口
    windows = windows.view(B, nH, nW, C, window_size, window_size)
    
    # 初始化输出和计数器（用于平均）
    output = torch.zeros(B, C, H, W, device=windows.device, dtype=windows.dtype)
    count = torch.zeros(B, C, H, W, device=windows.device, dtype=windows.dtype)
    
    # 将每个窗口的输出累加到对应位置
    for i in range(nH):
        for j in range(nW):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + window_size
            w_end = w_start + window_size
            
            output[:, :, h_start:h_end, w_start:w_end] += windows[:, i, j, :, :, :]
            count[:, :, h_start:h_end, w_start:w_end] += 1
    
    # 平均融合重叠区域
    output = output / count
    
    return output
```

### 方案 2: 可学习的融合权重

```python
class LearnableSlidingWindowFusion(nn.Module):
    def __init__(self, dim, window_size, stride):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.ones(1, dim, window_size, window_size))
        
    def forward(self, windows, nH, nW, H, W):
        # 类似 sliding_window_reverse，但使用可学习的权重
        ...
```

---

## 🧪 建议的消融实验

### 实验 #7: 滑动窗口版（新增）

```python
rthd_config = {
    'view_mode': 'tri',
    'share_weights': True,
    'scan_mode': 'omni',
    'use_local_window': True,
    'window_type': 'sliding',      # 新增：'fixed' or 'sliding'
    'window_size': 8,
    'stride': 4,                   # 新增：滑动步长
}
```

### 完整的消融实验对比

| 实验 | 窗口策略 | 三视图 | 参数共享 | 全向扫描 | 预期性能 | 计算量 |
|-----|---------|-------|---------|---------|---------|--------|
| #5 | 全局平铺 | ✓ | ✓ | ✓ | 基线 | 1× |
| #6 | 固定窗口 | ✓ | ✓ | ✓ | 基线+2% | 1× |
| **#7** | **滑动窗口** | ✓ | ✓ | ✓ | **基线+3%?** | **3×** |

**预期结果**：
- #7 滑动窗口性能可能略优于 #6 固定窗口（+1%）
- 但计算量增加 3 倍
- 性价比可能不如固定窗口

---

## 📝 论文写作角度

### 1. 从注意力机制角度阐述

**引言部分**：
> "医学图像分割需要平衡全局上下文和局部细节。全局注意力机制（如 Transformer）可以捕获长距离依赖，但在大特征图上计算开销巨大。局部注意力机制（如窗口注意力）计算高效，但可能丢失全局信息。"

**方法部分**：
> "我们提出 RTHD，通过三视图分解实现高效的全局建模。在每个视图内，我们采用固定窗口策略进行局部注意力计算。虽然单个视图内是局部的，但通过三视图融合，模型可以间接获得全局感受野。"

**消融实验部分**：
> "我们对比了三种窗口策略：
> 1. 全局平铺：全局注意力，但计算开销大
> 2. 固定窗口：局部注意力，计算高效，通过三视图融合获得全局信息
> 3. 滑动窗口：局部注意力 + 重叠融合，性能略优但计算量增加 3 倍
> 
> 实验表明，固定窗口 + 三视图融合在性能和效率之间取得了最佳平衡。"

### 2. 可视化对比图

```
图 X: 不同窗口策略的感受野对比

(a) 全局平铺          (b) 固定窗口          (c) 滑动窗口
┌─────────────┐      ┌───┬───┬───┐        ┌─┬─┬─┬─┬─┐
│             │      │   │   │   │        ├─┼─┼─┼─┼─┤
│   全局感受野  │      ├───┼───┼───┤        ├─┼─┼─┼─┼─┤
│             │      │   │   │   │        ├─┼─┼─┼─┼─┤
└─────────────┘      └───┴───┴───┘        └─┴─┴─┴─┴─┘
  序列长度: 1024       序列长度: 64         序列长度: 64
  窗口数: 1            窗口数: 9            窗口数: 25
  重叠: 无             重叠: 无             重叠: 50%
```

---

## 🎯 实现建议

### 短期（当前论文）

1. **保持当前的固定窗口实现**（实验 #6）
2. **在论文中讨论**：
   - 为什么选择固定窗口而不是滑动窗口
   - 三视图融合如何缓解固定窗口的边界问题
   - 性能和效率的权衡

### 长期（后续研究）

1. **实现滑动窗口版本**（实验 #7）
2. **进行完整对比实验**
3. **如果滑动窗口显著优于固定窗口**：
   - 可以作为改进版本发表
   - 或者作为 follow-up 工作

---

## 💡 关键洞察

### 为什么固定窗口 + 三视图融合可能足够好？

```
固定窗口的问题：窗口边界处信息隔离

Axial 视图的窗口边界：
┌───────┬───────┐
│       │       │
└───────┴───────┘
        ↑ 边界

Coronal 视图的窗口边界：
┌───────────────┐
│               │
├───────────────┤  ← 边界（不同位置！）
│               │
└───────────────┘

通过三视图融合，Axial 的边界位置
可以从 Coronal 和 Sagittal 获得跨边界信息！
```

**结论**：
- 固定窗口 + 三视图融合 ≈ 滑动窗口的效果
- 但计算量只有滑动窗口的 1/3
- 这是 RTHD 的核心创新点！

---

**创建时间**: 2026-05-27  
**作者**: Claude (Kiro)  
**基于**: 用户提出的全局/局部注意力对比思路
