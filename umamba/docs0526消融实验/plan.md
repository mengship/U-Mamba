Role: Senior Medical Image Deep Learning and PyTorch Expert

Task:
我正在写一篇关于 3D 脑肿瘤分割（BraTS）的毕业论文。目前我设计了一个名为 RTHD 的基础架构（基于 3D 降维 2D Mamba 的思想）。为了完成论文中的“大方向一：空间降维与结构保持”的消融实验，我需要你在我现有的模型/模块代码中，引入一套【控制变量的消融实验开关】。

请阅读我提供的代码，重构相关的 Mamba 模块（或 U-Net 编码器层），使其能够通过初始化参数（或 config 字典）动态切换 5 种不同的消融变体。

---
我目前已经运行了一个了，在 nnUnetTrainer/nnUNetTrainerUMambaEncRTHD.py 帮我看一下，他是哪个消融实验
### 📊 消融实验设计矩阵（你需要实现的逻辑）

你需要通过在 `__init__` 中引入以下四个核心控制参数：
1. view_mode (str): 可选 ['tri', 'single']。'tri' 表示 1.1 三视图分解；'single' 表示仅保留 Axial 轴状位。
2. share_weights (bool): 默认为 True。True 表示 1.2-A 三视图复用同一个 Mamba 参数；False 表示定义三个独立的 Mamba 块。
3. scan_mode (str): 可选 ['omni', 'standard']。'omni' 表示 1.2-B 二维全向扫描（4向/8向）；'standard' 表示标准的 1D 或双向扫描。
4. use_local_window (bool): 默认为 True。True 表示 1.3 局部滑窗扫描（LoMamba 思想）；False 表示不分窗，直接全局平铺（Global Flatten）扫描。

对应的 5 个消融实验配置如下：
- [#2 单视图降级版]: view_mode='single', share_weights=False, scan_mode='standard', use_local_window=False
- [#3 独立参数版]: view_mode='tri', share_weights=False, scan_mode='omni', use_local_window=True
- [#4 常规扫描版]: view_mode='tri', share_weights=True, scan_mode='standard', use_local_window=True
- [#5 全局平铺版]: view_mode='tri', share_weights=True, scan_mode='omni', use_local_window=False
- [#6 完整创新版]: view_mode='tri', share_weights=True, scan_mode='omni', use_local_window=True

---

### 🛠 具体的重构逻辑要求（请严格遵守维度匹配）

1. 关于 view_mode == 'single' (#2):
   - 输入张量为 (B, C, D, H, W)。
   - 如果为 'single'，直接对 D 维度进行处理（例如投影或切片），只保留 (B, C, H, W) 视角送入 Mamba 模块。Coronal 和 Sagittal 分支不执行，对应的前向传播位置直接补零或不激活。为了保持主干网络输出形状依然是 3D 的 (B, C, D, H, W) 以便后续解码，单视图处理完后需要通过 `repeat` 或 `expand` 将 D 维度还原。

2. 关于 share_weights == False (#3):
   - 在 `__init__` 中，如果 `share_weights=True`，只需实例化一个 Mamba 扫描模块（如 `self.mamba_block`）；
   - 如果 `share_weights=False`，请实例化三个独立的模块：`self.mamba_axial`、`self.mamba_coronal`、`self.mamba_sagittal`。并在 `forward` 中分别对应处理三个视图的张量。

3. 关于 scan_mode == 'standard' (#4):
   - 在 Mamba 扫描函数中，如果是 'omni'，保留原有的全向/多向扫描拼装逻辑；
   - 如果是 'standard'，则退化为最基础的 1D 线性平铺正向扫描，或者标准的 2D 纵横双向扫描（Bidirectional Scan）。

4. 关于 use_local_window == False (#5):
   - 在进入 Mamba 之前，如果 `use_local_window=True`，执行原有的 Window Partition 逻辑（LoMamba 局部滑窗）；
   - 如果 `use_local_window=False`，直接跳过分窗步骤，将整个 2D 尺寸 (H, W) 展平为 (H*W) 的长序列送入 Mamba 扫描，跑完后再 reshape 回原样。

---

### 📥 输出要求
1. 请直接修改我下面给出的代码文件，保持原有类的名称和核心前向传播接口不变。
2. 注释要清晰，在每一个消融开关对应的 `if-else` 分支处，写明对应的是实验 `#2`, `#3`, `#4` 还是 `#5`。
3. 确保重构后的代码具有健壮的维度检查，绝对不能出现 Tensor 形状不匹配导致的 RuntimeError。

以下是我目前的核心代码，请对其进行重构：
```python
# ——请在这里粘贴你现有的 rthd_modules.py 或编码器层的完整代码——