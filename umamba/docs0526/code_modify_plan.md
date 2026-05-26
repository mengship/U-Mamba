# 代码修改说明
我是一名硕士研究生，正在做脑肿瘤分割方向，我的第一个修改点是：引入 3D深度可分离卷积（DS-Conv），并结合 三视图递归（RTHD）机制。将 3D 脑肿瘤张量解耦投影为 Axial（轴状位）、Coronal（冠状位）、Sagittal（矢状位） 三个 2D 正交切片特征流，利用参数共享的 VMamba (SS2D) 思想进行轻量化二维扫描，大幅斩断计算流。我们在代码中通过 permute 和 view 操作，强行将 3D 特征图在物理层面上切片解耦为轴状位（Axial）、冠状位（Coronal）和矢状位（Sagittal）。随后，让它们通过同一个 2D VMamba 模块进行参数共享的二维选择性扫描（SS2D），最后再逆向重组回 3D。
## 1. 实施目标
在 U-Mamba / nnUNetv2 架构中，将原本消耗大量显存的 3D 全量密集扫描或重型 3D 卷积替换为轻量化的 **三视图递归（RTHD）视觉状态空间块（`TriViewVMambaBlock`）**。
该模块将 3D 体积张量解耦投影为三个正交的 2D 切片流（Axial 轴状位、Coronal 冠状位、Sagittal 矢状位），并利用 **单个参数共享的 2D VMamba / VSSBlock** 进行轻量化二维扫描。此举将 Mamba 扫描的序列长度从 $O(D \times H \times W)$ 彻底斩断至 $O(H \times W)$，从而在消费级 GPU（如 RTX 3090）上释放海量显存。

## 2. 目标文件与路径
- **创建新模块文件**：`nnunetv2/nets/rthd_modules.py` （用于存放所有自定义的 RTHD 轻量化组件）。
- **无损集成目标文件，如果必要，可以新建代码文件**：`nnunetv2/nets/UMambaBot.py` 或 `nnunetv2/nets/UMambaEnc.py` （取决于你目前使用的 U-Mamba 变体）。

## 3. 核心依赖与前置检查（AI 助手注意）
在开始编写代码前，请先检索代码库，确认原版 2D VMamba 核心块（例如 `VSSBlock` 或 `SS2D`）的输入通道格式（Channels Format）：
- **通道格式问题**：原生的 2D VMamba 通常为了对齐语言模型，采用 **Channels-Last** `(B, H, W, C)` 格式进行二维扫描；而 nnUNetv2 / U-Mamba 则是标准的医疗影像 **Channels-First** `(B, C, D, H, W)`。
- 本规范在实现中已经内置了自动转换逻辑，并通过 `channels_last=True/False` 参数进行动态兼容，防止形状不匹配（Shape Mismatch）或连续性内存（Contiguous）报错。

## 4.在修改时，做到无损原始代码，我们可以新建文件的方式，来实现新的思路