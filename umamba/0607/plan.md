# RTHD 方案按 2024-2026 年脑肿瘤分割趋势的调整建议

## 一、当前问题判断

原始想法：

> 我正在写一篇关于 3D 脑肿瘤分割（BraTS）的毕业论文。以 U-Mamba 为基础做修改，目前设计了一个名为 RTHD 的基础架构，基于 3D 降维 2D Mamba 的思想，在编码区做三视图模块优化。但目前来看有些薄弱，不足以支撑一个章节；画总体架构图时也有点单薄，只改动编码区，感觉图都画不出来，希望给一些新的思路。

我的判断：这个问题不是 RTHD 本身方向错，而是现在的叙事粒度太小。

如果只写成“把 3D 特征投影到三个 2D 视图，然后用 2D Mamba 扫描”，它确实像一个编码器局部替换模块。  
但如果按照近两年脑肿瘤分割的发展趋势重新包装，它可以升级成：

`面向多模态 3D MRI 脑肿瘤分割的轻量化多视图状态空间建模网络`

这一表述会比“编码器三视图模块优化”更完整，也更贴近 2024-2026 年主流关注点。


## 二、近两年趋势对 RTHD 的启发

### 2.1 趋势一：Mamba/SSM 仍然有价值，但不能只讲“替代 Transformer”

近两年医学图像分割里的 Mamba 方向，重点已经从“我用了 Mamba”转向：

- 如何降低 3D MRI 长序列建模的显存和计算开销
- 如何保留 3D 结构连续性
- 如何融合局部纹理、全局上下文和边界信息
- 如何在多模态或缺失模态输入下保持鲁棒

对 RTHD 来说，核心卖点应该写成：

`用三视图二维状态空间扫描近似 3D 长程依赖建模，在降低序列长度和计算负担的同时，保留轴向、冠状位、矢状位的互补空间信息。`

不要只说：

`把 3D 变成 2D，所以更省。`


### 2.2 趋势二：脑肿瘤分割越来越强调临床真实场景

BraTS 近两年的变化说明，研究不再只满足于标准术前胶质瘤分割，而是更关注：

- post-treatment MRI
- lesion-wise Dice / HD95
- 小病灶、边界模糊、坏死区域、增强区域
- 跨中心、跨扫描协议泛化
- 临床中不完整模态输入

所以 RTHD 的论文叙事最好不要只围绕平均 Dice 提升，而应补充：

- ET、TC、WT 三个区域的差异性分析
- HD95 与边界质量分析
- 小肿瘤或小增强区域的单独讨论
- 参数量、FLOPs、显存、推理速度对比


### 2.3 趋势三：缺失模态是非常适合扩展的一条线

BraTS 是多模态 MRI，标准输入通常包括：

- T1
- T1ce
- T2
- FLAIR

但临床场景中四模态不一定完整。2025 年已有工作专门研究 incomplete modalities 下的 Mamba 融合。

如果你觉得当前 RTHD 只改编码区太单薄，最自然、最贴近趋势、也最容易解释的扩展是：

`把三视图建模与模态鲁棒融合结合起来。`

也就是从：

`Tri-view spatial modeling`

升级为：

`Modality-aware tri-view spatial modeling`


### 2.4 趋势四：基础模型很热，但不建议作为第一章主线

VISTA3D、MedSAM2、SAM/MedSAM 这类基础模型是 2025-2026 年大趋势。  
但对当前毕业论文第一章来说，不建议贸然把 foundation model 加进主方法里。

原因：

- 实现成本高
- 训练和对比复杂
- 容易把论文主线从 U-Mamba/RTHD 拉散
- BraTS 上专用 3D 网络仍然是强基线

更合适的处理方式：

- 在绪论和相关工作里讨论 foundation model 趋势
- 在展望里写 RTHD 可作为 3D 医学基础模型的轻量空间建模组件
- 主方法仍然聚焦 U-Mamba + RTHD


## 三、建议把 RTHD 扩成一个完整章节的方法

章节标题建议：

`基于多视图状态空间协同建模的轻量化 3D 脑肿瘤分割方法`

英文可写为：

`Lightweight Multi-view State Space Collaborative Modeling for 3D Brain Tumor Segmentation`

方法总体结构建议由四个部分组成：

1. `Tri-view Projection`
2. `View-specific 2D Mamba Scanning`
3. `Cross-view Collaborative Fusion`
4. `Structure-aware 3D Reconstruction`

这样架构图就不再只是“编码器里塞一个模块”，而是能画成一个完整的 RTHD block：

`3D feature -> 三视图投影 -> 三路 Mamba 扫描 -> 跨视图交互 -> 结构感知重建 -> residual fusion`


## 四、最推荐的新增思路

### 4.1 新思路 A：跨视图协同交互

当前三视图模块最大的问题是：

`三个视图各扫各的，只在最后融合。`

这会让审稿人或答辩老师质疑：

- 三个视图之间有没有信息交流？
- 是不是只是三个 2D 分支的简单拼接？
- 三视图真的比单视图强在哪里？

建议加入：

`Cross-view Collaborative Fusion`

实现方式可以很轻量：

- 先把 axial、coronal、sagittal 三个视图重建为临时 3D 表示
- 用 `1x1x1 Conv3d + sigmoid` 生成三路 view gate
- 三路 gate 分别调制三个视图特征
- 再进行最终 3D 重建

论文中可以写成：

`跨视图协同模块用于自适应建模不同解剖平面之间的互补关系，避免独立二维扫描造成的空间一致性不足。`

这是最值得做的第一优先级。


### 4.2 新思路 B：结构感知重建

当前从 2D 视图恢复到 3D，如果只是 broadcast 或简单相加，会显得粗糙。  
脑肿瘤分割特别依赖边界和区域连续性，所以可以把重建模块升级为：

`Structure-aware Reconstruction`

可选实现：

- 在三视图融合后加入轻量 `Conv3d(3x3x3)`
- 加一个边界增强分支，预测 boundary attention
- 用 boundary attention 调制解码器特征
- 损失函数里加入 boundary loss 或 surface loss，作为可选实验

论文中可以把它和 BraTS 的 HD95 指标联系起来：

`结构感知重建模块用于缓解二维投影带来的空间细节损失，并提升肿瘤边界与小区域分割稳定性。`


### 4.3 新思路 C：编码器-解码器非对称部署

你现在觉得“只改编码区，图画不出来”，这里有一个很自然的解决办法：

`不要只把 RTHD 放在编码器，也不要全网络硬塞，而是做非对称部署。`

建议：

- 编码器浅层：保留卷积，提取局部纹理
- 编码器中高层：使用 RTHD，建模长程空间依赖
- 解码器高层或跳跃连接处：加入轻量 RTHD refinement
- 输出前：加入结构感知细化模块

这样图上可以画出：

- encoder RTHD
- skip fusion RTHD
- decoder refinement
- final structure-aware head

论文叙事也更完整：

`编码器负责全局上下文建模，解码器负责结构恢复和边界细化。`


### 4.4 新思路 D：模态感知三视图门控

这是最贴近最新趋势的一条扩展，但实现复杂度比 A、B、C 高一点。

思路：

`不同 MRI 模态对不同肿瘤区域的贡献不同，不同解剖视图对空间结构的贡献也不同。`

例如：

- FLAIR 对 WT 更重要
- T1ce 对 ET 更重要
- T2 对水肿区域有帮助
- 三视图对不同形态肿瘤的贡献不同

可以设计：

`Modality-view Gate`

输入四模态特征，输出每个模态、每个视图的权重：

`gate shape: B x modality x view x C`

低成本版本：

- 对每个模态做 global average pooling
- 经过 MLP 得到模态权重
- 对三视图输出做 view-level gate
- 最后融合

这个模块可以作为第二阶段增强，不建议第一版就做得很复杂。


## 五、最推荐的实施路线

### 5.1 第一优先级：把第一章做扎实

建议不要一下子追 foundation model 或复杂缺失模态。  
第一章先把 RTHD 从“小模块”扩成完整方法：

1. `跨视图协同交互`
2. `结构感知重建`
3. `编码器-解码器非对称部署`

这三件事足够支撑：

- 方法章节
- 总体架构图
- 模块细节图
- 消融实验
- 复杂度分析


### 5.2 第二优先级：加一个缺失模态鲁棒实验

如果时间允许，可以不新增太多代码，只做实验层面的 missing modality：

- 四模态完整输入
- 随机丢 1 个模态
- 随机丢 2 个模态
- 单模态输入

然后比较：

- U-Mamba
- 当前 RTHD
- RTHD + cross-view gate

如果 RTHD 在缺失模态下更稳定，就能把论文和最新趋势明显挂上钩。


### 5.3 第三优先级：写入 foundation model 展望

基础模型方向建议放在绪论、相关工作和展望，不作为第一章主实验：

- 说明 3D 医学分割正在走向 foundation model
- 说明当前 3D foundation model 仍面临高计算成本和临床适配问题
- 说明 RTHD 这类轻量多视图 SSM 模块未来可作为 foundation model 的高效 3D 表示组件


## 六、建议实验设计

### 6.1 主对比实验

建议至少包含：

- nnU-Net
- U-Mamba
- SwinUNETR 或 SegResNet
- 当前 RTHD
- RTHD + cross-view fusion
- RTHD + cross-view fusion + structure-aware reconstruction

指标：

- Dice: WT / TC / ET
- HD95: WT / TC / ET
- Params
- FLOPs
- GPU memory
- inference time


### 6.2 消融实验

建议消融表这样设计：

| 编号 | Tri-view | 2D Mamba | Cross-view | Structure-aware | Decoder RTHD | 目的 |
|---|---|---|---|---|---|---|
| A0 | 否 | 否 | 否 | 否 | 否 | U-Mamba baseline |
| A1 | 是 | 是 | 否 | 否 | 否 | 验证基础 RTHD |
| A2 | 是 | 是 | 是 | 否 | 否 | 验证跨视图交互 |
| A3 | 是 | 是 | 是 | 是 | 否 | 验证结构感知重建 |
| A4 | 是 | 是 | 是 | 是 | partial | 验证解码器细化 |


### 6.3 补充分析

建议加三类分析，让论文更像完整研究而不是工程改块：

- 可视化：展示 ET、小肿瘤、边界模糊病例
- 复杂度：比较序列长度、参数量、显存
- 鲁棒性：随机缺失模态或噪声扰动


## 七、架构图怎么画

详细绘制方案见：[architecture_diagram.md](architecture_diagram.md)

可以画三张图：

### 图 1：整体网络

`Input MRI -> Stem -> Encoder with RTHD -> Bottleneck -> Decoder Refinement -> Segmentation Head`

重点标出：

- RTHD 在编码器中高层
- decoder partial RTHD refinement
- skip connection


### 图 2：RTHD block

`3D Feature -> Tri-view Projection -> Shared/Independent 2D Mamba -> Cross-view Gate -> 3D Reconstruction -> Residual`

这是第一章最核心的图。


### 图 3：跨视图协同模块

`Axial / Coronal / Sagittal -> temporary 3D fusion -> gate generation -> view recalibration`

这张图能解释为什么你的方法不是简单三分支。


## 八、论文创新点建议写法

可以写成三点：

1. 提出一种面向 3D 脑肿瘤 MRI 的轻量化三视图状态空间建模模块，将 3D 长序列建模分解为多个解剖平面的二维状态空间扫描，在降低计算开销的同时保留多方向空间上下文。

2. 设计跨视图协同融合机制，通过 3D 临时重建与视图门控建模轴向、冠状位和矢状位之间的互补关系，缓解独立二维扫描造成的空间一致性不足。

3. 引入结构感知重建与编码器-解码器非对称部署策略，在全局上下文建模和局部边界恢复之间取得平衡，提升脑肿瘤子区域尤其是边界模糊区域的分割稳定性。


## 九、当前最应该避免的方向

不建议现在做：

- 完整接入 SAM/MedSAM/VISTA3D 作为主方法
- 同时做 foundation model、缺失模态、跨域泛化、测试时自适应
- 为了让图复杂而堆很多注意力模块
- 只报告 Dice，不报告 HD95 和复杂度
- 只和很弱的 baseline 比，不和 nnU-Net/U-Mamba/SwinUNETR/SegResNet 比


## 十、最终建议

最稳妥、最像毕业论文、也最贴近近两年趋势的路线是：

`RTHD 基础三视图建模 -> 跨视图协同交互 -> 结构感知重建 -> 编码器-解码器非对称部署 -> 缺失模态鲁棒性补充实验`

这条路线的优点：

- 不偏离你已经实现的 U-Mamba/RTHD 主线
- 能自然画出完整架构图
- 能支撑一个方法章节
- 能设计清晰的消融实验
- 能和 2024-2026 年 Mamba、缺失模态、临床鲁棒性趋势接上

一句话总结：

`不要把 RTHD 写成一个编码器替换块，要把它写成一个面向 3D 多模态脑肿瘤分割的轻量多视图空间建模框架。`


## 十一、可参考的近期方向

- BraTS-UMamba, MICCAI 2025：Mamba 用于脑肿瘤分割，强调长程依赖、低复杂度，并加入频域增强。
- IM-Fuse, MICCAI 2025：Mamba 用于不完整模态脑肿瘤分割，说明 missing modality 已经是非常重要的临床鲁棒性方向。
- BraTS 2024 Post-treatment Glioma Challenge：强调 post-treatment MRI、lesion-wise Dice、HD95 和临床真实场景。
- VISTA3D, CVPR 2025：说明 3D 医学分割正在向 foundation model 发展，但更适合作为论文背景和未来展望。
