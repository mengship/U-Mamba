# 基于高效三视图状态空间建模的轻量化脑肿瘤MRI分割网络

## 摘要

针对三维脑肿瘤MRI分割中跨切面上下文建模不足、体数据空间建模开销较大以及解码阶段结构细节恢复不充分等问题，提出一种融合高效三视图状态空间建模与阶段感知解码的轻量化分割网络。该方法以U-Mamba为基础框架，将三维特征投影到轴状位、冠状位和矢状位三个二维视图，并通过参数共享的SS2D模块和轻量跨视图门控进行多方向上下文建模，以降低三维状态空间建模开销。针对解码阶段不同分辨率特征的恢复需求，设计阶段感知解码策略，在低分辨率层补偿全局结构信息，在高分辨率层保留卷积细化能力；同时构建语义引导跳跃连接特征标定模块，利用解码端高级语义对编码端浅层细节特征进行自适应筛选。在BraTS2020脑肿瘤分割数据集上的五折实验结果表明，本文方法平均Dice由U-Mamba的85.27%提升至85.53%，平均HD95由4.606降低至4.077；参数量由42.75M降低至37.10M，峰值显存占用由4.22GB降低至2.46GB。结果说明，该方法在保持区域分割精度的同时，能够改善肿瘤边界定位质量，并降低模型复杂度。

## 关键词

脑肿瘤分割，磁共振成像，状态空间模型，高效三视图状态空间建模，阶段感知解码，跳跃连接标定

## Abstract

To address insufficient cross-slice contextual modeling, high computational cost of volumetric spatial representation, and limited structural detail recovery in three-dimensional brain tumor MRI segmentation, this paper proposes a lightweight segmentation network integrating Efficient Tri-view State-space Modeling and stage-aware decoding. Built upon U-Mamba, the proposed method projects three-dimensional features into axial, coronal, and sagittal two-dimensional views, and uses a shared SS2D module with lightweight cross-view gating for multi-directional contextual modeling. This design reduces the cost of direct volumetric state-space modeling. In the decoder, a stage-aware recovery strategy is designed according to feature resolutions, where low-resolution stages compensate global structural information and high-resolution stages retain convolutional refinement. In addition, a semantic-guided skip feature calibration module is developed to adaptively select shallow encoder features using high-level decoder semantics. Five-fold experiments on the BraTS2020 brain tumor segmentation dataset showed that the proposed method improved the average Dice score from 85.27% to 85.53%, reduced the average HD95 from 4.606 to 4.077, decreased the number of parameters from 42.75M to 37.10M, and reduced peak memory consumption from 4.22GB to 2.46GB compared with U-Mamba. The results indicate that the method improves tumor boundary localization and reduces model complexity while maintaining regional segmentation accuracy.

## Key words

brain tumor segmentation, magnetic resonance imaging, state space model, efficient tri-view state-space modeling, stage-aware decoding, skip feature calibration

## 0 引言

脑肿瘤是中枢神经系统中常见且危害较高的疾病之一，病灶区域通常呈现形态不规则、边界模糊和组织异质性强等特点。磁共振成像（magnetic resonance imaging，MRI）能够从多序列、多对比度角度反映脑组织结构及肿瘤区域特征，是脑肿瘤诊断、治疗方案制定和疗效评估中的重要影像手段。准确分割增强肿瘤、肿瘤核心和水肿区域，对于辅助医生判断肿瘤范围、制定放疗计划以及评估疾病进展具有重要意义。然而，人工勾画脑肿瘤区域不仅耗时耗力，而且容易受到医生经验、影像质量和病灶复杂程度的影响。因此，研究自动化、精确化的脑肿瘤MRI分割方法具有重要的临床价值和应用前景。

近年来，深度学习方法已成为医学图像分割领域的主流技术。以U-Net[1]和3D U-Net[2]为代表的编码解码网络通过跳跃连接融合浅层空间细节和深层语义信息，在多种医学图像分割任务中取得了良好效果。nnU-Net[3]进一步通过自适应数据预处理、网络配置和训练策略，在脑肿瘤分割等任务中表现出较强的鲁棒性。然而，卷积神经网络主要依赖局部感受野进行特征提取，虽然可通过堆叠卷积层或下采样操作扩大感受野，但对三维脑MRI中复杂的长程空间依赖和跨切面上下文关系建模仍然不足。

为增强全局建模能力，Transformer被引入医学图像分割任务[4-7]。自注意力机制能够捕获远距离像素或体素之间的依赖关系，有助于提升复杂结构区域的分割性能。但是，三维医学图像通常具有较大的体数据尺寸，直接在3D特征上进行全局自注意力计算会带来较高的显存占用和计算开销，限制了其在高分辨率脑肿瘤分割任务中的应用。此外，Transformer模型通常需要较大规模的数据支撑，其训练稳定性和部署效率仍需进一步优化。

状态空间模型（state space model，SSM）近年来在视觉任务中受到广泛关注。与自注意力机制相比，基于Mamba[8]的选择性状态空间模型能够以近似线性的复杂度建模长程依赖关系，为高效处理高分辨率图像和三维医学影像提供了新的思路。已有研究将Mamba引入医学图像分割网络中，如U-Mamba[9]和SegMamba[10]等，通过结合卷积结构与状态空间建模能力，在全局上下文表达和计算效率之间取得了较好平衡。然而，现有方法在三维脑肿瘤分割任务中仍存在以下不足：一是直接对三维体特征进行空间建模仍可能带来较高计算负担；二是三维医学图像包含轴状位、冠状位和矢状位等多方向结构信息，单一方向或简单展平方式难以充分利用跨视图互补特征；三是编码端全局建模增强后，解码阶段如何恢复肿瘤边界和局部细节仍有待进一步研究。

针对上述问题，本文提出一种融合高效三视图状态空间建模与阶段感知解码的脑肿瘤MRI分割方法。该方法以U-Mamba编码解码结构为基础，在编码端设计高效三视图状态空间建模模块（Efficient Tri-view State-space Modeling，ETSM），将三维特征分别投影为轴状位、冠状位和矢状位三个正交二维视图，并通过参数共享的二维视觉状态空间模块进行高效扫描，以增强多方向空间上下文表达并降低三维空间建模复杂度。进一步地，考虑到解码阶段不同分辨率特征的恢复需求不同，本文设计阶段感知解码恢复策略，在低分辨率解码层利用ETSM模块补偿全局结构信息，在高分辨率解码层保留卷积细化能力，以恢复局部纹理与边界细节。同时，本文引入语义引导的跳跃连接特征标定模块，利用解码端高级语义信息对编码端跳跃特征进行自适应筛选，缓解浅层空间细节与深层语义信息之间的不一致问题。

## 1 基本方法

### U-Mamba网络结构

U-Mamba[9]是一种融合卷积神经网络与状态空间模型的医学图像分割网络，其整体结构延续U-Net类编码解码框架。编码端通过多阶段下采样逐步提取由浅层空间细节到深层语义信息的多尺度特征，解码端通过逐层上采样恢复空间分辨率，并结合跳跃连接融合编码端同尺度特征，以获得更加精细的分割结果。对于输入的三维多模态MRI图像，可表示为：X\in\mathbb{R}^{B\times C\times D\times H\times W}，其中，B表示批大小，C表示输入模态或通道数，D、H和W分别表示三维体数据的深度、高度和宽度。

与传统U-Net主要依赖卷积操作不同，U-Mamba在网络中引入Mamba模块以增强长程依赖建模能力。卷积模块擅长提取局部纹理、边缘和形态特征，状态空间模块则能够在较低计算复杂度下捕获更大范围的空间上下文信息。因此，U-Mamba兼具局部特征提取能力和全局依赖建模能力，可作为三维脑肿瘤MRI分割的有效基础网络。

### 视觉状态空间模型

状态空间模型最初常用于序列建模任务，其基本思想是通过隐状态递推描述输入序列与输出序列之间的动态关系。连续形式的状态空间模型可表示为：
h\prime\left(t\right)=Ah\left(t\right)+Bx\left(t\right)
y\left(t\right)=Ch\left(t\right)+Dx\left(t\right)
其中，x(t)为输入信号，h(t)为隐状态，y(t)为输出信号，A、B、C和D 为可学习参数。经过离散化后，状态空间模型可写为：
h_t=\bar{A}h_{t-1}+\bar{B}x_t
y_t=Ch_t+Dx_t
该递推形式使模型能够以线性复杂度处理长序列数据。与Transformer中自注意力机制通常具有平方级复杂度不同，状态空间模型在长序列建模时具有更高的计算效率，因而适合处理高分辨率图像和三维医学影像等大规模数据。在视觉任务中，视觉状态空间模型通常将图像特征按照一定空间顺序展开为序列，并利用选择性扫描机制对空间依赖关系进行建模。对于二维图像特征，模型可沿水平、垂直或多方向路径进行扫描，从而捕获空间上下文信息。对于三维医学图像，若直接将整个体数据展平为长序列进行扫描，序列长度会随D×H×W快速增长，计算和显存开销也随之增加。因此，如何在保持三维空间信息的同时降低状态空间建模复杂度，是三维医学图像分割中的关键问题之一。

### 多视图三维医学图像分割

三维医学图像同时包含轴状位、冠状位和矢状位等多方向结构信息。为降低三维体数据建模开销，早期研究常采用多平面二维网络分别处理不同视图，再通过投票、平均或级联融合获得三维分割结果[15]。这类方法能够利用不同正交切面的互补信息，但不同视图通常由独立分支建模，参数量和后处理复杂度较高，且视图间交互多发生在预测层或特征融合末端。随后，多视图卷积和多视图Transformer方法进一步在特征层引入跨视图融合，但自注意力或多分支结构在高分辨率三维数据上仍可能带来较高显存开销。对于Mamba类视觉状态空间模型，直接将三维体特征展开为长序列能够保留完整体素序列关系，但序列长度随DHW增长，难以兼顾全局建模能力与部署效率。

与上述方法相比，本文ETSM并不为三个视图分别构建独立网络，而是将三维特征投影到轴状位、冠状位和矢状位三个二维视图后，使用参数共享的SS2D模块进行状态空间扫描，并在重建前通过轻量跨视图门控补充视图间信息。该设计的差异主要体现在三个方面：首先，三视图投影将三维长序列建模转化为多个二维视图建模，降低主要扫描序列规模；其次，参数共享避免了多视图独立分支带来的参数线性增加；最后，跨视图门控在特征重建前进行交互，使不同方向结构信息能够在解码恢复前得到融合。因此，ETSM更侧重在三维结构表达和计算开销之间取得折中。

### 三维医学图像分割中的解码恢复机制

在编码解码式医学图像分割网络中，编码器通过连续下采样获得高层语义特征，但下采样过程会不可避免地造成空间细节损失。对于脑肿瘤MRI分割任务而言，肿瘤区域常存在边界模糊、形态不规则和小目标区域占比低等问题。若解码阶段不能有效恢复空间结构和边界细节，容易出现肿瘤边缘分割不完整、增强肿瘤区域漏分以及水肿区域误分等现象。

跳跃连接是U-Net类网络中常用的解码恢复机制，其基本思想是将编码端浅层特征传递至解码端同尺度层级，使解码器在恢复空间分辨率时能够利用更丰富的局部细节信息。常见融合方式包括特征拼接和逐元素相加，其中拼接操作能够保留更多通道信息，因此在医学图像分割任务中应用较为广泛。然而，浅层编码特征虽然包含边缘、纹理等细节信息，但也可能包含噪声和与目标无关的背景响应；深层解码特征具有较强语义表达能力，但空间细节相对不足。因此，如何缓解浅层细节特征与深层语义特征之间的不一致，是提升解码恢复质量的重要问题。

此外，三维医学图像分割不仅需要恢复单个切片内的二维边界结构，还需要保持跨切片方向上的空间连续性。对于脑肿瘤区域而言，不同MRI序列之间存在明显互补信息，不同肿瘤子区域在形态、灰度和空间分布上也具有差异。因此，解码阶段应同时关注全局结构一致性和局部边界细节恢复。基于上述分析，本文在U-Mamba基础上进一步设计三视图高效建模与阶段感知解码恢复策略，以提升三维脑肿瘤MRI分割的准确性和计算效率。

## 2 本文方法

### 网络整体结构

本文方法整体采用编码器-解码器结构，并在U-Mamba基础上引入高效三视图状态空间建模、阶段感知解码和语义引导跳跃连接特征标定三个组成部分。给定输入多模态脑MRI图像：
X\in\mathbb{R}^{B\times C\times D\times H\times W}
其中，B表示批大小，C表示输入通道数，D、H和W分别表示三维图像的深度、高度和宽度。编码器通过多阶段下采样逐步提取多尺度特征，得到不同分辨率层级的编码表示；解码器通过逐层上采样恢复空间尺寸，并利用跳跃连接融合编码端同尺度特征，最终输出脑肿瘤分割结果。

与传统U-Net类网络不同，本文在编码阶段使用ETSM模块增强多尺度空间建模能力，使网络能够从轴状位、冠状位和矢状位三个方向提取互补空间信息。考虑到解码阶段不同层级特征的恢复需求不同，本文仅在由深层到浅层的前两个低分辨率解码阶段引入ETSM模块进行全局结构补偿，而在后续高分辨率解码阶段保留卷积细化能力，以降低额外计算开销并增强局部边界恢复。同时，语义引导跳跃连接特征标定模块也仅作用于低分辨率跳跃连接，高分辨率跳跃连接保持直接融合，以避免对局部纹理和边界细节产生过度调制。整体流程可表示为：
Y=f_{seg}\left(f_{dec}\left(f_{enc}\left(X\right)\right)\right)
其中，f_{enc}表示编码器，f_{dec}表示阶段感知解码器，f_{seg}表示最终分割预测头。整体网络架构如图3-1所示。

![overall_architecture_v4](paper_assets/overall_architecture_v4.png)

图3-1总体架构图
Figure 3-1 Overall Architecture Diagram

需要说明的是，图3-1为整体流程示意图，其中Skip Fusion Gate用于表示跳跃连接特征标定机制的作用方式；本文主模型中，解码端ETSM和Skip Fusion Gate均按阶段感知策略仅在低分辨率解码阶段启用。

### 高效三视图状态空间建模模块

三维医学图像天然包含多个正交方向的空间结构信息。若直接对三维体数据进行全局建模，序列长度会随D×H×W快速增长，导致计算复杂度和显存占用明显增加。为此，本文设计高效三视图状态空间建模模块，将三维特征分解为三个二维正交视图，并在二维空间中进行状态空间建模，以在保留多方向结构信息的同时降低三维建模开销。

从序列建模规模来看，若直接将三维特征展平后进行状态空间扫描，其序列长度可表示为L_{3D}=DHW。ETSM将三维特征分别投影到轴状位、冠状位和矢状位三个二维视图后，对应的扫描序列长度分别为L_a=HW、L_c=DW和L_s=DH。由于状态空间扫描的计算量通常与序列长度近似线性相关，直接三维扫描的主要时间复杂度可近似表示为O(DHW)，而三视图二维扫描的主要时间复杂度可近似表示为O(HW+DW+DH)。若三个视图分别使用独立SS2D模块，参数量会随视图数近似增加为3P_{SS2D}；本文采用参数共享方式，三视图扫描部分仍只引入P_{SS2D}级别参数，从而避免多视图独立建模带来的参数线性增加。由于Mamba/SS2D包含自定义选择性扫描算子，本文不对其精确FLOPs作伪精确估计，而主要从序列长度、参数共享和实测显存三个层面分析轻量化效果。

![etsm_block_v2](paper_assets/etsm_block_v2.png)

图3-2高效三视图状态空间建模模块。SS2D表示二维视觉状态空间扫描模块。
Figure 3-2 Efficient Tri-view State-space Modeling. SS2D denotes the two-dimensional visual state-space scanning module.

输入特征为：

F \in \mathbb{R}^{B \times C \times D \times H \times W}

ETSM首先沿不同空间维度进行平均池化投影，得到轴状位、冠状位和矢状位三个二维视图：

F_a = P_a(F), \quad F_c = P_c(F), \quad F_s = P_s(F)

其中，F_a、F_c、F_s分别表示axial、coronal和sagittal视图特征。随后，对三个二维视图进行状态空间扫描建模。对于编码端ETSM，本文采用非重叠局部窗口划分以降低高分辨率二维视图的扫描开销，窗口大小设置为8×8；当视图尺寸不能被窗口大小整除时，在高度和宽度方向进行零填充，扫描完成后再裁剪回原始尺寸。对于低分辨率解码端ETSM，由于特征尺寸较小，直接采用全局二维扫描。以窗口扫描形式为例，该过程可表示为：

U_a=\mathcal{W}(F_a),\quad U_c=\mathcal{W}(F_c),\quad U_s=\mathcal{W}(F_s)

\hat{F}_a=\mathcal{W}^{-1}\left(M_\theta(U_a)\right),\quad \hat{F}_c=\mathcal{W}^{-1}\left(M_\theta(U_c)\right),\quad \hat{F}_s=\mathcal{W}^{-1}\left(M_\theta(U_s)\right)

其中，\mathcal{W}(\cdot)表示非重叠窗口划分操作，用于将二维视图划分为若干8×8局部窗口；\mathcal{W}^{-1}(\cdot)表示窗口还原操作，用于将扫描后的局部窗口恢复为原二维视图布局；M_\theta表示参数为\theta的共享SS2D扫描模块。参数共享能够避免为三个视图分别引入独立模型参数，从而降低多视图建模带来的额外开销。

为增强不同视图之间的信息交互，本文在三个视图完成二维扫描后引入轻量级跨视图交互门控。首先，将三个视图临时广播并融合为三维特征F_m，然后通过Conv3d、GELU激活函数和Conv3d生成视图相关的门控响应：

T=Conv_{3d}^{(2)}\left(GELU\left(Conv_{3d}^{(1)}(F_m)\right)\right)

其中，GELU表示高斯误差线性单元。门控响应T被划分为三个方向的调节分量，并通过tanh函数约束到[-1,1]范围后分别投影回对应二维视图，用于对\hat{F}_a、\hat{F}_c和\hat{F}_s进行残差式修正。该过程先临时聚合三视图三维信息，再生成面向不同视图的调节权重，使轴状位、冠状位和矢状位结构信息能够在最终重建前进行补充。

在得到交互增强后的三个视图特征后，ETSM将其重新映射回三维空间。对于每个视图，首先沿其缺失维度进行广播，得到三维特征：

\tilde{F}_a, \quad \tilde{F}_c, \quad \tilde{F}_s \in \mathbb{R}^{B \times C \times D \times H \times W}

随后，采用门控融合方式自适应整合三个视图的信息：

G = Softmax\left( Conv_{1 \times 1 \times 1}\left([\tilde{F}_a, \tilde{F}_c, \tilde{F}_s]\right)\right)

F_r = G_a \odot \tilde{F}_a + G_c \odot \tilde{F}_c + G_s \odot \tilde{F}_s

其中，G_a、G_c、G_s分别表示三个视图对应的空间门控权重，\odot表示逐元素乘法。最后，通过残差连接得到输出特征：

F_{out} = F + F_r

通过上述方式，ETSM将三维空间建模转化为多个二维视图建模，使模型能够以较低代价整合轴状位、冠状位和矢状位的互补结构信息。

### 阶段感知解码恢复策略

编码阶段引入ETSM模块后，网络能够获得更强的空间上下文表达能力。然而，脑肿瘤分割不仅依赖深层全局语义信息，还需要在解码阶段恢复局部边界和细节结构。若在所有解码层均使用ETSM模块，虽然能够增强空间建模能力，但会引入额外计算开销；同时，高分辨率层中过强的全局建模可能削弱局部纹理细化能力。基于此，本文提出阶段感知解码恢复策略，根据解码阶段的空间分辨率差异选择不同的特征恢复方式。

解码器共有N个阶段，第i个解码阶段的输入包括上一层解码特征D_{i+1}以及编码端对应尺度的跳跃连接特征S_i。首先，对上一层解码特征进行上采样，得到与当前尺度相匹配的解码特征：

D_i^{up}=Up\left(D_{i+1}\right)

然后，将跳跃连接特征与上采样后的解码特征进行融合，得到当前阶段的输入特征：

Z_i = \operatorname{Concat}(D_i^{up}, \tilde{S}_i)

其中，\tilde{S}_i表示经过语义引导标定后的跳跃连接特征，其具体生成方式将在2.4节中介绍。

对于低分辨率解码阶段，特征具有较强语义表达能力，但空间细节相对不足。此时引入ETSM模块有助于建模跨切面的全局结构关系，增强肿瘤区域的整体形态恢复能力。对于高分辨率解码阶段，特征逐渐接近原始图像分辨率，更需要保留局部纹理和边界细节，因此采用卷积模块进行细化恢复。阶段感知解码过程可表示为：

D_i=\left\{\begin{matrix}ETSM\left(Conv\left(Z_i\right)\right),&i\in\mathrm{\Omega}_d\\Conv\left(Z_i\right),&i\notin\mathrm{\Omega}_d\end{matrix}\right.

其中，\mathrm{\Omega}_d表示使用ETSM的低分辨率解码阶段集合。本文实现中，\mathrm{\Omega}_d={0,1}，即按照由深层到浅层的解码顺序，在前两个低分辨率解码阶段使用ETSM，在后续高分辨率阶段采用卷积恢复。该设置主要基于特征分辨率和计算开销的权衡：低分辨率阶段特征尺寸较小，适合引入全局结构建模；高分辨率阶段更关注局部边界和纹理细化，继续使用ETSM会显著增加显存和时间开销。消融实验中，阶段感知解码在HD95指标上较仅编码端ETSM有所改善，也说明低分辨率阶段补充全局结构信息对边界距离误差具有一定作用。

### 语义引导的跳跃连接特征标定

传统U-Net类网络通常直接将编码端跳跃特征与解码端特征进行拼接。编码端跳跃特征包含丰富的边缘、纹理和空间位置信息，但也可能包含背景噪声以及与肿瘤无关的响应；解码端特征经过深层语义建模后，对肿瘤区域具有更强的类别和结构感知能力。因此，本文借鉴注意力门控思想[12]，利用解码端语义信息引导跳跃连接特征选择，使传入解码器的同尺度编码特征更加关注与肿瘤分割相关的区域。与图3-1中的示意形式一致，本文仅在低分辨率解码阶段启用该模块，高分辨率跳跃连接仍采用直接拼接，以保留局部边界和纹理细节。

![skip_fusion_gate](paper_assets/skip_fusion_gate.png)

图3-3语义引导跳跃连接特征标定模块。IN表示实例归一化，GELU表示高斯误差线性单元，Sigmoid用于生成空间门控权重。
Figure 3-3 Semantic-guided skip feature calibration module. IN denotes instance normalization, GELU denotes Gaussian error linear unit, and Sigmoid is used to generate spatial gating weights.

第i个解码阶段中，编码端跳跃连接特征为S_i，上采样后的解码特征为D_i^{up}。首先，将二者在空间尺寸上进行对齐，并沿通道维度拼接：
Q_i=Concat\left(S_i,D_i^{up}\right)
随后，拼接特征依次经过Conv3d、InstanceNorm3d、GELU、Conv3d和Sigmoid生成空间门控图，并将Sigmoid输出线性映射至[-1,1]：
G_i=2\sigma\left(Conv_2\left(GELU\left(IN\left(Conv_1\left(Q_i\right)\right)\right)\right)\right)-1

其中，IN表示实例归一化，GELU表示高斯误差线性单元，\sigma表示Sigmoid激活函数，G_i表示由解码语义和编码细节共同生成的空间注意力权重。经过线性映射后，G_i的取值范围为[-1,1]，能够对跳跃连接特征进行双向调节。与直接使用门控图乘以跳跃特征不同，本文采用残差式特征标定方式，以避免训练初期对有效细节特征造成过度抑制：

\tilde{S}_i = S_i + \alpha S_i \odot G_i

其中，\alpha为可学习门控缩放系数，实验中初始化为0.1，用于控制训练初期门控扰动幅度；\odot表示逐元素乘法。经过标定后的跳跃连接特征再与上采样后的解码特征进行拼接，并作为当前解码阶段的输入：

Z_i = \mathrm{Concat}(D_i^{up}, \tilde{S}_i)

通过该设计，网络能够利用解码端高级语义信息对编码端跳跃特征进行筛选和增强，从而减少背景噪声对解码恢复的干扰，并提升肿瘤边界及小区域结构的表达能力。

### 损失函数

本文采用医学图像分割中常用的Dice损失与交叉熵损失联合优化网络。设网络预测结果为P，真实标签为Y，总损失函数定义为：

\mathcal{L}=\mathcal{L}_{Dice}+\mathcal{L}_{CE}

其中，Dice损失用于缓解前景与背景类别不平衡问题，交叉熵损失用于增强逐体素分类约束。Dice损失可表示为：

\mathcal{L}_{Dice}=1-\frac{2\sum_{i} p_iy_i+\epsilon}{\sum_{i} p_i+\sum_{i} y_i+\epsilon}

其中，p_i表示第i个体素的预测概率，y_i表示对应真实标签，ϵ为平滑项。联合损失能够同时约束区域重叠程度和体素级分类准确性，从而提升脑肿瘤多区域分割性能。



## 3 实验结果与分析

为验证所提方法在三维脑肿瘤MRI分割任务中的有效性，本节在BraTS脑肿瘤分割数据集上开展实验。实验从分割精度、五折稳定性、消融分析、模型复杂度和可视化结果等方面进行评价，并与典型分割方法进行比较。

### 实验环境与数据集

本文实验基于PyTorch深度学习框架实现，网络训练和推理均在GPU环境下完成。实验硬件环境包括NVIDIA GeForce RTX3090型GPU和24GB显存。软件环境包括Python3.11.8、PyTorch2.2.1、CUDA12.1和nnU-Netv2，开发工具为PyCharm。

本文采用BraTS2020脑肿瘤分割数据集[13-14]进行实验验证。该数据集包含多模态脑部MRI图像，每个病例均包括T1、T1ce、T2和FLAIR四种模态。不同模态能够从不同角度反映脑组织及肿瘤区域特征，其中T1ce对增强肿瘤区域具有较好显示效果，T2和FLAIR对水肿区域较为敏感。本文按照BraTS常用评价方式，将分割区域划分为增强肿瘤区域（enhancing tumor，ET）、肿瘤核心区域（tumor core，TC）和全肿瘤区域（whole tumor，WT）。其中，WT包含所有肿瘤相关区域，TC包含坏死、非增强肿瘤和增强肿瘤区域，ET表示增强肿瘤区域。实验中，所有MRI图像均按照nnU-Net框架进行预处理，包括重采样、裁剪、强度归一化和数据增强等操作。数据集划分方式为训练集295例、验证集74例。

### 参数设置

训练过程中，采用nnU-Net自动规划得到的3d_fullres配置，输入图像patch size设置为128×128×128，batch size设置为2。优化器采用带Nesterov动量的SGD，初始学习率设置为0.01，动量为0.99，权重衰减系数为3×10^-5，并采用Poly学习率衰减策略。训练轮数设置为150。损失函数采用Dice损失与交叉熵损失的组合。为降低过拟合风险，训练过程中采用随机旋转、随机缩放、强度扰动和镜像翻转等数据增强策略。

### 评价指标

为全面评价脑肿瘤分割性能，本文采用Dice相似系数和95%Hausdorff距离（HD95）作为主要评价指标。Dice相似系数用于衡量预测分割区域与真实标注区域之间的重叠程度，定义为：

Dice=\frac{2\left|P\cap G\right|}{\left|P\right|+\left|G\right|}

其中，P表示模型预测区域，G表示真实标注区域。Dice值越高，表示分割结果与真实标注越接近。

HD95用于衡量预测边界与真实边界之间的距离误差，相比普通Hausdorff距离能够减少少量异常点对评价结果的影响。HD95值越低，表示模型在边界区域的分割效果越好。本文分别统计ET、TC和WT三个区域的Dice和HD95，并计算平均结果。

### 对比实验

为进一步分析本文方法的性能，本文将所提方法与典型医学图像分割网络进行对比，包括U-Net[1]、nnU-Net[3]、TransUNet[4]、TransBTS[5]、Swin-Unet[6]和U-Mamba[9]等。其中，U-Mamba和本文方法采用相同数据划分与训练设置，其余方法结果参考公开文献或已有实验结果。由于不同方法的训练策略和实现细节可能存在差异，相关结果主要用于性能参考。

表4-1不同方法的Dice对比

**Table 4-1 Comparison of Dice coefficients among different methods**

| 方法      | WT/%      | TC/%      | ET/%      | Mean/%    |
| --------- | --------- | --------- | --------- | --------- |
| U-Net     | 88.71     | 81.02     | 75.93     | 81.89     |
| nnU-Net   | 88.71     | 85.45     | **81.56** | 85.24     |
| TransUNet | 87.41     | 82.75     | 79.83     | 83.33     |
| Swin-Unet | 87.95     | 83.31     | 80.02     | 83.76     |
| TransBTS  | 85.73     | 81.26     | 77.81     | 81.60     |
| U-Mamba   | 91.28     | 87.31     | 77.22     | 85.27     |
| 本文方法  | **91.61** | **87.47** | 77.51     | **85.53** |

从表4-1可以看出，本文方法在U-Mamba基线模型的基础上进一步提升了WT、TC和ET三个区域的Dice指标。其中，平均Dice由85.27%提升至85.53%，提升0.26个百分点。由于U-Mamba本身融合了卷积局部建模与Mamba长程依赖建模能力，在BraTS2020数据集上已经取得较强的区域重叠性能，因此在Dice指标上继续获得大幅提升较为困难。本文方法在三个肿瘤子区域上均保持小幅增益，表明高效三视图状态空间建模、阶段感知解码和跳跃连接特征标定对区域重叠精度具有一定补充作用。

表4-2不同方法在HD95对比

**Table 4-2 Comparison of HD95 among different methods**

| 方法      | WT       | TC       | ET       | Mean     |
| --------- | -------- | -------- | -------- | -------- |
| U-Net     | 6.15     | 9.33     | 20.76    | 12.08    |
| nnU-Net   | 5.35     | 6.44     | 7.46     | 6.42     |
| TransUNet | 5.87     | 6.82     | 7.59     | 6.76     |
| Swin-Unet | 6.73     | 7.59     | 8.35     | 7.75     |
| TransBTS  | 7.03     | 8.52     | 9.37     | 8.31     |
| U-Mamba   | 4.79     | 4.83     | 4.20     | 4.61     |
| 本文方法  | **4.04** | **4.28** | **3.91** | **4.08** |

由表4-2可以看出，本文方法在WT、TC和ET三个区域上均取得更低的HD95。平均HD95由4.606降低至4.077，相对下降约11.49%。与Dice主要衡量区域重叠程度不同，HD95更加关注预测边界与真实边界之间的距离误差，对少量边界漏分、远端误分和轮廓不连续现象更敏感。脑肿瘤MRI分割中病灶常呈现边界模糊、形态不规则和跨切片变化明显等特点，单纯Dice提升有限并不意味着结构定位没有改善。本文方法在HD95指标上的下降表明，所提出的三视图上下文建模和语义引导跳跃连接标定更主要地改善了肿瘤轮廓一致性与边界定位质量。

### 五折交叉验证结果分析

为了进一步分析模型在不同数据划分下的稳定性，本文统计了U-Mamba与本文方法在五折验证集上的实验结果，如表4-3所示。

表4-3五折Mean Dice与Mean HD95对比

**Table 4-3 Comparison of 5-fold Mean Dice and Mean HD95**

| Fold | 方法     | Mean Dice/% | Mean HD95 |
| ---- | -------- | ----------- | --------- |
| f0   | U-Mamba  | 88.61       | 4.376     |
| f0   | 本文方法 | 88.86       | 3.444     |
| f1   | U-Mamba  | 84.95       | 5.451     |
| f1   | 本文方法 | 85.96       | 3.58      |
| f2   | U-Mamba  | 85.84       | 4.225     |
| f2   | 本文方法 | 86.52       | 3.614     |
| f3   | U-Mamba  | 83.18       | 4.873     |
| f3   | 本文方法 | 82.58       | 5.951     |
| f4   | U-Mamba  | 83.77       | 4.103     |
| f4   | 本文方法 | 83.73       | 3.795     |
| 平均 | U-Mamba  | 85.27       | 4.606     |
| 平均 | 本文方法 | 85.53       | 4.077     |

从表4-3可以看出，本文方法在f0、f1和f2上均取得更高的Mean Dice，并且HD95下降；在f4上，本文方法的Dice与基线基本持平，但HD95仍有所改善；在f3上，本文方法的Mean Dice由83.18%下降至82.58%，Mean HD95由4.873升高至5.951，说明模型在部分数据划分上仍存在一定波动。为进一步分析f3折退化来源，本文对该折74个验证病例进行了逐病例统计。结果显示，本文方法在42例病例上的Mean Dice高于U-Mamba，在32例病例上低于U-Mamba，且逐病例Mean Dice差值的中位数为0.0009，说明f3折并非整体性普遍退化，而是受到少数困难病例的明显影响。

为进一步定位f3折退化来源，本文按真实ET体素数对验证病例进行分组，并统计本文方法相对U-Mamba的指标差值，结果如表4-4所示。

表4-4 f3折按ET体积分组的性能差异分析

**Table 4-4 Performance difference analysis of fold f3 grouped by ET volume**

| ET体积分组        | 病例数 | Mean Dice差值 | Mean HD95差值 | ET Dice差值 | ET HD95差值 |
| ----------------- | ------ | ------------- | ------------- | ----------- | ----------- |
| ET=0              | 6      | +0.0018       | +1.1456       | 0.0000      | -           |
| 0<ET<1000         | 5      | -0.1276       | +16.2781      | -0.1333     | +21.2189    |
| 1000<=ET<5000     | 14     | +0.0035       | +1.7957       | -0.0011     | +1.7875     |
| ET>=5000          | 49     | +0.0027       | -0.6266       | +0.0027     | -0.0065     |

注：差值均表示“本文方法-U-Mamba”；Dice差值越大表示重叠精度越高，HD95差值越小表示边界距离误差越低。

由表4-4可见，f3折性能下降主要集中在0<ET<1000的小体积增强肿瘤病例中。该组仅包含5例，但Mean Dice平均下降0.1276，Mean HD95平均升高16.2781，ET Dice下降0.1333，ET HD95升高21.2189，说明极小体积ET区域的漏检或弱响应会对该折平均结果产生较大影响。相比之下，当ET体积大于等于5000个体素时，本文方法的Mean Dice平均提升0.0027，Mean HD95平均下降0.6266，说明本文方法在较大病灶上并未出现同样的退化趋势。以BraTS20_Training_087为例，该病例真实ET体积仅为406个体素，本文方法对ET区域的预测体素数仅为2，导致该病例Mean Dice由U-Mamba的0.5072下降至0，逐病例Mean HD95由9.24升高至94.95。该异常病例对f3折平均性能产生了较大影响，也说明当前固定阶段的三视图建模和跳跃特征标定策略在极小病灶或弱响应区域上仍存在不稳定性。

整体来看，虽然个别fold存在退化，本文方法在五折平均结果上仍优于U-Mamba基线模型，尤其在HD95指标上改善更加明显。这表明所提出的三视图空间建模和跳跃连接特征标定策略总体上对肿瘤边界恢复具有积极作用。同时，f3折逐病例统计也提示当前方法对极小体积增强肿瘤、低对比度病灶和复杂病例分布的适应性仍有进一步提升空间。后续可结合病灶尺度感知、视图权重自适应或病例难度建模来增强模型稳定性，并通过局部放大可视化进一步展示典型失败病例的分割差异。

### 消融实验

为分析各模块对模型性能的影响，本文设计消融实验。以U-Mamba为基础模型，逐步加入编码端ETSM、阶段感知解码策略和语义引导跳跃连接特征标定模块，以验证不同结构设计对模型性能的影响。

表4-5第0折消融实验结果

**Table 4-5 Fold-0 ablation experiment results**

| 方法                      | WT/%      | TC/%      | ET/%      | Mean Dice/% | Mean HD95 |
| ------------------------- | --------- | --------- | --------- | ----------- | --------- |
| U-Mamba                   | 91.86     | 90.24     | 83.74     | 88.61       | 4.376     |
| +ETSM                     | 91.86     | 89.61     | 84.10     | 88.53       | 4.577     |
| +ETSM+Stage               | 92.12     | 89.88     | 83.42     | 88.47       | 4.308     |
| +ETSM+Stage+Skip          | **92.30** | **90.03** | **84.26** | **88.86**   | **3.444** |

由表4-5可以看出，各模块的增益并非简单单调叠加。仅在编码器中引入ETSM后，模型在ET区域上由83.74%提升至84.10%，说明三视图状态空间建模有助于增强小区域肿瘤结构表达；但TC区域由90.24%下降至89.61%，导致平均Dice未超过基线。这表明单独增强编码端全局上下文后，不同肿瘤子区域的响应可能出现差异，尤其是肿瘤核心与增强肿瘤之间的结构关系仍需要解码阶段进一步协调。

加入阶段感知解码策略后，WT区域提升至92.12%，Mean HD95由4.577降低至4.308，说明在低分辨率解码阶段引入ETSM有助于改善肿瘤整体结构恢复和边界距离误差。但该设置下ET区域下降至83.42%，Mean Dice仍低于基线，说明仅依赖低分辨率全局结构补偿可能不足以稳定恢复细粒度增强肿瘤区域。

在此基础上进一步加入语义引导跳跃连接特征标定后，模型取得本组消融实验中的最佳结果，WT、TC和ET分别达到92.30%、90.03%和84.26%，Mean Dice提升至88.86%，Mean HD95降低至3.444。该结果说明，跳跃连接中的浅层特征虽然包含丰富空间细节，但也可能混入背景噪声和语义不一致响应。利用解码端高级语义对skip feature进行自适应标定后，模型能够更有效地融合全局结构信息与局部细节信息，从而有助于同时改善区域重叠精度和边界定位质量。

### 模型复杂度对比

为分析本文方法的计算开销，本文统计了不同模型的参数量、单patch前向推理时间和峰值显存占用，结果如表4-6所示。推理时间在NVIDIA GeForce RTX3090 GPU上测试，输入为nnU-Net自动规划得到的单个128×128×128 patch，batch size为1，采用FP32精度和`torch.no_grad()`模式，测试前进行5次预热，并统计20次前向传播的平均耗时；峰值显存由CUDA峰值显存统计获得。

表4-6不同模型复杂度对比

**Table 4-6 Comparison of Different Model Complexities**

| Model    | Params/M | Forward time/s | Peak memory/GB |
| -------- | -------- | -------------- | -------------- |
| nnU-Net  | 31.20    | 0.059          | 2.04           |
| U-Mamba  | 42.75    | 0.184          | 4.22           |
| 本文方法 | 37.10    | 0.171          | 2.46           |

由表4-6可知，本文方法的参数量为37.10M，低于U-Mamba的42.75M，参数规模减少约13.22%。在推理效率方面，本文方法的单patch前向推理时间为0.171s，相较于U-Mamba的0.184s略有降低；峰值显存占用由4.22GB降低至2.46GB，降幅约为41.71%。该结果与ETSM的结构设计相一致：一方面，三视图二维扫描将主要序列建模规模由DHW级别转化为HW+DW+DH级别；另一方面，参数共享的SS2D模块避免了为不同视图分别构建独立状态空间模块。此外，阶段感知解码策略仅在低分辨率解码阶段引入ETSM，避免在高分辨率阶段引入过多额外计算。因此，本文方法在保持Mamba长程建模能力的同时，能够降低模型参数规模和显存开销，具有较好的部署友好性。与nnU-Net相比，本文方法由于引入状态空间建模结构，参数量和推理开销有所增加，但在平均Dice和HD95指标上取得一定改善，体现了精度与复杂度之间的折中。

### 可视化分析

为直观分析本文方法的分割效果，选取典型病例进行可视化比较。图4-1展示了不同方法在脑肿瘤MRI图像上的分割结果。可以观察到，基础U-Mamba在肿瘤边界模糊区域容易出现漏分或边界不连续现象；本文方法能够更完整地分割肿瘤核心和水肿区域，并在增强肿瘤等小区域上保持较好的结构一致性。

特别是在肿瘤边界复杂、形态不规则以及跨切片变化明显的病例中，本文方法能够获得更加连续的分割结果。这主要得益于ETSM模块对三维多方向上下文信息的建模能力，以及阶段感知解码策略和语义引导跳跃连接特征标定模块对结构恢复和边界表达的增强作用。

![brats_visualization_all_modalities5](paper_assets/brats_visualization_all_modalities5.png)

图4-1可视化结果对比

**Figure 4-1 Comparison of visualization results**

综合定量结果和可视化结果可知，本文方法能够在计算复杂度可控的前提下改善三维脑肿瘤MRI分割结果，尤其在边界细节和结构完整性方面具有一定优势。

## 4 结语

本文针对三维脑肿瘤MRI分割中空间上下文建模开销较大、跨切面信息利用不足以及解码阶段细节恢复不充分等问题，提出了一种融合高效三视图状态空间建模与阶段感知解码的脑肿瘤MRI分割方法。该方法以U-Mamba为基础网络，在编码阶段将三维特征映射到轴状位、冠状位和矢状位三个二维视图，并通过参数共享的视觉状态空间模块进行高效建模，以增强多方向空间上下文表达能力。在解码阶段，根据不同分辨率层级的特征恢复需求设计阶段感知解码策略，在低分辨率层利用ETSM模块补偿全局结构信息，在高分辨率层保留卷积细化能力。同时，通过语义引导的跳跃连接特征标定模块，对编码端浅层细节特征进行自适应筛选，缓解浅层特征与深层语义之间的不一致问题。

在BraTS2020脑肿瘤分割数据集上的实验结果表明，本文方法相较于U-Mamba基线模型在平均Dice和HD95指标上均取得一定改善。五折平均Dice由85.27%提升至85.53%，平均HD95由4.606降低至4.077，说明该方法在保持区域分割精度的同时，能够改善肿瘤边界定位质量。模型复杂度实验表明，本文方法的参数量由U-Mamba的42.75M降低至37.10M，峰值显存占用由4.22GB降低至2.46GB，具有较好的轻量化效果和部署友好性。消融实验结果表明，在ETSM和阶段感知解码基础上引入语义引导跳跃连接特征标定后，模型取得本组消融实验中的最佳结果，说明解码端语义约束对跳跃特征筛选和结构恢复具有积极作用。

尽管本文方法取得了一定效果，但仍存在不足。首先，本文方法在部分数据划分上的性能仍存在波动，说明模型对不同病例分布的适应性仍有提升空间。其次，ETSM采用平均池化方式生成三视图特征，虽然能够降低序列建模开销，但也可能削弱极小病灶或低对比度区域的细粒度响应。再次，当前阶段感知策略采用固定的低分辨率阶段集合，尚不能根据肿瘤尺度、边界模糊程度或病例难度自适应调整建模范围。此外，本文方法未显式引入边界损失或不确定性约束，对复杂模糊边界的进一步建模仍有提升空间。未来工作将从多中心和多年份数据集验证、自适应视图权重学习、病灶尺度感知解码、边界细节约束以及模型量化或蒸馏等方面进一步完善模型，并探索与临床交互式辅助标注流程结合的可能性。

## 参考文献

[1] RONNEBERGER O,FISCHER P,BROX T.U-Net:Convolutional Networks for Biomedical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015.Cham:Springer,2015:234-241.DOI:10.1007/978-3-319-24574-4_28.

[2] ÇIÇEK Ö,ABDULKADIR A,LIENKAMP S S,et al.3D U-Net:Learning Dense Volumetric Segmentation from Sparse Annotation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2016.Cham:Springer,2016:424-432.DOI:10.1007/978-3-319-46723-8_49.

[3] ISENSEE F,JAEGER P F,KOHL S A A,et al.nnU-Net:a self-configuring method for deep learning-based biomedical image segmentation[J].Nature Methods,2021,18(2):203-211.DOI:10.1038/s41592-020-01008-z.

[4] CHEN J,LU Y,YU Q,et al.TransUNet:Transformers Make Strong Encoders for Medical Image Segmentation[EB/OL].arXiv:2102.04306,2021.https://arxiv.org/abs/2102.04306.

[5] WANG W,CHEN C,DING M,et al.TransBTS:Multimodal Brain Tumor Segmentation Using Transformer[EB/OL].arXiv:2103.04430,2021.https://arxiv.org/abs/2103.04430.

[6] CAO H,WANG Y,CHEN J,et al.Swin-Unet:Unet-like Pure Transformer for Medical Image Segmentation[EB/OL].arXiv:2105.05537,2021.https://arxiv.org/abs/2105.05537.

[7] HATAMIZADEH A,TANG Y,NATH V,et al.UNETR:Transformers for 3D Medical Image Segmentation[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.Waikoloa:IEEE,2022:1748-1758.

[8] GU A,DAO T.Mamba:Linear-Time Sequence Modeling with Selective State Spaces[EB/OL].arXiv:2312.00752,2023.https://arxiv.org/abs/2312.00752.

[9] MA J,LI F,WANG B.U-Mamba:Enhancing Long-range Dependency for Biomedical Image Segmentation[EB/OL].arXiv:2401.04722,2024.https://arxiv.org/abs/2401.04722.

[10] XING Z,YE T,YANG Y,et al.SegMamba:Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation[EB/OL].arXiv:2401.13560,2024.https://arxiv.org/abs/2401.13560.

[11] LIU Y,TIAN Y,ZHAO Y,et al.VMamba:Visual State Space Model[EB/OL].arXiv:2401.10166,2024.https://arxiv.org/abs/2401.10166.

[12] OKTAY O,SCHLEMPER J,LE FOLGOC L,et al.Attention U-Net:Learning Where to Look for the Pancreas[EB/OL].arXiv:1804.03999,2018.https://arxiv.org/abs/1804.03999.

[13] MENZE B H,JAKAB A,BAUER S,et al.The Multimodal Brain Tumor Image Segmentation Benchmark(BRATS)[J].IEEE Transactions on Medical Imaging,2015,34(10):1993-2024.DOI:10.1109/TMI.2014.2377694.

[14] BAKAS S,AKBARI H,SOTIRAS A,et al.Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features[J].Scientific Data,2017,4:170117.DOI:10.1038/sdata.2017.117.

[15] PANDEY S,CHANGDAR S,PERSLEV M,et al.Fully Automated Tumor Segmentation for Brain MRI data using Multiplanner UNet[EB/OL].arXiv:2401.06499,2024.https://arxiv.org/abs/2401.06499.
