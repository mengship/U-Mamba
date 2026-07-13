基于高效三视图状态空间建模的脑肿瘤MRI分割网络

摘要：针对脑肿瘤磁共振成像（MRI）分割中跨切面利用不足和体建模开销较大的问题，提出高效三视图状态空间网络。网络将三维特征投影为3个正交视图，以共享二维选择性扫描建模，并通过阶段感知解码和语义引导跳跃标定恢复细节。BraTS2020五折实验中，平均Dice提高0.26个百分点，平均95%豪斯多夫距离（HD95）、参数量和峰值显存较U-Mamba分别降低11.49%、13.22%和41.71%。结果表明，该方法保持了分割精度，并降低了状态空间基线的边界误差和资源开销。

关键词：脑肿瘤分割；磁共振成像；状态空间模型；高效三视图状态空间建模；阶段感知解码；跳跃连接标定

中图分类号：TP391.41　文献标志码：A

**Brain tumor MRI segmentation network based on efficient tri-view state-space modeling**

**Abstract:** To address insufficient cross-slice information utilization and high volumetric modeling cost in three-dimensional brain tumor magnetic resonance imaging (MRI) segmentation, an efficient tri-view state-space network was proposed. Three-dimensional features were projected into three orthogonal views and modeled by a shared two-dimensional selective scan. Stage-aware decoding and semantic-guided skip calibration were employed for detail recovery. In five-fold experiments on BraTS2020, the average Dice score increased by 0.26 percentage points, while the average 95% Hausdorff distance (HD95), parameter count, and peak memory decreased by 11.49%, 13.22%, and 41.71%, respectively, compared with U-Mamba. The results indicate that the method maintains segmentation accuracy while reducing the boundary error and resource consumption of the state-space baseline.

**Key words:** brain tumor segmentation; magnetic resonance imaging; state space model; efficient tri-view state-space modeling; stage-aware decoding; skip feature calibration

**0**　引言

脑肿瘤MRI病灶具有形态不规则、边界模糊和组织异质性强等特点。准确分割增强肿瘤、肿瘤核心和水肿区域，可辅助肿瘤范围判断、治疗规划与疗效评估，但人工勾画耗时且受医生经验影响，因此自动分割具有重要应用价值。

U-Net[1]、3D U-Net[2]和nnU-Net[3]等编码解码网络已广泛用于医学图像分割，但卷积局部感受野对三维长程依赖的建模能力有限。Transformer方法[4-7]能够建立全局关联，却在高分辨率体数据上产生较高开销。Mamba[8]以选择性状态空间模型处理长序列，U-Mamba[9]和SegMamba[10]验证了其在三维医学分割中的可行性，VMamba[11]进一步建立了二维视觉扫描范式。Attention U-Net[12]则表明解码语义可用于筛选跳跃特征。BraTS基准[13-14]推动了上述结构在多模态脑肿瘤分割中的验证。

多平面与体状态空间方法分别从切面分解和三维序列建模利用空间信息。Multiplanner U-Net[15]采用多平面预测融合，相关三维Mamba研究[16-17]直接建模体特征；三平面CNN和QuickNAT[18-19]也采用独立切面分支，存在参数重复和交互偏晚的问题。TransFuse、CoTr和nnFormer[20-22]将卷积与注意力结合；VM-UNet、Swin-UMamba和LightM-UNet[23-25]扩展了二维医学Mamba。李孟灵等[26]采用三平面滑动窗口状态空间建模，侯蓓蓓等[27]研究轻量级Transformer脑肿瘤分割，侯向宁等[28]将轻量Mamba用于3D U-Net。余唯一等[29]总结了MRI病灶分割的尺度差异与边界模糊问题，邵虹等[30]和姜舒等[31]分别从注意力门控、轴向注意力及局部—全局协同角度改进二维医学分割。

现有方法仍面临三点不足：直接处理三维体特征的序列规模随DHW增长；独立多视图分支缺少参数共享；解码端直接融合浅层特征可能引入背景噪声。因此，需要在控制体数据建模开销的同时协调跨切面上下文、局部边界与深层语义。

针对上述问题，提出高效三视图状态空间建模模块（Efficient Tri-view State-space Modeling，ETSM），将三维特征投影为三个正交二维视图并共享二维选择性扫描（2D selective scan，SS2D）参数；同时根据分辨率选择性部署ETSM，并利用解码端语义标定跳跃特征。

本文主要工作如下：

（1）提出ETSM，将主要扫描序列规模由DHW转化为HW+DW+DH，并通过三视图参数共享减少重复建模开销。

（2）设计阶段感知解码和语义引导跳跃特征标定，仅在低分辨率阶段补充全局建模与特征调节。

（3）通过BraTS2020五折验证、消融、病例级统计和资源测试，评估模型性能、稳定性及小体积病灶局限。

表1比较了相关方法与ETSM。ETSM在三个正交二维视图上共享SS2D参数，并在三维重建前融合视图信息。

表1　医学图像分割中多视图与长程建模方法对比

| 方法类别 | 代表方法 | 建模与融合方式 | 主要差异 |
| -------- | -------- | -------------- | -------- |
| 多平面CNN | 三平面CNN、QuickNAT、Multiplanner U-Net[15,18-19] | 独立切面建模，预测或级联融合 | 分支参数重复，跨视图交互偏晚 |
| 三平面医学Mamba | TP-WSSM[26] | 三视图窗口建模，加权融合 | 侧重编码端局部窗口建模 |
| CNN-Transformer融合 | TransFuse、CoTr、nnFormer、轻量级Transformer脑肿瘤分割[20-22,27] | 二维分支或三维特征注意力融合 | 并行分支或注意力计算开销较高 |
| 二维医学Mamba | VM-UNet、Swin-UMamba、LightM-UNet[23-25] | 二维特征状态空间扫描 | 对三维跨切片关系的显式建模有限 |
| 三维医学Mamba | U-Mamba、SegMamba、轻量级Mamba脑肿瘤分割等[9-10,16-17,28] | 三维体特征状态空间建模 | 主要计算规模与体数据空间尺寸相关 |
| 本文ETSM | 三视图投影与共享SS2D | 三视图共享扫描，重建前融合 | 主要扫描序列项为HW+DW+DH，但平均池化可能削弱极小病灶响应 |

**1**　本文方法

**1.1**　网络整体结构

本文以U-Mamba[9]为基础，引入ETSM、阶段感知解码和语义引导跳跃特征标定。输入为$X\in\mathbb{R}^{B\times C\times D\times H\times W}$，其中$B$、$C$及$D,H,W$分别表示批大小、通道数和空间尺寸。编码器提取多尺度特征，解码器仅在前两个低分辨率阶段使用ETSM和跳跃标定，其余阶段采用卷积恢复与直接融合。网络输出为：

\hat{Y}=f_{seg}\left(f_{dec}\left(f_{enc}\left(X\right)\right)\right) \qquad（1）

其中，$f_{enc}$、$f_{dec}$和$f_{seg}$分别表示编码器、阶段感知解码器和分割头。整体结构如图1所示。

![overall_architecture_v4](paper_assets/overall_architecture_v4.png)

图1　总体架构图。解码端ETSM与跳跃连接特征标定仅在由深层到浅层的前两个低分辨率阶段启用，其余阶段采用卷积恢复和直接跳跃融合。

**1.2**　高效三视图状态空间建模模块

状态空间模型通过隐状态递推建立长程依赖，其离散形式为：

h_t=\bar{A}h_{t-1}+\bar{B}x_t,\quad y_t=Ch_t+Dx_t \qquad（2）

其中，$x_t$、$h_t$和$y_t$分别为输入、隐状态和输出，$\bar{A}$、$\bar{B}$、$C$和$D$为模型参数。VMamba[11]通过二维选择性扫描扩展了视觉空间建模。若直接展平三维特征，序列长度为$DHW$；ETSM投影得到的轴状位、冠状位和矢状位序列长度分别为$HW$、$DW$和$DH$，选择性扫描主要序列项由$O(DHW)$变为$O(HW+DW+DH)$。该估计不包含投影、重建和门控开销，完整效率由参数量、前向时间和峰值显存衡量。三视图共享SS2D参数，避免独立分支约$3P_{SS2D}$的重复参数。

![etsm_block_v2](paper_assets/etsm_block_v2.png)

图2　高效三视图状态空间建模模块。SS2D表示二维视觉状态空间扫描模块；跨视图交互在实现中依次包含三路门控拆分、tanh约束和投影回对应二维视图。
给定$F\in\mathbb{R}^{B\times C\times D\times H\times W}$，沿不同维度平均池化得到三视图：

F_a=P_a(F),\quad F_c=P_c(F),\quad F_s=P_s(F) \qquad（3）

编码端采用8×8非重叠窗口，尺寸不足时先填充、扫描后裁剪；低分辨率解码端采用全局扫描。共享SS2D处理过程为：

U_v=\mathcal{W}(F_v),\quad \hat{F}_v=\mathcal{W}^{-1}\left(M_\theta(U_v)\right),\quad v\in\{a,c,s\} \qquad（4）

其中，$\mathcal{W}$和$\mathcal{W}^{-1}$分别表示窗口划分与还原，$M_\theta$为参数共享的SS2D模块。

扫描后的三视图临时广播并融合为$F_m$，经两层三维卷积生成门控响应：

T=Conv_{3d}^{(2)}\left(GELU\left(Conv_{3d}^{(1)}(F_m)\right)\right) \qquad（5）

响应$T$被拆分为三个分量，经$tanh$约束后投影回对应视图并残差修正。增强后的视图广播为$\tilde{F}_a,\tilde{F}_c,\tilde{F}_s$，采用门控融合并与输入残差相加：

G=Softmax\left(Conv_{1\times1\times1}[\tilde{F}_a,\tilde{F}_c,\tilde{F}_s]\right),\quad F_{out}=F+\sum_{v\in\{a,c,s\}}G_v\odot\tilde{F}_v \qquad（6）

**1.3**　阶段感知解码恢复策略

解码阶段既需恢复整体结构，也需保留局部边界。设第$i$阶段的上采样特征和标定后跳跃特征分别为$D_i^{up}$与$\tilde{S}_i$，阶段感知解码为：

D_i^{up}=Up(D_{i+1}),\quad Z_i=Concat(D_i^{up},\tilde{S}_i),\quad D_i=\begin{cases}ETSM(Conv(Z_i)),&i\in\Omega_d\\Conv(Z_i),&i\notin\Omega_d\end{cases} \qquad（7）

本文设置$\Omega_d=\{0,1\}$，仅在前两个低分辨率阶段使用ETSM。全部阶段部署在128×128×128图像块和RTX 3090 24 GB条件下发生显存不足；前三阶段部署的第0折平均Dice和HD95为88.73%和4.249 mm，较前两阶段且未加入跳跃标定的配置（88.47%、4.308 mm）仅小幅改善。因此，该设置是当前实验条件下的资源折中，并非所有阶段组合的全局最优方案。

**1.4**　语义引导的跳跃连接特征标定

编码端跳跃特征包含细节，也可能引入背景噪声。借鉴注意力门控思想[12]，采用解码语义对低分辨率跳跃特征进行双向残差标定，高分辨率阶段仍直接拼接以保留边界细节。

![skip_fusion_gate](paper_assets/skip_fusion_gate.png)

图3　语义引导跳跃连接特征标定模块。IN表示实例归一化，GELU表示高斯误差线性单元；Sigmoid输出经2σ-1映射至[-1,1]后，再由初始化为0.1的可学习系数α进行残差缩放。
第$i$阶段将编码特征$S_i$与上采样特征$D_i^{up}$拼接，经Conv3d、实例归一化、GELU、Conv3d和Sigmoid生成门控图：

Q_i=Concat(S_i,D_i^{up}),\quad G_i=2\sigma\!\left(Conv_2(GELU(IN(Conv_1(Q_i))))\right)-1,\quad \tilde{S}_i=S_i+\alpha S_i\odot G_i \qquad（8）

其中，$G_i\in[-1,1]$，$\alpha$为初始化为0.1的可学习缩放系数。残差形式可减弱训练初期对有效细节的过度抑制。

**1.5**　损失函数

采用Dice损失与交叉熵损失联合优化：

\mathcal{L}=\mathcal{L}_{Dice}+\mathcal{L}_{CE},\quad \mathcal{L}_{Dice}=1-\frac{2\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon} \qquad（9）

其中，$p_i$和$y_i$分别为第$i$个体素的预测概率和真实标签，$\epsilon$为平滑项。



**2**　实验结果与分析

在BraTS2020数据集上，从分割精度、五折稳定性、消融、资源开销和可视化等方面评价所提方法。

**2.1**　实验环境与数据集

实验采用多模态脑肿瘤分割挑战赛2020（Brain Tumor Segmentation Challenge 2020，BraTS2020）数据集[13-14]，共369例T1、T1ce、T2和FLAIR四模态MRI，评价区域为全肿瘤（whole tumor，WT）、肿瘤核心（tumor core，TC）和增强肿瘤（enhancing tumor，ET）。图像按nnU-Net流程完成重采样、裁剪和强度归一化。采用五折交叉验证，前四折验证集各74例，第五折73例。实验基于Python 3.11.8、PyTorch 2.2.1、CUDA 12.1和nnU-Net v2实现，硬件为NVIDIA GeForce RTX 3090 24 GB显卡。

**2.2**　参数设置

nnU-Net、U-Mamba和本文方法均采用nnU-Net自动规划的3d_fullres配置，图像块为128×128×128，批大小为2，训练150轮。采用带Nesterov动量的随机梯度下降（stochastic gradient descent，SGD），初始学习率、动量和权重衰减分别为0.01、0.99和3×10^-5，并使用多项式学习率衰减以及随机旋转、缩放、强度扰动和镜像翻转。SegMamba采用官方实现，并对齐五折划分、训练轮数及评价口径。

**2.3**　评价指标

采用Dice相似系数评价区域重叠程度：

Dice=\frac{2\left|P\cap G\right|}{\left|P\right|+\left|G\right|} \qquad（10）

其中，$P$和$G$分别为预测区域和真实标注。HD95依据图像记录的体素间距计算表面距离，单位为mm；Dice越高、HD95越低表示性能越好。双侧为空和仅一侧为空的HD95分别记为NaN和Inf，均值仅统计有限病例，并另报空掩膜计数。表2—表4和表6先在折内按WT、TC和ET求均值再汇总；病例级检验则先对每例的可用区域求均值，因此两种聚合结果略有差异。

**2.4**　对比实验

将所提方法与nnU-Net[3]、U-Mamba[9]和SegMamba[10]比较。前三者采用相同五折划分、训练设置和评价流程；SegMamba采用官方实现并对齐主要实验口径。其他公开结果因训练划分和评价流程不可完全追溯，未纳入主表。

表2　不同方法的Dice对比

| 方法      | WT/%      | TC/%      | ET/%      | 平均/%    |
| --------- | --------- | --------- | --------- | --------- |
| nnU-Net   | 91.28     | 87.23     | **77.98** | 85.49     |
| SegMamba  | 91.23     | 85.25     | 76.37     | 84.28     |
| U-Mamba   | 91.28     | 87.31     | 77.22     | 85.27     |
| 本文方法  | **91.61** | **87.47** | 77.51     | **85.53** |

由表2可见，本文方法的平均Dice为85.53%，较U-Mamba提高0.26个百分点，WT、TC和ET均有小幅改善；相较SegMamba提高1.25个百分点，与nnU-Net结果接近。

表3　不同方法的HD95对比

| 方法      | WT/mm    | TC/mm    | ET/mm    | 平均/mm  |
| --------- | -------- | -------- | -------- | -------- |
| nnU-Net   | 4.46     | **4.25** | **3.42** | **4.04** |
| SegMamba  | 4.68     | 5.34     | 4.32     | 4.78     |
| U-Mamba   | 4.79     | 4.83     | 4.20     | 4.61     |
| 本文方法  | **4.04** | 4.28     | 3.91     | 4.08     |

表3（续）　ET区域HD95空掩膜统计

| 方法 | 有限HD95/例 | 双侧为空/例 | 仅真实标注为空/例 | 仅预测为空/例 |
| ---- | -----------: | ------------: | --------------------: | --------------: |
| nnU-Net | 339 | 8 | 19 | 3 |
| SegMamba | 341 | 8 | 19 | 1 |
| U-Mamba | 341 | 4 | 23 | 1 |
| 本文方法 | 340 | 4 | 23 | 2 |

注：HD95均值仅统计有限病例。WT在各方法中均有369个有限值；TC仅nnU-Net和本文方法分别有2例和1例预测为空，其余均为有限值。“仅真实标注为空”和“仅预测为空”分别对应假阳性和完全漏检。

由表3可见，本文方法平均HD95为4.077 mm，较U-Mamba下降11.49%，较SegMamba也更低；与nnU-Net总体接近，并在WT上更低。空掩膜统计表明，本文方法对ET小病灶及假阳性的稳定性仍需改进。

**2.5**　五折交叉验证结果分析

U-Mamba与本文方法的五折结果见表4。

表4　五折平均Dice与平均HD95对比

| 折次 | U-Mamba Dice/% | 本文Dice/% | U-Mamba HD95/mm | 本文HD95/mm |
| ---- | ---------------: | ----------: | ----------------: | -----------: |
| f0   | 88.61 | **88.86** | 4.376 | **3.444** |
| f1   | 84.95 | **85.96** | 5.451 | **3.580** |
| f2   | 85.84 | **86.52** | 4.225 | **3.614** |
| f3   | **83.18** | 82.58 | **4.873** | 5.951 |
| f4   | **83.77** | 83.73 | 4.103 | **3.795** |
| 平均 | 85.27 | **85.53** | 4.606 | **4.077** |

本文方法在f0—f2上同时提高Dice并降低HD95，在f4上Dice基本持平、HD95下降，但在f3上退化。U-Mamba与本文方法的折间Dice标准差分别为2.13和2.46个百分点，HD95标准差分别为0.556 mm和1.055 mm，表明本文方法的折间波动较高。对369个配对病例检验时，本文方法在227例上的Dice更高、142例更低，平均和中位差值分别为0.0028和0.0016；Wilcoxon符号秩检验$p=0.00014$，提示分布呈小幅正向偏移，但配对$t$检验$p=0.2179$，均值差异未显著。病例级HD95平均降低0.5733 mm，配对$t$检验和Wilcoxon检验分别为$p=0.1984$和$p=0.1077$，呈下降趋势但未显著。聚合口径差异见第2.3节。

为定位f3退化来源，按真实ET体素数统计本文方法相对U-Mamba的差值，结果见表5。

表5　f3折按ET体积分组的性能差异分析

| ET体积分组        | 病例数 | 平均Dice差值 | 平均HD95差值/mm | ET Dice差值 | ET HD95差值/mm |
| ----------------- | ------ | ------------- | ------------- | ----------- | ----------- |
| ET=0              | 6      | +0.0018       | +1.1456       | 0.0000      | -           |
| 0<ET<1000         | 5      | -0.1276       | +16.2781      | -0.1333     | +21.2189    |
| 1000≤ET<5000      | 14     | +0.0035       | +1.7957       | -0.0011     | +1.7875     |
| ET≥5000           | 49     | +0.0027       | -0.6266       | +0.0027     | -0.0065     |

注：差值均表示“本文方法-U-Mamba”；Dice差值越大表示重叠精度越高，HD95差值越小表示边界距离误差越低；“—”表示该组ET真实标注为空，ET HD95差值不适用，该组平均HD95差值由可用的WT和TC结果计算。

f3的下降主要集中在5例$0<ET<1000$的小体积病灶中，而$ET\geq5000$时未出现同类退化。BraTS20_Training_087的真实ET仅406个体素，本文方法预测2个体素，使病例平均Dice由0.5072降至0、HD95由9.24 mm升至94.95 mm。结果说明五折总体Dice略有改善、HD95呈下降趋势，但收益受小体积ET和病例分布影响，不宜表述为显著均值提升。

**2.6**　消融实验

以U-Mamba为基线，依次加入编码端ETSM、阶段感知解码（Stage）和跳跃特征标定（Skip），五折结果见表6。

表6　五折主线消融实验结果

| 方法             | 编码端ETSM | 阶段感知解码 | 跳跃标定 | 平均Dice/% | 平均HD95/mm |
| ---------------- | ------------ | ------------------- | ---------------- | ----------- | --------- |
| U-Mamba          | ×            | ×                   | ×                | 85.27±2.13  | 4.606±0.556 |
| +ETSM            | √            | ×                   | ×                | 85.21±2.02  | 4.263±0.992 |
| +ETSM+Stage      | √            | √                   | ×                | 85.23±2.20  | 4.603±1.198 |
| +ETSM+Stage+Skip | √            | √                   | √                | **85.53±2.46** | **4.077±1.055** |

各模块增益并非单调叠加。仅加入ETSM时Dice基本不变、HD95下降；加入Stage后未稳定改善；完整配置取得85.53%的Dice和4.077 mm的HD95，表明低分辨率全局建模需与浅层细节标定协同。由于采用累加式消融且未检验模块交互，不将结果解释为单一模块的独立显著增益。

在第0折机制筛查中，以Attention U-Net式单向门控替换残差标定后，Dice和HD95为88.46%和4.608 mm，本文配置为88.86%和3.444 mm。该结果仅说明残差双向调节在该折更优，仍需更多折次验证。

**2.7**　模型资源开销分析

在RTX 3090上统一输入1个4通道128×128×128图像块，以32位浮点精度（FP32）和无梯度模式预热5次，并统计20次前向传播的平均耗时及峰值显存，结果见表7。

表7　不同模型资源开销对比

| 模型    | 参数量/M | 前向推理时间/s | 峰值显存/GiB |
| -------- | -------- | -------------- | -------------- |
| nnU-Net  | **31.20** | **0.059**      | **2.04**       |
| U-Mamba  | 42.75    | 0.184          | 4.22           |
| SegMamba | 67.42    | 0.655          | 3.26           |
| 本文方法 | 37.10    | 0.171          | 2.46           |

相较U-Mamba，本文方法的参数量、推理时间和峰值显存分别降低13.22%、7.07%和41.71%；相较SegMamba分别降低44.97%、73.89%和24.56%。三视图共享扫描降低了状态空间基线开销，但投影、重建和门控仍有成本；nnU-Net在三项资源指标上仍占优势。因此，效率结论限定为相对U-Mamba和SegMamba的改善。

**2.8**　可视化分析

图4给出典型病例。在所选切片中，本文方法对部分肿瘤核心、水肿区域和不规则边界的预测较连续，但定性观察不替代五折结果。

![brats_visualization_all_modalities5](paper_assets/brats_visualization_all_modalities5.png)

图4　可视化结果对比

针对f3的小体积ET退化，图5给出BraTS20_Training_087、299和315的局部放大结果。三例均出现不同程度的小目标漏分或响应减弱，与表5的统计一致。

![brats_f3_failure_cases_zoom](paper_assets/brats_f3_failure_cases_zoom.png)

图5　f3折失败案例局部放大对比

失败案例表明，较低的五折汇总HD95并不意味着对所有病例均稳定，极小ET仍是主要局限。

**3**　总结

提出了一种高效三视图状态空间脑肿瘤MRI分割网络。该网络将三维特征投影至三个正交视图，以共享SS2D建模多方向上下文，并通过阶段感知解码和语义引导跳跃标定恢复结构细节。BraTS2020五折实验中，相较U-Mamba，平均Dice由85.27%增至85.53%，平均HD95由4.606 mm降至4.077 mm，参数量和峰值显存分别由42.75M、4.22 GiB降至37.10M、2.46 GiB。病例级检验显示Dice分布呈小幅正向偏移、HD95呈下降趋势，但均值差异未显著；消融结果也表明各模块需协同发挥作用。因此，性能结论应限定为在保持区域重叠精度的同时改善状态空间基线的边界距离误差和资源开销。

当前方法在不同折间仍有波动，平均池化投影和固定阶段部署对极小ET或弱响应病灶存在不稳定性，且未显式约束模糊边界。后续将开展多中心验证，并研究病灶尺度感知投影、自适应视图权重、边界约束及模型压缩，以提高小病灶鲁棒性和部署效率。

参考文献：

[1] RONNEBERGER O, FISCHER P, BROX T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015. Cham: Springer, 2015: 234-241.

[2] ÇIÇEK Ö, ABDULKADIR A, LIENKAMP S S, et al. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2016. Cham: Springer, 2016: 424-432.

[3] ISENSEE F, JAEGER P F, KOHL S A A, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation[J]. Nature Methods, 2021, 18(2): 203-211.

[4] CHEN J, MEI J, LI X, et al. TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers[J]. Medical Image Analysis, 2024, 97: 103280.

[5] WANG W, CHEN C, DING M, et al. TransBTS: Multimodal Brain Tumor Segmentation Using Transformer[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2021. Cham: Springer, 2021: 109-119.

[6] CAO H, WANG Y, CHEN J, et al. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation[C]//Computer Vision-ECCV 2022 Workshops. Cham: Springer, 2023: 205-218.

[7] HATAMIZADEH A, TANG Y, NATH V, et al. UNETR: Transformers for 3D Medical Image Segmentation[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. Waikoloa: IEEE, 2022: 1748-1758.

[8] GU A, DAO T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces[C]//Proceedings of the First Conference on Language Modeling. 2024.

[9] MA J, LI F, WANG B. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation[EB/OL]. arXiv:2401.04722, 2024. https://arxiv.org/abs/2401.04722.

[10] XING Z, YE T, YANG Y, et al. SegMamba: Long-range Sequential Modeling Mamba for 3D Medical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2024. Cham: Springer, 2024: 578-588.

[11] LIU Y, TIAN Y, ZHAO Y, et al. VMamba: Visual State Space Model[C]//Advances in Neural Information Processing Systems. 2024, 37: 103031-103063.

[12] OKTAY O, SCHLEMPER J, LE FOLGOC L, et al. Attention U-Net: Learning Where to Look for the Pancreas[EB/OL]. arXiv:1804.03999, 2018. https://arxiv.org/abs/1804.03999.

[13] MENZE B H, JAKAB A, BAUER S, et al. The Multimodal Brain Tumor Image Segmentation Benchmark(BRATS)[J]. IEEE Transactions on Medical Imaging, 2015, 34(10): 1993-2024.

[14] BAKAS S, AKBARI H, SOTIRAS A, et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features[J]. Scientific Data, 2017, 4: 170117.

[15] PANDEY S, CHANGDAR S, PERSLEV M, et al. Fully Automated Tumor Segmentation for Brain MRI data using Multiplanner UNet[EB/OL]. arXiv:2401.06499, 2024. https://arxiv.org/abs/2401.06499.

[16] WANG C, XIE Y, CHEN Q, et al. A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation[EB/OL]. arXiv:2503.19308, 2025. https://arxiv.org/abs/2503.19308.

[17] JI H. DM-SegNet: Dual-Mamba Architecture for 3D Medical Image Segmentation with Global Context Modeling[EB/OL]. arXiv:2506.05297, 2025. https://arxiv.org/abs/2506.05297.

[18] PRASOON A, PETERSEN K, IGEL C, et al. Deep Feature Learning for Knee Cartilage Segmentation Using a Triplanar Convolutional Neural Network[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2013. Berlin: Springer, 2013: 246-253.

[19] ROY A G, CONJETI S, NAVAB N, et al. QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy[J]. NeuroImage, 2019, 186: 713-727.

[20] ZHANG Y, LIU H, HU Q. TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2021. Cham: Springer, 2021: 14-24.

[21] XIE Y, ZHANG J, SHEN C, et al. CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention-MICCAI 2021. Cham: Springer, 2021: 171-180.

[22] ZHOU H Y, GUO J, ZHANG Y, et al. nnFormer: Interleaved Transformer for Volumetric Segmentation[J]. IEEE Transactions on Image Processing, 2023, 32: 4036-4045.

[23] RUAN J, XIANG S. VM-UNet: Vision Mamba UNet for Medical Image Segmentation[EB/OL]. arXiv:2402.02491, 2024. https://arxiv.org/abs/2402.02491.

[24] LIU J, YU R, WANG Y, et al. Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining[EB/OL]. arXiv:2402.03302, 2024. https://arxiv.org/abs/2402.03302.

[25] LIAO W, ZHU Y, WANG X, et al. LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation[J]. Neural Networks, 2024, 178: 106539.

[26] 李孟灵, 蔡冬怡, 王金强, 等. 基于三平面状态空间建模的三维医学图像分割模型[J/OL]. 计算机科学: 1-14[2026-07-11]. https://link.cnki.net/urlid/50.1075.tp.20260109.1648.023.

LI Mengling, CAI Dongyi, WANG Jinqiang, et al. Three-dimensional medical image segmentation model based on tri-planar state-space modeling[J/OL]. Computer Science: 1-14[2026-07-11] (in Chinese).

[27] 侯蓓蓓, 关赛宗, 王亚敏. 基于Transformer的轻量级脑肿瘤图像分割算法[J/OL]. 北京邮电大学学报[2026-07-11].

HOU Beibei, GUAN Saizong, WANG Yamin. Lightweight brain tumor image segmentation algorithm based on Transformer[J/OL]. Journal of Beijing University of Posts and Telecommunications[2026-07-11] (in Chinese).

[28] 侯向宁, 黄孝斌, 徐草草, 等. 基于Mamba的轻量级多模态脑肿瘤MRI图像分割[J]. 现代电子技术, 2026, 49(9): 32-37.

HOU Xiangning, HUANG Xiaobin, XU Caocao, et al. Lightweight multimodal brain tumor MRI image segmentation based on Mamba[J]. Modern Electronics Technique, 2026, 49(9): 32-37 (in Chinese).

[29] 余唯一, 陈涛, 张军平, 等. 基于深度学习的MRI脑卒中病灶分割方法综述[J]. 智能科学与技术学报, 2023, 5(3): 293-312.

YU Weiyi, CHEN Tao, ZHANG Junping, et al. A survey of deep learning-based MRI stroke lesion segmentation methods[J]. Chinese Journal of Intelligent Science and Technology, 2023, 5(3): 293-312 (in Chinese).

[30] 邵虹, 左常升, 张萍. 结合Attention U-Net与瓶颈检测的肺部细胞图像分割方法[J]. 智能科学与技术学报, 2022, 4(4): 610-616.

SHAO Hong, ZUO Changsheng, ZHANG Ping. Lung cell image segmentation method combining Attention U-Net and bottleneck detection[J]. Chinese Journal of Intelligent Science and Technology, 2022, 4(4): 610-616 (in Chinese).

[31] 姜舒, 陈琨, 丁卫平, 等. Axial-FNet: 基于模糊卷积结合门控轴向自注意力的皮肤癌图像分割模型[J]. 智能科学与技术学报, 2025, 7(2): 221-233.

JIANG Shu, CHEN Kun, DING Weiping, et al. Axial-FNet: skin cancer image segmentation model based on fuzzy convolution combined with gated axial self-attention[J]. Chinese Journal of Intelligent Science and Technology, 2025, 7(2): 221-233 (in Chinese).
