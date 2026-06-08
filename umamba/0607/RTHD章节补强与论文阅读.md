# RTHD 章节补强思路与近期论文阅读清单

## 一、当前问题

现有 `RTHD` 的核心贡献集中在编码器：

`3D feature -> tri-view projection -> 2D VMamba scanning -> 3D reconstruction`

这个思路本身有价值，尤其适合作为：

`高效 3D 空间建模与轻量化部署基础`

但如果作为毕业论文中的一个完整方法章节，仅修改编码器会显得单薄。  
更合理的做法是围绕 RTHD 继续补充 2-3 个低成本、强相关、能消融的模块，让方法从“编码器局部替换”升级为“编码-解码协同的轻量化 3D 分割网络”。


## 二、最推荐的补强路线

### 2.1 主线建议

建议第一章不要扩得太散，而是围绕一个统一问题：

`如何在保持轻量化的前提下，提升 3D 脑肿瘤分割中的全局建模、跨层融合和结构恢复能力？`

围绕这个问题，可以把方法扩成：

`Encoder RTHD + Decoder RTHD Refinement + Skip Fusion Gate + Boundary-aware Head`

如果还想贴近多模态趋势，再加：

`Modality-aware Stem`


## 三、改进思路 1：Decoder RTHD Refinement

### 3.1 为什么值得做

你现在只改编码器，论文会被问：

`既然 RTHD 能做三视图空间建模，为什么不在解码阶段用于结构恢复？`

脑肿瘤分割最终依赖精细的 3D 结构恢复，尤其是：

- ET 小区域
- TC 内部结构
- WT 边界
- 术后或边界模糊区域

所以把 RTHD 扩展到解码器是最自然的补强。


### 3.2 怎么做

不要全解码器都加，建议 partial：

- `Decoder Stage 4`: 加 RTHD refinement
- `Decoder Stage 3`: 加 RTHD refinement
- `Decoder Stage 2/1`: 保留普通卷积

你当前代码中已经有：

- `UNetResDecoder_RTHD`
- `decoder_rthd_mode = none / partial / full`
- `rthd_stages_decoder`

所以这一条是最容易落地的。


### 3.3 论文写法

可以命名为：

`Asymmetric Encoder-Decoder RTHD Deployment`

核心表述：

`编码器中高层使用 RTHD 捕获长程空间依赖，解码器高层使用轻量 RTHD refinement 恢复 3D 结构，从而在全局语义建模和局部边界恢复之间取得平衡。`


### 3.4 解码器恢复建议加入的策略

解码器不要只写成：

`在 decoder 中加入 RTHD`

更推荐写成：

`Stage-aware Structure Restoration Decoder`

也就是解码器恢复阶段包含三层策略：

1. `阶段选择策略`
2. `语义引导跳连融合策略`
3. `结构/边界恢复策略`


#### 策略一：Stage-aware RTHD Placement

不要所有 decoder stage 都使用 RTHD。  
更合理的是：

- 深层 decoder：语义强、分辨率低，适合 RTHD 建模全局结构
- 浅层 decoder：分辨率高、计算重，适合普通卷积恢复局部细节

推荐配置：

| Decoder stage | 分辨率 | 建议模块 | 理由 |
|---|---|---|---|
| D4 | 低分辨率 | RTHD refinement | 建模全局结构，计算成本低 |
| D3 | 中分辨率 | RTHD refinement | 恢复中尺度肿瘤区域 |
| D2 | 高分辨率 | Conv refinement | 保留局部边界，避免计算过重 |
| D1 | 最高分辨率 | Conv + head | 输出细节，不再加重模块 |

论文中可以写：

`考虑到解码器不同阶段的语义层级与空间分辨率差异，本文采用阶段感知的 RTHD 部署策略，仅在低分辨率高语义阶段引入 RTHD refinement，从而在结构恢复能力和计算效率之间取得平衡。`


#### 策略二：Semantic-guided Skip Recalibration

解码器恢复不是单独靠 decoder feature，也依赖 skip feature。  
但 skip feature 里有噪声，所以可以加一个语义引导门控：

`decoder semantic feature -> guide skip feature`

最小流程：

1. decoder feature 上采样到 skip feature 尺寸
2. 临时拼接 decoder feature 和 skip feature，仅用于生成 gate
3. 用 gate 重新标定 skip feature，得到 refined skip
4. 将 decoder feature 与 refined skip 做最终 concat 或 add
5. 将融合后的特征送入 decoder block

注意这里不是循环：

`第一次 concat` 只是 gate generator 的输入，不作为最终融合结果；  
`第二次 concat/add` 才是送入 decoder block 的正式融合。

更清晰的伪代码：

```python
x_up = upsample(lres_input)
skip = encoder_skip

gate = sigmoid(gate_conv(concat([x_up, skip], dim=1)))
skip_refined = skip + skip * gate

x = concat([x_up, skip_refined], dim=1)
x = decoder_block(x)
```

论文中可以写：

`为了避免浅层跳跃特征中的背景纹理和模态噪声直接传播到解码器，本文设计语义引导的跳跃特征重标定策略，利用高级解码语义选择性增强与肿瘤区域相关的边界和纹理响应。`


#### 策略三：High-low Frequency Structure Recovery

RTHD 在解码器中恢复的是 3D 结构，但肿瘤边界和小病灶更依赖高频细节。  
可以加入一个轻量高低频恢复策略：

`low-frequency structure + high-frequency boundary residual`

最小实现：

- `low = avg_pool3d(feature)`
- `high = feature - upsample(low)`
- `gate = sigmoid(conv(high))`
- `output = feature + gate * high`

这不一定要做复杂 FFT/DCT，平均池化近似低频就够写论文第一版。

论文中可以写：

`在解码器结构恢复阶段，本文显式引入高频残差信息以补充肿瘤边界和小区域细节，同时保留低频语义结构，从而提升边界模糊区域的恢复质量。`


#### 策略四：Boundary-supervised Decoder Output

如果想让解码器恢复更有监督，可以在最后输出端加边界辅助：

`decoder feature -> segmentation head + boundary head`

训练目标：

`L = L_seg + lambda * L_boundary`

边界标签可以从 mask 形态学操作生成：

- label erosion
- label dilation
- boundary = dilation - erosion

论文中可以写：

`为了进一步约束解码器的结构恢复过程，本文引入边界辅助监督，使模型在优化区域重叠度的同时关注肿瘤轮廓一致性。`


### 3.5 最推荐的解码器策略组合

如果只选一个：

`Stage-aware RTHD Placement`

如果选两个：

`Stage-aware RTHD Placement + Semantic-guided Skip Recalibration`

如果想让章节更完整：

`Stage-aware RTHD Placement + Semantic-guided Skip Recalibration + Boundary-supervised Decoder Output`

如果想贴近最新 Mamba/频域论文，可把下面模块作为额外消融，而不是主模型默认组件：

`High-low Frequency Structure Recovery`

最终推荐版本：

`Stage-aware RTHD Decoder + Skip Gate + Boundary-aware Head`

这个组合最稳，因为它能形成清晰叙事：

- RTHD decoder 负责恢复全局 3D 结构
- skip gate 负责选择浅层细节
- boundary head 负责约束最终轮廓
- frequency refinement 仅作为可选消融，避免和 boundary head 在主模型中形成重复的局部细节增强

一句话可以写成：

`本文在解码阶段设计阶段感知的结构恢复策略，通过低分辨率 RTHD refinement 建模全局结构、语义引导跳连融合恢复局部细节，并结合边界辅助监督提升肿瘤轮廓质量。`


## 四、改进思路 2：Skip Fusion Gate

### 4.1 为什么值得做

U-Net 的 skip connection 会把浅层纹理直接传给解码器。  
但浅层特征有两个问题：

- 有用：保留边界、纹理、局部细节
- 有害：包含噪声、伪影、模态冗余

如果你只改编码器，不处理跳跃连接，整体结构会缺少“编码-解码协同”的设计。


### 4.2 怎么做

在 skip connection 上加入轻量门控：

`encoder skip + decoder feature -> gate -> recalibrated skip`

最小实现：

- 把 decoder feature 上采样到 skip feature 尺寸
- 拼接 `skip` 和 `decoder`
- 用 `1x1x1 Conv + sigmoid` 得到 gate
- 用 gate 调制 skip feature
- 再与 decoder feature concat


### 4.3 论文写法

可以命名为：

`RTHD-guided Skip Fusion, RGSF`

核心表述：

`利用解码器高级语义对编码器浅层特征进行选择性校准，增强与肿瘤相关的边界和纹理信息，抑制冗余背景响应。`


## 五、改进思路 3：Boundary-aware Head

### 5.1 为什么值得做

BraTS 不只看 Dice，也很重视 HD95。  
HD95 对边界错误非常敏感。  
如果方法里加入边界辅助分支，论文可以自然连接到：

- 边界模糊
- 小病灶
- HD95 改善
- 结构可信性


### 5.2 怎么做

最终 decoder feature 分成两个 head：

- segmentation head：输出 WT / TC / ET 或多类 mask
- boundary head：输出边界图或边界 attention

训练时可以用：

- segmentation loss: Dice + CE
- boundary loss: BCE / Dice boundary loss
- total loss: `L = L_seg + lambda * L_boundary`

如果不想动太多训练逻辑，也可以先只做 boundary attention，不加额外标注损失。


### 5.3 论文写法

可以命名为：

`Boundary-aware Segmentation Head`

核心表述：

`通过边界辅助监督或边界注意力增强网络对肿瘤轮廓和小区域结构的感知能力，提升 HD95 和边界模糊区域的分割稳定性。`


## 六、改进思路 4：Frequency-guided RTHD

### 6.1 为什么值得做

近期 Mamba 医学分割论文很喜欢把频域信息和 Mamba 结合。原因很直接：

- Mamba 擅长长程依赖和全局建模
- 频域高频信息更关注边缘、纹理、小结构
- 频域低频信息更关注整体形态

这和脑肿瘤分割非常契合。


### 6.2 怎么做

在 RTHD 前或重建后加一个轻量频域增强：

方案 A：RTHD 前置频域门控

`3D feature -> frequency gate -> RTHD`

方案 B：RTHD 后置高频残差

`RTHD output + high-frequency residual -> refined feature`

最小实现不一定要做完整 3D DCT，可以先做：

- 3D average pooling 得到低频近似
- 原特征减低频得到高频残差
- 用 `1x1x1 Conv + sigmoid` 生成频域 gate
- 融合低频结构和高频边界


### 6.3 论文写法

可以命名为：

`Frequency-guided RTHD Enhancement`

核心表述：

`通过显式分离低频全局结构与高频边界细节，补充 RTHD 在二维状态空间扫描过程中可能损失的局部细粒度信息。`


## 七、改进思路 5：Modality-aware Stem

### 7.1 为什么值得做

BraTS 是多模态 MRI，不同模态对不同肿瘤区域的贡献不同：

- FLAIR 对 WT 重要
- T1ce 对 ET 重要
- T2 对水肿区域重要
- T1 对解剖结构有帮助

如果输入端只是四模态 concat，论文显得没有充分利用多模态特性。


### 7.2 怎么做

最小实现：

- 每个模态先过一个小 Conv stem
- global average pooling 得到模态描述
- MLP + sigmoid 得到四个模态权重
- 加权融合后送入后续 encoder


### 7.3 论文写法

可以命名为：

`Modality-aware Feature Stem`

核心表述：

`在输入端自适应建模不同 MRI 模态的重要性，为后续三视图空间建模提供更稳健的多模态表示。`


## 八、最推荐的最终组合

如果时间有限，建议做：

`Encoder RTHD + Decoder RTHD Refinement + Skip Fusion Gate`

这是最能解决“只改编码器太薄”的组合。

如果还想进一步丰满章节：

`+ Boundary-aware Head`

如果想贴近 2025-2026 多模态趋势：

`+ Modality-aware Stem`

如果想贴近最新 Mamba 论文，可以把下面方向作为可选消融或后续章节：

`+ Frequency-guided RTHD`

最终优先级：

1. `Decoder RTHD Refinement`
2. `Skip Fusion Gate`
3. `Boundary-aware Head`
4. `Modality-aware Stem`
5. `Frequency-guided RTHD`


## 九、消融实验设计

建议消融表：

| 编号 | Encoder RTHD | Decoder RTHD | Skip Gate | Boundary Head | Frequency Gate | 目的 |
|---|---|---|---|---|---|---|
| A0 | 否 | 否 | 否 | 否 | 否 | U-Mamba baseline |
| A1 | 是 | 否 | 否 | 否 | 否 | 验证基础 RTHD |
| A2 | 是 | 是 | 否 | 否 | 否 | 验证解码器结构恢复 |
| A3 | 是 | 是 | 是 | 否 | 否 | 验证跳连门控融合 |
| A4 | 是 | 是 | 是 | 是 | 否 | 验证边界感知输出 |
| A5 | 是 | 是 | 是 | 是 | 是 | 验证频域增强 |

主指标：

- Dice: WT / TC / ET
- HD95: WT / TC / ET
- Params
- FLOPs
- GPU memory
- inference time

补充分析：

- ET 小区域可视化
- 边界模糊病例
- 推理显存和速度
- 缺失模态下的鲁棒性测试


## 十、建议重点阅读的最新论文

### 10.1 和 RTHD 最直接相关

1. [CDA-Mamba: cross-directional attention mamba for enhanced 3D medical image segmentation, Scientific Reports 2025](https://www.nature.com/articles/s41598-025-06462-3)

建议重点看：

- tri-directional Mamba
- multi-frequency gated convolution
- 为什么只用 Mamba 不够，还需要频域和注意力补充

对你的启发：

- RTHD 可以加入 `Frequency-guided RTHD`
- 可以强调三视图/三方向对 3D 体数据的重要性


2. [BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency based Feature Enhancement for Brain Tumor Segmentation, MICCAI 2025](https://papers.miccai.org/miccai-2025/0117-Paper0487.html)

建议重点看：

- dual-band frequency enhancement
- Mamba + U-Net 在 BraTS 上怎么讲创新
- 辅助分类 loss 怎么增强分割

对你的启发：

- 可以做频域增强或辅助 head
- 可以学习它如何把 Mamba 和脑肿瘤分割痛点连接起来


3. [CFG-MambaNet: Contextual and Frequency-Guided Mamba Network for medical image segmentation, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02393-z)

建议重点看：

- variable-scale state space block
- frequency-guided representation
- adaptive context aggregation
- composite loss + deep supervision

对你的启发：

- 你的 RTHD 可以补 `multi-scale context` 或 `frequency-guided boundary`
- 章节里可以增加边界、频域和 deep supervision 叙事


4. [A comprehensive analysis of Mamba for 3D volumetric medical image segmentation, Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320325013640)

建议重点看：

- Mamba 在 3D 体数据分割中的优缺点
- 哪些扫描策略和部署位置有效
- BraTS 上的比较和失败分析

对你的启发：

- 帮你判断 RTHD 应该放 encoder、decoder 还是 bottleneck
- 帮你写“为什么不是直接 flatten 3D”


### 10.2 多模态和缺失模态方向

5. [IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities, MICCAI 2025](https://papers.miccai.org/miccai-2025/0437-Paper0747.html)

建议重点看：

- incomplete modalities
- Mamba fusion block
- 如何建模缺失模态条件下的多模态融合

对你的启发：

- RTHD 后续可以扩展成 `modality-view joint modeling`


6. [DC-Seg: Disentangled Contrastive Learning for Brain Tumor Segmentation with Missing Modalities, MICCAI 2025](https://papers.miccai.org/miccai-2025/0213-Paper0653.html)

建议重点看：

- anatomical representation
- modality-specific representation
- contrastive learning 如何提升缺失模态鲁棒性

对你的启发：

- 可以把第二章做成 `RTHD + modality-invariant anatomical representation`


7. [Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for Incomplete Multimodal Brain Tumor Segmentation, MICCAI 2024](https://papers.miccai.org/miccai-2024/520-Paper0067.html)

建议重点看：

- 把缺失模态看成 masking
- cross-modal feature reconstruction
- 如何训练完整模态和缺失模态混合样本

对你的启发：

- 可以做一个低成本 missing modality 实验，不一定立刻改网络


8. [Semantic-guided Masked Mutual Learning for Multi-modal Brain Tumor Segmentation with Arbitrary Missing Modalities, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32545)

建议重点看：

- masked mutual learning
- SAM semantic prior
- arbitrary missing modalities

对你的启发：

- 可以借鉴“语义先验/互学习”写第二章或展望


### 10.3 可信性、边界和不确定性方向

9. [UD-Mamba: A pixel-level uncertainty-driven Mamba model for medical image segmentation, arXiv 2025](https://arxiv.org/abs/2502.02024)

建议重点看：

- uncertainty-guided refinement
- pixel-level uncertainty
- Mamba 和可信分割怎么结合

对你的启发：

- 第三章可以做 `uncertainty-guided RTHD refinement`
- 第一章也可以加 boundary/uncertainty 分支作为轻量扩展


### 10.4 趋势综述

10. [Efficient Medical Image Segmentation in Multisensor Imaging: A Survey in the Era of Mamba and Foundation Models, Sensors 2026](https://www.mdpi.com/1424-8220/26/8/2558)

建议重点看：

- Mamba and State Space Models
- Efficient Adaptation of Foundation Models
- Advanced Lightweight Architectures
- Data-Efficient Strategies

对你的启发：

- 帮你把 `RTHD` 从“轻量化模块”提升到“效率驱动医学分割范式”的叙事里


## 十一、阅读顺序建议

如果你现在要最快找思路，建议按这个顺序读：

1. `BraTS-UMamba`
2. `CDA-Mamba`
3. `CFG-MambaNet`
4. `IM-Fuse`
5. `DC-Seg`
6. `Missing as Masking`
7. `UD-Mamba`
8. `Efficient Medical Image Segmentation Survey 2026`

先看 1-3，是为了补强你的第一章。  
再看 4-6，是为了规划第二章缺失模态。  
最后看 7-8，是为了找第三章可信分割或轻量化部署叙事。


## 十二、最后建议

当前最现实的补强路线是：

`保留现有 Encoder RTHD -> 启用/完善 Decoder RTHD partial refinement -> 加 Skip Fusion Gate -> 加 Boundary-aware Head`

这条线最稳，因为：

- 不推翻现有代码
- 能自然扩展总体架构图
- 每个模块都有明确问题
- 每个模块都能做消融
- 能和近期 Mamba 医学分割论文对齐

一句话总结：

`不要把 RTHD 只写成编码器轻量化模块，要把它扩成编码器建模、解码器恢复、跳连融合和边界输出协同的轻量化 3D 分割框架。`
