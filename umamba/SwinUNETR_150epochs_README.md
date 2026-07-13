# SwinUNETR Trainer Implementation for BraTS2020

## 实现概述

为BraTS2020 Dataset705创建了两个新的训练器，用于在RTX 3090 24GB上进行150轮五折训练：

### 新增文件

1. **nnUNetTrainerSwinUNETR_150epochs.py**
   - 完整的150轮训练器
   - 实现正确的2步梯度累积
   - AdamW优化器 + CosineAnnealingLR调度器
   - 兼容BraTS区域式标签

2. **nnUNetTrainerSwinUNETR_SmokeTest.py**
   - 快速验证训练器
   - 仅1轮、4次训练迭代、1次验证迭代
   - 用于验证模型初始化和内存占用

### 原始文件状态

✅ **nnUNetTrainerSwinUNETR.py** - 未修改，保持原样

## 关键技术实现

### 1. 网络配置

```python
SwinUNETR(
    in_channels=4,              # 从plans自动获取
    out_channels=3,             # BraTS: WT, TC, ET
    img_size=(128, 128, 128),   # 从plans自动获取
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    feature_size=48,            # 保持48，不改为24
    norm_name="instance",
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    normalize=True,
    use_checkpoint=True,        # 启用梯度检查点节省显存
    spatial_dims=3,
    downsample="merging",
    use_v2=False
)
```

### 2. 梯度累积实现

**核心逻辑：**
- 实际batch_size = 1
- 累积步数 = 2
- 有效batch_size = 2

**实现细节：**

```python
# 1. 仅在累积窗口开始时清零梯度
if self._grad_accum_counter == 0:
    self.optimizer.zero_grad(set_to_none=True)

# 2. 损失除以累积步数后反向传播
loss_scaled = loss / self.gradient_accumulation_steps
self.grad_scaler.scale(loss_scaled).backward()

# 3. AMP下先unscale再梯度裁剪
if self._grad_accum_counter >= self.gradient_accumulation_steps:
    self.grad_scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    self.grad_scaler.step(self.optimizer)
    self.grad_scaler.update()
    self._grad_accum_counter = 0

# 4. 日志返回未缩放的原始loss
return {'loss': loss.detach().cpu().numpy()}
```

**关键点：**
- ✅ epoch开始时重置累积计数器
- ✅ 不允许跨epoch遗留未执行的梯度
- ✅ 残余梯度会被丢弃并记录警告

### 3. 迭代次数调整

**基线配置（nnU-Net）：**
- batch_size = 2
- num_iterations_per_epoch = 250
- 每轮样本数 = 500
- 每轮optimizer步数 = 250

**SwinUNETR配置：**
- batch_size = 1
- gradient_accumulation_steps = 2
- num_iterations_per_epoch = 500
- 每轮样本数 = 500 ✅
- 每轮optimizer步数 = 250 ✅

**验证配置：**
- num_val_iterations_per_epoch = 50（保持不变）

### 4. 优化器配置

```python
optimizer = AdamW(
    self.network.parameters(),
    lr=8e-4,
    weight_decay=0.01,
    eps=1e-5
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=150,
    eta_min=1e-6
)
```

**梯度裁剪：** threshold = 12

### 5. AMP支持

```python
# CUDA环境：启用AMP
if self.device.type == 'cuda':
    self.grad_scaler = GradScaler()
    with autocast('cuda', enabled=True):
        output = self.network(data)
        loss = self.loss(output, target)

# 非CUDA环境：自动禁用AMP
else:
    self.grad_scaler = None
    # 不使用autocast
```

### 6. 验证逻辑

- 使用`torch.no_grad()`语义（由run_training提供）
- 验证阶段不调用`optimizer.zero_grad()`
- 区域式标签：sigmoid阈值0.5
- 非区域式标签：argmax
- 正确处理ignore_label
- 返回loss、tp_hard、fp_hard、fn_hard

## 训练命令

### SmokeTest（必须先运行）

**目的：** 验证模型初始化、前向/反向传播、无OOM

```bash
# Fold 0 SmokeTest
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 \
  -tr nnUNetTrainerSwinUNETR_SmokeTest

# 预期输出：
# - 1 epoch
# - 4 training iterations (2 optimizer steps)
# - 1 validation iteration
# - 成功保存checkpoint
# - 无OOM或NaN错误
```

### 完整训练（150 epochs）

```bash
# Fold 0
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 \
  -tr nnUNetTrainerSwinUNETR_150epochs

# Fold 1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 1 \
  -tr nnUNetTrainerSwinUNETR_150epochs

# Fold 2
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 2 \
  -tr nnUNetTrainerSwinUNETR_150epochs

# Fold 3
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 3 \
  -tr nnUNetTrainerSwinUNETR_150epochs

# Fold 4
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 4 \
  -tr nnUNetTrainerSwinUNETR_150epochs
```

### 验证和预测

```bash
# Fold 0 验证
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i /path/to/nnUNet_raw/Dataset705_BraTS2020/imagesTs \
  -o /path/to/predictions/fold_0 \
  -d 705 -c 3d_fullres -f 0 \
  -tr nnUNetTrainerSwinUNETR_150epochs \
  -chk checkpoint_best.pth

# 五折集成预测
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i /path/to/nnUNet_raw/Dataset705_BraTS2020/imagesTs \
  -o /path/to/predictions/ensemble \
  -d 705 -c 3d_fullres -f 0 1 2 3 4 \
  -tr nnUNetTrainerSwinUNETR_150epochs \
  -chk checkpoint_best.pth
```

## 日志信息

训练开始时会记录以下信息：

```
================================================================================
SwinUNETR 150-epoch Training Configuration
================================================================================
MONAI version: 1.3.0
PyTorch version: 2.2.1

Network Configuration:
  Patch size: [128, 128, 128]
  Spatial dims: 3
  Input channels: 4
  Output channels: 3
  Feature size: 48
  Depths: (2, 2, 2, 2)
  Num heads: (3, 6, 12, 24)
  Use checkpoint: True

Training Configuration:
  Num epochs: 150
  Plans batch size: 2
  Actual batch size: 1
  Gradient accumulation steps: 2
  Effective batch size: 2
  Num iterations per epoch: 500
  Optimizer steps per epoch: 250
  Total samples per epoch: ~500
  Num validation iterations: 50

Optimizer Configuration:
  Optimizer: AdamW
  Initial learning rate: 0.0008
  Weight decay: 0.01
  Epsilon: 1e-5
  Scheduler: CosineAnnealingLR
  Eta min: 1e-6
  Gradient clipping: 12

AMP Configuration:
  Device type: cuda
  AMP enabled: True
================================================================================
```

## 注意事项

### ⚠️ 警告检查

1. **Plans batch_size检查**
   ```
   WARNING: Expected plans batch_size=2, but got X. Effective batch size may differ from baseline!
   ```
   - 如果看到此警告，检查plans.json配置

2. **残余梯度警告**
   ```
   WARNING: Epoch ended with N residual gradient(s). These gradients are discarded.
   ```
   - 正常情况：num_iterations_per_epoch=500时不应出现
   - 如果出现：检查是否修改了迭代次数

### ✅ 验证清单

运行SmokeTest后检查：
- [ ] 模型成功初始化
- [ ] 4次训练迭代完成（无OOM）
- [ ] 1次验证迭代完成
- [ ] checkpoint_latest.pth已保存
- [ ] 日志显示正确的batch size和累积步数
- [ ] 无NaN或Inf损失
- [ ] 显存使用在24GB以内

### 🚫 禁止事项

1. ❌ 不要修改原始nnUNetTrainerSwinUNETR.py
2. ❌ 不要将feature_size从48改为24
3. ❌ 不要加载预训练权重
4. ❌ 不要修改SwinUNETR网络结构
5. ❌ 不要直接启动五折训练（先运行SmokeTest）

## 预期性能

### 内存占用（预期）
- 训练：< 20 GB
- 验证：< 15 GB

### 训练时间（预期）
- 单个epoch：~20-30分钟（取决于硬件）
- 150 epochs：~50-75小时/fold
- 五折总计：~250-375小时

### 与基线对比

| 指标 | nnU-Net | U-Mamba | SwinUNETR (预期) |
|------|---------|---------|------------------|
| Mean Dice | 85.49% | 85.27% | 待测试 |
| Mean HD95 | 4.04 mm | 4.61 mm | 待测试 |
| 参数量 | 31.20M | 42.75M | ~62M |
| 显存 | 2.04 GiB | 4.22 GiB | < 20 GiB |

## 故障排除

### OOM错误
1. 检查use_checkpoint=True是否启用
2. 减少num_iterations_per_epoch（但会改变样本数）
3. 使用更小的patch_size（需重新planning）

### 梯度累积问题
- 检查_grad_accum_counter在每个epoch开始时是否重置
- 确认optimizer.step()仅在累积完成后调用
- 验证loss是否正确缩放

### AMP问题
- 确认grad_scaler.unscale_在梯度裁剪前调用
- 检查PyTorch版本是否为2.2.1
- 验证CUDA是否可用

## 文件结构

```
nnunetv2/training/nnUNetTrainer/
├── nnUNetTrainerSwinUNETR.py                    # 原始文件（未修改）
├── nnUNetTrainerSwinUNETR_150epochs.py          # 新增：150轮训练器
└── nnUNetTrainerSwinUNETR_SmokeTest.py          # 新增：快速验证
```

## 版本兼容性

- ✅ Python 3.11
- ✅ PyTorch 2.2.1
- ✅ MONAI 1.3.0
- ✅ CUDA 12.1
- ✅ nnU-Net v2
- ✅ 单卡训练

## 参考文献

1. Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." MICCAI 2022.
2. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
3. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods, 2021.

## 更新日志

- 2026-07-13: 初始实现
  - 创建nnUNetTrainerSwinUNETR_150epochs.py
  - 创建nnUNetTrainerSwinUNETR_SmokeTest.py
  - 实现正确的2步梯度累积逻辑
  - 调整迭代次数以保持样本数和optimizer步数一致
