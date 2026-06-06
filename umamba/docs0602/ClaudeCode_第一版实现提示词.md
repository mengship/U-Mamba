# Claude Code 第一版实现提示词

你现在在一个本地代码仓库中工作，需要帮助我实现第一章方法的第一版增强。  
请直接修改代码并自行完成必要的检查。  
不要只给建议，要实际改代码。


## 一、任务目标

当前项目已经有一个 `RTHD (Recursive Tri-view Hierarchical Decomposition)` 模块，核心文件是：

- `umamba/nnunetv2/nets/rthd_modules.py`
- `umamba/nnunetv2/nets/UMambaEnc_RTHD.py`

当前问题是：

- 三视图主要是“各自扫描，最后再融合”
- 重建主要是简单 broadcast + average / global weighted
- 第一章的方法还不够厚，想先做一个第一版增强

这次只实现第一版，不要把范围扩太大。  
第一版只做两件事：

1. `gated reconstruction`
2. `minimal cross-view interaction`

不要顺手加入第二章或第三章的内容，例如：

- 缺失模态
- foundation model
- 视觉语言
- uncertainty
- test-time adaptation


## 二、实现范围

### 1. 必做项 A：实现 gated reconstruction

目标：

- 让 `TriViewReconstruction` 不再只支持简单平均或全局 `(3,)` 权重
- 支持位置相关的三视图门控融合

重点修改文件：

- `umamba/nnunetv2/nets/rthd_modules.py`

重点修改类：

- `TriViewReconstruction`
- `TriViewVMambaBlock`
- `RTHDBlock`

具体要求：

- 扩展 `reconstruction_mode`
- 保留现有兼容模式：
  - `"broadcast"`
  - `"weighted"`
- 新增模式：
  - `"gated"`

`gated` 模式建议实现方式：

1. 先构造：
   - `axial_3d`
   - `coronal_3d`
   - `sagittal_3d`
2. 将三者按通道拼接成 `(B, 3C, D, H, W)`
3. 用轻量 `1x1x1 Conv3d` 生成 `(B, 3, D, H, W)` 的 gate logits
4. 对 gate 在视图维做 `softmax`
5. 按 gate 融合三个 3D 视图

实现注意：

- `TriViewReconstruction` 目前没有 `dim`，如果要用 `Conv3d`，请合理重构它的 `__init__`
- 改动时要尽量保持现有调用链清晰，不要引入很乱的兼容逻辑
- 保持现有接口尽可能兼容，但允许在内部增加必要参数


### 2. 必做项 B：实现最小版 cross-view interaction

目标：

- 不再让三视图完全独立
- 但第一版只做“最小实现”，不要引入过重结构

重点修改文件：

- `umamba/nnunetv2/nets/rthd_modules.py`

重点修改类：

- `TriViewVMambaBlock`

具体要求：

- 新增参数，例如：
  - `cross_view_interaction: bool = False`
  - `interaction_mode: str = "post"`
  - `interaction_type: str = "gate"`

第一版只需要支持：

- `cross_view_interaction=False`
- `cross_view_interaction=True` 且 `interaction_mode="post"`

不要一开始实现太多模式。

推荐做法：

- 三个视图各自 `_process_view` 后
- 先临时重建成一个 `fused_3d`
- 再通过轻量 3D 交互模块生成对三个视图的引导
- 再修正 axial/coronal/sagittal 分支

重点是：

- 解决视图尺寸不一致的问题
- 不要粗暴直接拼接 `(B, C, H, W)` / `(B, C, D, W)` / `(B, C, D, H)` 这三种不同 shape

建议最低可行实现：

- 在 post-interaction 阶段使用临时 3D 融合特征
- 用轻量门控方式修正各视图输出
- 保持额外参数量较小


## 三、当前代码结构背景

请先理解以下代码结构再改：

- `TriViewProjection`
  负责 3D -> axial/coronal/sagittal

- `TriViewVMambaBlock`
  负责三视图扫描和重建

- `TriViewReconstruction`
  负责三视图 -> 3D

- `RTHDBlock`
  外层封装，负责 norm + tri-view vmamba + ds-conv

- `ResidualMambaEncoder_RTHD`
  在编码器 stage 中挂载 `RTHDBlock`

- `UNetResDecoder_RTHD`
  在解码器中挂载 `RTHDBlock`

本次第一版**先不要**改网络级部署策略，例如：

- 不做 partial decoder RTHD
- 不做 stage-wise window sizes
- 不做 encoder/decoder 非对称策略

这些属于下一阶段。


## 四、实现要求

### 1. 兼容性要求

- 不要破坏当前已有的 `view_mode/share_weights/scan_mode/use_local_window`
- 现有 `broadcast` 和 `weighted` 行为要保持可用
- 现有 trainer 若不传新参数，默认行为应尽量与当前一致

### 2. 代码风格要求

- 保持现有类名和主接口不变
- 只在必要位置增加新参数
- 对新增逻辑写清晰注释
- 复杂逻辑前可以加少量说明性注释，但不要过度注释

### 3. 稳定性要求

- 严格检查 shape
- 不允许出现 tensor 维度不匹配
- 处理好 `view_mode='single'` 的兼容
- 处理好 `reconstruction_mode='broadcast'/'weighted'/'gated'`
- 如果使用 `Conv3d` 模块，确保参数和输入通道一致


## 五、建议新增参数

建议在 `rthd_config` 中支持以下新参数：

```python
{
    "reconstruction_mode": "gated",
    "cross_view_interaction": True,
    "interaction_mode": "post",
    "interaction_type": "gate",
}
```

但注意：

- 不要要求用户必须传这些参数
- 要有安全默认值


## 六、建议测试与自检

实现完成后，请至少自行做以下检查：

### 1. 静态检查

- 确认新增参数在调用链里都能正确透传
- 确认不会因为 `TriViewReconstruction` 改造导致旧调用报错

### 2. 运行检查

至少运行一个最小前向测试，例如：

- `TriViewReconstruction` 的三种模式都能 forward
- `TriViewVMambaBlock(dim=64)` 能处理 `(2, 64, 8, 16, 16)`
- `RTHDBlock(dim=64)` 能正常 forward

如果项目里已有自测脚本，可优先复用；没有的话可以补一个最小测试逻辑。

### 3. 验收要求

最终代码应满足：

- 原有模式可运行
- 新增 `gated reconstruction` 可运行
- 新增最小版 `cross-view interaction` 可运行
- 不出现明显的接口断裂


## 七、不要做的事情

- 不要改第二章、第三章相关内容
- 不要做大规模重构整个仓库
- 不要顺手引入过多新实验开关
- 不要把 `UMambaEnc_RTHD.py` 改成新的复杂配置系统
- 不要删掉原有功能


## 八、输出方式

完成后请给出：

1. 你修改了哪些文件
2. 新增了哪些参数
3. 前向流程有什么变化
4. 你做了哪些检查
5. 还剩下哪些下一阶段工作没有做


## 九、额外说明

这个任务的目标不是“一次把整章做完”，而是完成第一版最关键的两个增强点，让第一章从：

`单纯三视图扫描模块`

提升为：

`具备多轴交互和结构感知重建能力的 RTHD 第一版`

请以“最小可行实现、稳定优先、接口清晰”为原则完成。
