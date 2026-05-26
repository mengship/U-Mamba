"""
RTHD (Recursive Tri-view Hierarchical Decomposition) Modules
三视图递归层次分解模块

核心思想：
将3D体积张量解耦投影为三个正交的2D切片流（Axial轴状位、Coronal冠状位、Sagittal矢状位），
利用参数共享的2D VMamba进行轻量化二维扫描，将序列长度从O(D×H×W)降至O(H×W)。

作者：研究生脑肿瘤分割项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
import math


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将2D特征图分割为不重叠的局部窗口（LoMamba思想）

    Args:
        x: 输入特征 (B, C, H, W)
        window_size: 窗口大小
    Returns:
        windows: 窗口特征 (B * num_windows, C, window_size, window_size)
        (H_pad, W_pad): padding后的尺寸
    """
    B, C, H, W = x.shape

    # 计算padding，确保H和W能被window_size整除
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    H_pad, W_pad = H + pad_h, W + pad_w

    # 分割为窗口: (B, C, H_pad, W_pad) -> (B, C, nH, window_size, nW, window_size)
    nH = H_pad // window_size
    nW = W_pad // window_size

    windows = x.view(B, C, nH, window_size, nW, window_size)
    windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nH, nW, C, window_size, window_size)
    windows = windows.view(B * nH * nW, C, window_size, window_size)

    return windows, (H_pad, W_pad)


def window_reverse(windows: torch.Tensor, window_size: int, H_pad: int, W_pad: int,
                   H_orig: int, W_orig: int) -> torch.Tensor:
    """
    将窗口特征合并回完整的2D特征图

    Args:
        windows: 窗口特征 (B * num_windows, C, window_size, window_size)
        window_size: 窗口大小
        H_pad, W_pad: padding后的尺寸
        H_orig, W_orig: 原始尺寸
    Returns:
        x: 合并后的特征 (B, C, H_orig, W_orig)
    """
    nH = H_pad // window_size
    nW = W_pad // window_size
    B = windows.shape[0] // (nH * nW)
    C = windows.shape[1]

    # (B * nH * nW, C, window_size, window_size) -> (B, nH, nW, C, window_size, window_size)
    x = windows.view(B, nH, nW, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, nH, window_size, nW, window_size)
    x = x.view(B, C, H_pad, W_pad)

    # 移除padding
    if H_pad > H_orig or W_pad > W_orig:
        x = x[:, :, :H_orig, :W_orig]

    return x


class TriViewProjection(nn.Module):
    """
    三视图投影模块
    将3D特征张量 (B, C, D, H, W) 解耦为三个2D视图：
    - Axial (轴状位): (B, C, H, W) - 沿D维度切片
    - Coronal (冠状位): (B, C, D, W) - 沿H维度切片
    - Sagittal (矢状位): (B, C, D, H) - 沿W维度切片
    """
    def __init__(self, mode='mean'):
        """
        Args:
            mode: 投影模式，'mean'表示平均池化，'max'表示最大池化，'slice'表示取中间切片
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入3D特征 (B, C, D, H, W)
        Returns:
            axial: 轴状位视图 (B, C, H, W)
            coronal: 冠状位视图 (B, C, D, W)
            sagittal: 矢状位视图 (B, C, D, H)
        """
        B, C, D, H, W = x.shape

        if self.mode == 'mean':
            # 平均池化投影
            axial = x.mean(dim=2)  # (B, C, H, W) - 沿D维度平均
            coronal = x.mean(dim=3)  # (B, C, D, W) - 沿H维度平均
            sagittal = x.mean(dim=4)  # (B, C, D, H) - 沿W维度平均
        elif self.mode == 'max':
            # 最大池化投影
            axial = x.max(dim=2)[0]
            coronal = x.max(dim=3)[0]
            sagittal = x.max(dim=4)[0]
        elif self.mode == 'slice':
            # 取中间切片
            axial = x[:, :, D//2, :, :]
            coronal = x[:, :, :, H//2, :]
            sagittal = x[:, :, :, :, W//2]
        else:
            raise ValueError(f"Unsupported projection mode: {self.mode}")

        return axial, coronal, sagittal


class TriViewReconstruction(nn.Module):
    """
    三视图重建模块
    将三个2D视图特征重建回3D体积张量
    """
    def __init__(self):
        """
        不再使用mode参数，改为通过forward的weights参数控制
        """
        super().__init__()

    def forward(
        self,
        axial: torch.Tensor,  # (B, C, H, W)
        coronal: torch.Tensor,  # (B, C, D, W)
        sagittal: torch.Tensor,  # (B, C, D, H)
        target_shape: Tuple[int, int, int],  # (D, H, W)
        weights: Optional[torch.Tensor] = None  # 可选的可学习权重 (3,)
    ) -> torch.Tensor:
        """
        Args:
            axial: 轴状位特征 (B, C, H, W)
            coronal: 冠状位特征 (B, C, D, W)
            sagittal: 矢状位特征 (B, C, D, H)
            target_shape: 目标3D形状 (D, H, W)
            weights: 可选的融合权重 (3,)，如果提供则使用加权融合
        Returns:
            x: 重建的3D特征 (B, C, D, H, W)
        """
        B, C = axial.shape[:2]
        D, H, W = target_shape

        # 广播重建：每个视图沿其缺失维度广播
        axial_3d = axial.unsqueeze(2).expand(B, C, D, H, W)  # (B, C, 1, H, W) -> (B, C, D, H, W)
        coronal_3d = coronal.unsqueeze(3).expand(B, C, D, H, W)  # (B, C, D, 1, W) -> (B, C, D, H, W)
        sagittal_3d = sagittal.unsqueeze(4).expand(B, C, D, H, W)  # (B, C, D, H, 1) -> (B, C, D, H, W)

        if weights is not None:
            # 使用Softmax归一化权重，确保融合物理意义正确
            attn = F.softmax(weights, dim=0)
            x = axial_3d * attn[0] + coronal_3d * attn[1] + sagittal_3d * attn[2]
        else:
            # 平均融合三个视图
            x = (axial_3d + coronal_3d + sagittal_3d) / 3.0

        return x


class TriViewVMambaBlock(nn.Module):
    """
    三视图VMamba块（支持消融实验）
    核心RTHD模块：将3D特征解耦为三个2D视图，使用参数共享的2D VMamba进行扫描，然后重建回3D

    消融实验控制参数：
    - view_mode: 'tri' (三视图) 或 'single' (仅轴状位)
    - share_weights: True (参数共享) 或 False (独立参数)
    - scan_mode: 'omni' (全向扫描) 或 'standard' (标准扫描)
    - use_local_window: True (局部滑窗) 或 False (全局平铺)
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        initialize: str = "v0",
        forward_type: str = "v2",
        projection_mode: str = 'mean',
        reconstruction_mode: str = 'broadcast',
        use_residual: bool = True,
        channels_last: bool = False,
        # 消融实验控制参数
        view_mode: str = 'tri',  # 'tri' or 'single'
        share_weights: bool = True,  # True: 参数共享, False: 独立参数
        scan_mode: str = 'omni',  # 'omni' or 'standard'
        use_local_window: bool = False,  # True: 局部滑窗, False: 全局平铺
        window_size: int = 8,  # 局部窗口大小
    ):
        """
        Args:
            dim: 特征维度
            d_state: SSM状态维度
            ssm_ratio: SSM扩展比例
            dt_rank: delta时间步的秩
            d_conv: 深度卷积核大小
            conv_bias: 是否使用卷积偏置
            dropout: Dropout比例
            bias: 是否使用偏置
            dt_min, dt_max, dt_init, dt_scale, dt_init_floor: delta初始化参数
            initialize: 初始化方式
            forward_type: 前向传播类型
            projection_mode: 投影模式 ('mean', 'max', 'slice')
            reconstruction_mode: 重建模式 ('broadcast', 'weighted')
            use_residual: 是否使用残差连接
            channels_last: 2D VMamba是否使用channels_last (B,H,W,C)格式
            view_mode: 'tri' (三视图) 或 'single' (仅轴状位) - 消融实验 #2
            share_weights: True (参数共享) 或 False (独立参数) - 消融实验 #3
            scan_mode: 'omni' (全向扫描) 或 'standard' (标准扫描) - 消融实验 #4
            use_local_window: True (局部滑窗) 或 False (全局平铺) - 消融实验 #5
            window_size: 局部窗口大小
        """
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual
        self.channels_last = channels_last

        # 消融实验控制参数
        self.view_mode = view_mode
        self.share_weights = share_weights
        self.scan_mode = scan_mode
        self.use_local_window = use_local_window
        self.window_size = window_size

        # 三视图投影（仅在tri模式下使用）
        if view_mode == 'tri':
            self.projection = TriViewProjection(mode=projection_mode)
        else:
            self.projection = None

        # 三视图重建（仅在tri模式下使用）
        if view_mode == 'tri':
            self.reconstruction = TriViewReconstruction()
        else:
            self.reconstruction = None

        # 可学习加权（仅在weighted模式下使用）
        if reconstruction_mode == 'weighted' and view_mode == 'tri':
            self.view_weights = nn.Parameter(torch.ones(3) / 3.0)
        else:
            self.view_weights = None

        # 导入SS2D（2D VMamba核心模块）
        SS2D = None

        # 方法1: 尝试绝对导入
        try:
            from umamba.instructions.vmamba import SS2D
        except ImportError:
            pass

        # 方法2: 尝试相对导入
        if SS2D is None:
            try:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                instructions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'instructions')
                if os.path.exists(instructions_dir) and instructions_dir not in sys.path:
                    sys.path.insert(0, instructions_dir)
                from vmamba import SS2D
            except Exception:
                pass

        # 方法3: 如果都失败，打印警告并使用占位符
        if SS2D is None:
            print("Warning: Cannot import SS2D from vmamba module.")
            print("Using placeholder conv layers instead.")

        # 根据 share_weights 决定实例化方式
        if share_weights:
            # 消融实验 #4, #5, #6: 参数共享，只实例化一个模块
            if SS2D is not None:
                self.vmamba_2d = SS2D(
                    d_model=dim,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=dt_rank,
                    d_conv=d_conv,
                    conv_bias=conv_bias,
                    dropout=dropout,
                    bias=bias,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_init=dt_init,
                    dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor,
                    initialize=initialize,
                    forward_type=forward_type if scan_mode == 'omni' else 'v0',  # 消融实验 #4: standard模式用v0
                    channel_first=(not channels_last),
                )
            else:
                # 占位符
                if channels_last:
                    self.vmamba_2d = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim),
                        nn.GELU(),
                    )
                else:
                    self.vmamba_2d = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                        nn.GELU(),
                        nn.Conv2d(dim, dim, kernel_size=1),
                    )
            self.vmamba_axial = None
            self.vmamba_coronal = None
            self.vmamba_sagittal = None
        else:
            # 消融实验 #3: 独立参数，实例化三个独立模块
            if SS2D is not None:
                self.vmamba_axial = SS2D(
                    d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
                    d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
                    dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor, initialize=initialize,
                    forward_type=forward_type if scan_mode == 'omni' else 'v0',
                    channel_first=(not channels_last),
                )
                self.vmamba_coronal = SS2D(
                    d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
                    d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
                    dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor, initialize=initialize,
                    forward_type=forward_type if scan_mode == 'omni' else 'v0',
                    channel_first=(not channels_last),
                )
                self.vmamba_sagittal = SS2D(
                    d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
                    d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
                    dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor, initialize=initialize,
                    forward_type=forward_type if scan_mode == 'omni' else 'v0',
                    channel_first=(not channels_last),
                )
            else:
                # 占位符
                if channels_last:
                    self.vmamba_axial = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
                    self.vmamba_coronal = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
                    self.vmamba_sagittal = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
                else:
                    self.vmamba_axial = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
                    self.vmamba_coronal = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
                    self.vmamba_sagittal = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
            self.vmamba_2d = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入3D特征 (B, C, D, H, W)
        Returns:
            out: 输出3D特征 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        identity = x if self.use_residual else None

        # 消融实验 #2: 单视图降级版
        if self.view_mode == 'single':
            # 只处理轴状位视图 (B, C, H, W)
            axial = x.mean(dim=2)  # 沿D维度平均投影
            axial_out = self._process_view(axial, 'axial')
            # 重建回3D: (B, C, H, W) -> (B, C, D, H, W)
            out = axial_out.unsqueeze(2).expand(B, C, D, H, W)

        # 消融实验 #3, #4, #5, #6: 三视图版本
        else:  # view_mode == 'tri'
            # Step 1: 三视图投影
            axial, coronal, sagittal = self.projection(x)
            # axial: (B, C, H, W)
            # coronal: (B, C, D, W)
            # sagittal: (B, C, D, H)

            # Step 2: 参数共享或独立参数的2D VMamba扫描
            axial_out = self._process_view(axial, 'axial')
            coronal_out = self._process_view(coronal, 'coronal')
            sagittal_out = self._process_view(sagittal, 'sagittal')

            # Step 3: 三视图重建
            out = self.reconstruction(
                axial_out, coronal_out, sagittal_out,
                target_shape=(D, H, W),
                weights=self.view_weights
            )

        # Step 4: 残差连接
        if self.use_residual and identity is not None:
            out = out + identity

        return out

    def _process_view(self, view: torch.Tensor, view_name: str) -> torch.Tensor:
        """
        处理单个视图，支持局部滑窗和全局平铺两种模式

        Args:
            view: 输入视图 (B, C, H, W)
            view_name: 视图名称 ('axial', 'coronal', 'sagittal')
        Returns:
            out: 处理后的视图 (B, C, H, W)
        """
        B, C, H, W = view.shape

        # 消融实验 #5: 全局平铺版 (use_local_window=False)
        if not self.use_local_window:
            # 直接全局扫描，不分窗
            if self.channels_last:
                view = view.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            # 根据 share_weights 选择模块
            if self.share_weights:
                out = self.vmamba_2d(view)
            else:
                if view_name == 'axial':
                    out = self.vmamba_axial(view)
                elif view_name == 'coronal':
                    out = self.vmamba_coronal(view)
                else:  # sagittal
                    out = self.vmamba_sagittal(view)

            if self.channels_last:
                out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 消融实验 #3, #4, #6: 局部滑窗版 (use_local_window=True)
        else:
            # Step 1: 窗口分割
            windows, (H_pad, W_pad) = window_partition(view, self.window_size)
            # windows: (B * num_windows, C, window_size, window_size)

            # Step 2: 对每个窗口进行VMamba扫描
            if self.channels_last:
                windows = windows.permute(0, 2, 3, 1).contiguous()

            # 根据 share_weights 选择模块
            if self.share_weights:
                windows_out = self.vmamba_2d(windows)
            else:
                if view_name == 'axial':
                    windows_out = self.vmamba_axial(windows)
                elif view_name == 'coronal':
                    windows_out = self.vmamba_coronal(windows)
                else:  # sagittal
                    windows_out = self.vmamba_sagittal(windows)

            if self.channels_last:
                windows_out = windows_out.permute(0, 3, 1, 2).contiguous()

            # Step 3: 窗口合并
            out = window_reverse(windows_out, self.window_size, H_pad, W_pad, H, W)

        return out


class DepthwiseSeparableConv3d(nn.Module):
    """
    3D深度可分离卷积 (DS-Conv)
    将标准3D卷积分解为深度卷积和逐点卷积，大幅减少参数量和计算量
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # 深度卷积：每个通道独立卷积
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键：groups=in_channels实现深度卷积
            bias=bias,
        )
        # 逐点卷积：1x1x1卷积混合通道
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class RTHDBlock(nn.Module):
    """
    完整的RTHD块（支持消融实验）
    结合深度可分离卷积和三视图VMamba

    消融实验控制参数：
    - view_mode: 'tri' (三视图) 或 'single' (仅轴状位)
    - share_weights: True (参数共享) 或 False (独立参数)
    - scan_mode: 'omni' (全向扫描) 或 'standard' (标准扫描)
    - use_local_window: True (局部滑窗) 或 False (全局平铺)
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        projection_mode: str = 'mean',
        reconstruction_mode: str = 'broadcast',
        use_ds_conv: bool = True,
        norm_layer: Type[nn.Module] = nn.InstanceNorm3d,
        norm_kwargs: dict = None,
        act_layer: Type[nn.Module] = nn.GELU,
        # 消融实验控制参数
        view_mode: str = 'tri',
        share_weights: bool = True,
        scan_mode: str = 'omni',
        use_local_window: bool = False,
        window_size: int = 8,
    ):
        """
        Args:
            dim: 特征维度
            d_state: SSM状态维度
            ssm_ratio: SSM扩展比例
            dt_rank: delta时间步的秩
            d_conv: 深度卷积核大小
            dropout: Dropout比例
            projection_mode: 投影模式
            reconstruction_mode: 重建模式
            use_ds_conv: 是否使用深度可分离卷积
            norm_layer: 归一化层
            norm_kwargs: 归一化层参数（如{'eps': 1e-5, 'affine': True}）
            act_layer: 激活函数
            view_mode: 'tri' (三视图) 或 'single' (仅轴状位) - 消融实验 #2
            share_weights: True (参数共享) 或 False (独立参数) - 消融实验 #3
            scan_mode: 'omni' (全向扫描) 或 'standard' (标准扫描) - 消融实验 #4
            use_local_window: True (局部滑窗) 或 False (全局平铺) - 消融实验 #5
            window_size: 局部窗口大小
        """
        super().__init__()

        # 使用传入的norm_kwargs，如果没有则使用空字典
        kw = norm_kwargs if norm_kwargs is not None else {}

        # 归一化
        self.norm1 = norm_layer(dim, **kw)

        # 三视图VMamba（带消融控制参数）
        self.tri_view_vmamba = TriViewVMambaBlock(
            dim=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=dropout,
            projection_mode=projection_mode,
            reconstruction_mode=reconstruction_mode,
            use_residual=False,  # 残差在外层处理
            # 消融实验控制参数
            view_mode=view_mode,
            share_weights=share_weights,
            scan_mode=scan_mode,
            use_local_window=use_local_window,
            window_size=window_size,
        )

        # 可选的深度可分离卷积
        if use_ds_conv:
            self.norm2 = norm_layer(dim, **kw)
            self.ds_conv = DepthwiseSeparableConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=1,
            )
            self.act = act_layer()
        else:
            self.ds_conv = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入3D特征 (B, C, D, H, W)
        Returns:
            out: 输出3D特征 (B, C, D, H, W)
        """
        # 三视图VMamba分支
        identity = x
        x = self.norm1(x)
        x = self.tri_view_vmamba(x)
        x = self.dropout(x)
        x = x + identity

        # 深度可分离卷积分支（可选）
        if self.ds_conv is not None:
            identity = x
            x = self.norm2(x)
            x = self.ds_conv(x)
            x = self.act(x)
            x = self.dropout(x)
            x = x + identity

        return x


if __name__ == "__main__":
    # 测试代码
    print("Testing RTHD Modules...")

    # 测试三视图投影
    print("\n1. Testing TriViewProjection...")
    projection = TriViewProjection(mode='mean')
    x = torch.randn(2, 64, 8, 16, 16)  # (B, C, D, H, W)
    axial, coronal, sagittal = projection(x)
    print(f"Input shape: {x.shape}")
    print(f"Axial shape: {axial.shape}")  # (2, 64, 16, 16)
    print(f"Coronal shape: {coronal.shape}")  # (2, 64, 8, 16)
    print(f"Sagittal shape: {sagittal.shape}")  # (2, 64, 8, 16)

    # 测试三视图重建
    print("\n2. Testing TriViewReconstruction...")
    reconstruction = TriViewReconstruction()  # 修复：不再接受mode参数
    x_recon = reconstruction(axial, coronal, sagittal, target_shape=(8, 16, 16))
    print(f"Reconstructed shape: {x_recon.shape}")  # (2, 64, 8, 16, 16)

    # 测试TriViewVMambaBlock
    print("\n3. Testing TriViewVMambaBlock...")
    block = TriViewVMambaBlock(dim=64, d_state=16)
    x = torch.randn(2, 64, 8, 16, 16)
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 测试深度可分离卷积
    print("\n4. Testing DepthwiseSeparableConv3d...")
    ds_conv = DepthwiseSeparableConv3d(64, 128, kernel_size=3, padding=1)
    x = torch.randn(2, 64, 8, 16, 16)
    out = ds_conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 测试完整的RTHDBlock
    print("\n5. Testing RTHDBlock...")
    rthd_block = RTHDBlock(dim=64, d_state=16, use_ds_conv=True)
    x = torch.randn(2, 64, 8, 16, 16)
    out = rthd_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    print("\nAll tests passed!")
