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
    三视图VMamba块
    核心RTHD模块：将3D特征解耦为三个2D视图，使用参数共享的2D VMamba进行扫描，然后重建回3D
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
        channels_last: bool = False,  # 2D VMamba是否使用channels_last格式
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
        """
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual
        self.channels_last = channels_last

        # 三视图投影
        self.projection = TriViewProjection(mode=projection_mode)

        # 三视图重建
        self.reconstruction = TriViewReconstruction()

        # 真正启用可学习加权
        if reconstruction_mode == 'weighted':
            self.view_weights = nn.Parameter(torch.ones(3) / 3.0)
        else:
            self.view_weights = None

        # 导入SS2D（2D VMamba核心模块）
        # 尝试多种导入方式，确保健壮性
        SS2D = None

        # 方法1: 尝试绝对导入（如果项目已正确安装）
        try:
            from umamba.instructions.vmamba import SS2D
        except ImportError:
            pass

        # 方法2: 尝试相对导入
        if SS2D is None:
            try:
                import sys
                import os
                # 获取当前文件的父目录的父目录的父目录，然后进入instructions
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
            print("Tried: 1) umamba.instructions.vmamba, 2) dynamic path to instructions/")
            print("Using placeholder conv layers instead. RTHD will work but without VMamba optimization.")

        if SS2D is not None:
            # 参数共享的2D VMamba模块
            # 注意：channel_first=True表示输入格式为(B,C,H,W)，这是标准的医疗影像格式
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
                forward_type=forward_type,
                channel_first=(not channels_last),  # 如果channels_last=True，则channel_first=False
            )
        else:
            # 占位符：根据channels_last选择合适的实现
            print("Warning: SS2D not available, using placeholder implementation")
            if channels_last:
                # channels_last格式：使用Linear层
                self.vmamba_2d = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )
            else:
                # channels_first格式：使用Conv2d
                self.vmamba_2d = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=1),
                )

        # 可选：视图融合权重（可学习）
        if reconstruction_mode == 'weighted':
            self.view_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入3D特征 (B, C, D, H, W)
        Returns:
            out: 输出3D特征 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        identity = x if self.use_residual else None

        # Step 1: 三视图投影 - 将3D解耦为三个2D视图
        axial, coronal, sagittal = self.projection(x)
        # axial: (B, C, H, W)
        # coronal: (B, C, D, W)
        # sagittal: (B, C, D, H)

        # Step 2: 参数共享的2D VMamba扫描
        # 注意：三个视图共享同一个vmamba_2d模块，实现参数共享

        # 处理axial视图 (B, C, H, W) - 已经是标准2D格式
        if self.channels_last:
            axial = axial.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        axial_out = self.vmamba_2d(axial)
        if self.channels_last:
            axial_out = axial_out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 处理coronal视图 (B, C, D, W) - 将D和W视为H和W
        if self.channels_last:
            coronal = coronal.permute(0, 2, 3, 1).contiguous()  # (B, D, W, C)
        coronal_out = self.vmamba_2d(coronal)
        if self.channels_last:
            coronal_out = coronal_out.permute(0, 3, 1, 2).contiguous()  # (B, C, D, W)

        # 处理sagittal视图 (B, C, D, H) - 将D和H视为H和W
        if self.channels_last:
            sagittal = sagittal.permute(0, 2, 3, 1).contiguous()  # (B, D, H, C)
        sagittal_out = self.vmamba_2d(sagittal)
        if self.channels_last:
            sagittal_out = sagittal_out.permute(0, 3, 1, 2).contiguous()  # (B, C, D, H)

        # Step 3: 三视图重建 - 将三个2D视图重建回3D
        # 将权重传入重建模块（如果使用weighted模式）
        out = self.reconstruction(
            axial_out, coronal_out, sagittal_out,
            target_shape=(D, H, W),
            weights=self.view_weights if hasattr(self, 'view_weights') else None
        )

        # Step 4: 残差连接
        if self.use_residual and identity is not None:
            out = out + identity

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
    完整的RTHD块
    结合深度可分离卷积和三视图VMamba
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
        norm_kwargs: dict = None,  # 接收来自nnUNet的kwargs
        act_layer: Type[nn.Module] = nn.GELU,
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
        """
        super().__init__()

        # 使用传入的norm_kwargs，如果没有则使用空字典
        kw = norm_kwargs if norm_kwargs is not None else {}

        # 归一化
        self.norm1 = norm_layer(dim, **kw)

        # 三视图VMamba
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
    reconstruction = TriViewReconstruction(mode='broadcast')
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
