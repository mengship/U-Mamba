"""
UMambaEnc with RTHD (Recursive Tri-view Hierarchical Decomposition)
基于UMambaEnc_3d.py，集成三视图递归（RTHD）机制

核心改进：
1. 使用TriViewVMambaBlock替代部分MambaLayer，实现轻量化三视图扫描
2. 将3D特征解耦为Axial/Coronal/Sagittal三个2D视图
3. 参数共享的2D VMamba扫描，序列长度从O(D×H×W)降至O(H×W)
4. 大幅降低显存占用，适合消费级GPU（RTX 3090）

作者：研究生脑肿瘤分割项目
"""

import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# 导入RTHD模块
from .rthd_modules import RTHDBlock, TriViewVMambaBlock


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    """原始的MambaLayer，用于对比和兼容"""
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

# 第三一步编码器
class ResidualMambaEncoder_RTHD(nn.Module):
    """
    集成RTHD的编码器
    在浅层使用RTHD块，在深层使用原始MambaLayer
    """
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 use_rthd: bool = True,  # 是否使用RTHD
                 rthd_stages: List[int] = None,  # 哪些stage使用RTHD，默认前3个stage
                 rthd_config: dict = None,  # 消融实验配置
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # 决定哪些stage使用RTHD
        if rthd_stages is None:
            # 默认：前3个stage使用RTHD（浅层特征图较大，RTHD效果更好）
            rthd_stages = list(range(min(3, n_stages)))

        self.use_rthd = use_rthd
        self.rthd_stages = rthd_stages

        # 计算每个stage的特征图大小，决定是否使用channel_token
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")
        print(f"RTHD enabled for stages: {rthd_stages}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        # Stem
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                use_1x1conv=True,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            ),
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        # 构建stages和mamba_layers
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            # 决定使用RTHD还是原始MambaLayer
            if use_rthd and s in rthd_stages:
                # 使用RTHD块
                print(f"Stage {s}: Using RTHDBlock (dim={features_per_stage[s]})")

                # 准备RTHD配置参数
                rthd_kwargs = {
                    'dim': features_per_stage[s],
                    'd_state': 16,
                    'ssm_ratio': 2.0,
                    'projection_mode': 'mean',
                    'reconstruction_mode': 'broadcast',
                    'use_ds_conv': True,
                    'norm_layer': norm_op,
                    'norm_kwargs': norm_op_kwargs,
                }

                # 如果提供了消融实验配置，则添加
                if rthd_config is not None:
                    rthd_kwargs.update(rthd_config)

                mamba_layers.append(RTHDBlock(**rthd_kwargs))
            else:
                # 使用原始MambaLayer
                print(f"Stage {s}: Using MambaLayer (dim={np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s]})")
                mamba_layers.append(
                    MambaLayer(
                        dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token = do_channel_token[s]
                    )
                )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    """解码器（原始版本，不使用RTHD）"""
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class UNetResDecoder_RTHD(nn.Module):
    """
    集成RTHD的解码器（支持部分stage使用RTHD）
    支持三种模式：none（不使用RTHD）、partial（部分stage使用）、full（所有stage使用）

    第二版增强：
    - SemanticSkipFusionGate: 语义引导的跳跃连接融合
    - BoundaryAttentionHead: 边界感知注意力
    - HighLowFrequencyRefinement: 高低频结构恢复
    """
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 rthd_config: dict = None,
                 decoder_rthd_mode: str = "full",
                 rthd_stages_decoder: List[int] = None,
                 # 第二版增强参数（默认全部关闭，向后兼容）
                 use_skip_fusion_gate: bool = False,
                 skip_gate_stages: List[int] = None,
                 skip_gate_reduction: int = 4,
                 skip_gate_type: str = "semantic",
                 use_boundary_attention_head: bool = False,
                 use_frequency_refinement: bool = False,
                 frequency_refinement_stages: List[int] = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.rthd_config = rthd_config or {}
        self.decoder_rthd_mode = decoder_rthd_mode

        # 第二版增强参数
        self.use_skip_fusion_gate = use_skip_fusion_gate
        self.use_boundary_attention_head = use_boundary_attention_head
        self.use_frequency_refinement = use_frequency_refinement
        self.skip_gate_type = skip_gate_type

        # 默认stage配置
        if skip_gate_stages is None:
            skip_gate_stages = [0, 1]
        if frequency_refinement_stages is None:
            frequency_refinement_stages = [0, 1]

        self.skip_gate_stages = skip_gate_stages
        self.frequency_refinement_stages = frequency_refinement_stages

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        # 确定哪些decoder stage使用RTHD
        if decoder_rthd_mode == "none":
            self.rthd_stages_decoder = []
            print(f"Decoder mode: none")
        elif decoder_rthd_mode == "partial":
            # 默认只在前两个decoder stage使用RTHD
            if rthd_stages_decoder is None:
                self.rthd_stages_decoder = [0, 1]
            else:
                self.rthd_stages_decoder = rthd_stages_decoder
            print(f"Decoder mode: partial, RTHD stages: {self.rthd_stages_decoder}")
        elif decoder_rthd_mode == "full":
            # 所有stage都使用RTHD
            self.rthd_stages_decoder = list(range(n_stages_encoder - 1))
            print(f"Decoder mode: full")
        else:
            raise ValueError(f"Invalid decoder_rthd_mode: {decoder_rthd_mode}. Must be 'none', 'partial', or 'full'")

        print(f"Decoder RTHD config: {self.rthd_config}")

        # 第二版增强模块打印信息
        if use_skip_fusion_gate:
            print(f"✓ Skip Fusion Gate enabled for stages: {skip_gate_stages}, type: {skip_gate_type}")
        if use_boundary_attention_head:
            print(f"✓ Boundary Attention Head enabled")
        if use_frequency_refinement:
            print(f"✓ Frequency Refinement enabled for stages: {frequency_refinement_stages}")

        # 导入第二版增强模块
        from .rthd_modules import (
            AttentionSkipFusionGate3d,
            BoundaryAttentionHead3d,
            HighLowFrequencyRefinement3d,
            SemanticSkipFusionGate3d,
        )

        stages = []
        upsample_layers = []
        seg_layers = []
        skip_gates = []  # 第二版：skip fusion gates
        frequency_refiners = []  # 第二版：frequency refinement modules

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            decoder_stage_idx = s - 1  # decoder stage index (0-based)

            # 上采样层
            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            # 判断当前stage是否使用RTHD
            use_rthd_this_stage = decoder_stage_idx in self.rthd_stages_decoder

            # 构建解码器stage
            stage_blocks = [
                # 第一个块：融合上采样特征和skip connection
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
            ]

            if use_rthd_this_stage:
                # 第二个块：RTHD三视图处理
                stage_blocks.append(
                    RTHDBlock(
                        dim=input_features_skip,
                        **self.rthd_config
                    )
                )
                # 额外的卷积块（如果需要）
                if n_conv_per_stage[s-1] > 2:
                    stage_blocks.extend([
                        BasicBlockD(
                            conv_op = encoder.conv_op,
                            input_channels = input_features_skip,
                            output_channels = input_features_skip,
                            kernel_size = encoder.kernel_sizes[-(s + 1)],
                            stride = 1,
                            conv_bias = encoder.conv_bias,
                            norm_op = encoder.norm_op,
                            norm_op_kwargs = encoder.norm_op_kwargs,
                            nonlin = encoder.nonlin,
                            nonlin_kwargs = encoder.nonlin_kwargs,
                        ) for _ in range(n_conv_per_stage[s-1] - 2)
                    ])
            else:
                # 不使用RTHD：只使用卷积块
                if n_conv_per_stage[s-1] > 1:
                    stage_blocks.extend([
                        BasicBlockD(
                            conv_op = encoder.conv_op,
                            input_channels = input_features_skip,
                            output_channels = input_features_skip,
                            kernel_size = encoder.kernel_sizes[-(s + 1)],
                            stride = 1,
                            conv_bias = encoder.conv_bias,
                            norm_op = encoder.norm_op,
                            norm_op_kwargs = encoder.norm_op_kwargs,
                            nonlin = encoder.nonlin,
                            nonlin_kwargs = encoder.nonlin_kwargs,
                        ) for _ in range(n_conv_per_stage[s-1] - 1)
                    ])

            stages.append(nn.Sequential(*stage_blocks))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

            # 第二版增强：skip fusion gate
            if use_skip_fusion_gate and decoder_stage_idx in skip_gate_stages:
                if skip_gate_type == "semantic":
                    skip_gate_cls = SemanticSkipFusionGate3d
                elif skip_gate_type == "attention":
                    skip_gate_cls = AttentionSkipFusionGate3d
                else:
                    raise ValueError(f"Invalid skip_gate_type: {skip_gate_type}. Must be 'semantic' or 'attention'")
                skip_gates.append(skip_gate_cls(dim=input_features_skip, reduction=skip_gate_reduction))
            else:
                skip_gates.append(nn.Identity())

            # 第二版增强：frequency refinement
            if use_frequency_refinement and decoder_stage_idx in frequency_refinement_stages:
                frequency_refiners.append(HighLowFrequencyRefinement3d(dim=input_features_skip))
            else:
                frequency_refiners.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.skip_gates = nn.ModuleList(skip_gates)
        self.frequency_refiners = nn.ModuleList(frequency_refiners)

        # 第二版增强：boundary attention head（只在最后stage使用）
        if use_boundary_attention_head:
            final_dim = encoder.output_channels[0]  # 最后一个decoder stage的特征维度
            self.final_boundary_attention = BoundaryAttentionHead3d(dim=final_dim)
        else:
            self.final_boundary_attention = nn.Identity()

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            # 上采样
            x_up = self.upsample_layers[s](lres_input)
            skip = skips[-(s+2)]

            # 第二版增强：语义引导skip融合门控
            if self.use_skip_fusion_gate and s in self.skip_gate_stages:
                skip = self.skip_gates[s](skip, x_up)

            # 融合skip和decoder特征
            x = torch.cat((x_up, skip), 1)

            # decoder stage处理
            x = self.stages[s](x)

            # 第二版增强：高低频结构恢复
            if self.use_frequency_refinement and s in self.frequency_refinement_stages:
                x = self.frequency_refiners[s](x)

            # 第二版增强：边界注意力（只在最后stage使用）
            if self.use_boundary_attention_head and s == (len(self.stages) - 1):
                x = self.final_boundary_attention(x)

            # 分割头
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

# 第二步
class UMambaEnc_RTHD(nn.Module):
    """
    UMambaEnc with RTHD
    集成三视图递归机制的U-Mamba编码器-解码器网络
    """
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None,
                 use_rthd: bool = True,
                 rthd_stages: List[int] = None,
                 rthd_config: dict = None,  # 统一配置（向后兼容）
                 rthd_config_encoder: dict = None,  # 编码器专用配置
                 rthd_config_decoder: dict = None,  # 解码器专用配置
                 use_rthd_decoder: bool = True,  # 是否使用RTHD解码器（向后兼容）
                 decoder_rthd_mode: str = "full",  # 解码器RTHD模式："none", "partial", "full"
                 rthd_stages_decoder: List[int] = None,  # 解码器使用RTHD的stage列表
                 # 第二版增强参数（默认全部关闭）
                 use_skip_fusion_gate: bool = False,
                 skip_gate_stages: List[int] = None,
                 skip_gate_reduction: int = 4,
                 skip_gate_type: str = "semantic",
                 use_boundary_attention_head: bool = False,
                 use_frequency_refinement: bool = False,
                 frequency_refinement_stages: List[int] = None,
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        # 处理配置回退逻辑
        # 优先级：rthd_config_encoder -> rthd_config -> 默认空字典
        final_encoder_config = rthd_config_encoder if rthd_config_encoder is not None else (rthd_config if rthd_config is not None else {})
        # 优先级：rthd_config_decoder -> rthd_config -> 默认空字典
        final_decoder_config = rthd_config_decoder if rthd_config_decoder is not None else (rthd_config if rthd_config is not None else {})

        print(f"Encoder RTHD config: {final_encoder_config}")
        print(f"Decoder RTHD config: {final_decoder_config}")

        self.encoder = ResidualMambaEncoder_RTHD(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels,
            use_rthd=use_rthd,
            rthd_stages=rthd_stages,
            rthd_config=final_encoder_config,  # 使用编码器专用配置
        )

        # 处理向后兼容：use_rthd_decoder=False 等价于 decoder_rthd_mode="none"
        if not use_rthd_decoder:
            decoder_rthd_mode = "none"

        # 创建解码器：根据decoder_rthd_mode选择
        if decoder_rthd_mode == "none":
            print("Using UNetResDecoder (原始卷积解码器)")
            self.decoder = UNetResDecoder(
                self.encoder,
                num_classes,
                n_conv_per_stage_decoder,
                deep_supervision,
            )
        else:
            print(f"Using UNetResDecoder_RTHD (decoder_rthd_mode={decoder_rthd_mode})")
            self.decoder = UNetResDecoder_RTHD(
                self.encoder,
                num_classes,
                n_conv_per_stage_decoder,
                deep_supervision,
                rthd_config=final_decoder_config,  # 使用解码器专用配置
                decoder_rthd_mode=decoder_rthd_mode,
                rthd_stages_decoder=rthd_stages_decoder,
                # 第二版增强参数
                use_skip_fusion_gate=use_skip_fusion_gate,
                skip_gate_stages=skip_gate_stages,
                skip_gate_reduction=skip_gate_reduction,
                skip_gate_type=skip_gate_type,
                use_boundary_attention_head=use_boundary_attention_head,
                use_frequency_refinement=use_frequency_refinement,
                frequency_refinement_stages=frequency_refinement_stages,
            )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


# 程序入口
def get_umamba_enc_rthd_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        rthd_config: dict = None,
        rthd_config_encoder: dict = None,
        rthd_config_decoder: dict = None,
        use_rthd_decoder: bool = True,
        decoder_rthd_mode: str = "full",
        rthd_stages_decoder: List[int] = None,
        # 第二版增强参数（默认全部关闭）
        use_skip_fusion_gate: bool = False,
        skip_gate_stages: List[int] = None,
        skip_gate_reduction: int = 4,
        skip_gate_type: str = "semantic",
        use_boundary_attention_head: bool = False,
        use_frequency_refinement: bool = False,
        frequency_refinement_stages: List[int] = None,
        rthd_stages_encoder: List[int] = None,
    ):
    """
    从plans创建UMambaEnc_RTHD网络

    Args:
        plans_manager: Plans管理器
        dataset_json: 数据集配置
        configuration_manager: 配置管理器
        num_input_channels: 输入通道数
        deep_supervision: 是否使用深度监督
        rthd_config: RTHD统一配置（向后兼容），包含:
            - view_mode: 'tri' or 'single'
            - share_weights: True or False
            - scan_mode: 'omni' or 'standard'
            - use_local_window: True or False
            - window_size: int (default 8)
            - reconstruction_mode: 'broadcast', 'weighted', 'gated'
            - cross_view_interaction: True or False
            - interaction_mode: 'post'
            - interaction_type: 'gate'
        rthd_config_encoder: 编码器专用RTHD配置（优先级高于rthd_config）
        rthd_config_decoder: 解码器专用RTHD配置（优先级高于rthd_config）
        rthd_stages_encoder: 编码器使用RTHD的stage列表；None保持原有[0,1,2,3,4]，[]表示全部使用原始MambaLayer
        use_rthd_decoder: 是否在解码器使用RTHD（向后兼容，False等价于decoder_rthd_mode="none"）
        decoder_rthd_mode: 解码器RTHD模式 - "none", "partial", "full" (默认"full")
        rthd_stages_decoder: 解码器使用RTHD的stage列表（仅在decoder_rthd_mode="partial"时生效）

        第二版增强参数（默认全部关闭，确保向后兼容）：
        use_skip_fusion_gate: 是否启用语义引导skip融合门控
        skip_gate_stages: skip gate作用的stage列表（默认[0,1]）
        skip_gate_reduction: skip gate隐藏层降维比例（默认4）
        skip_gate_type: skip gate类型，semantic为本文残差双向门控，attention为传统0-1注意力门控
        use_boundary_attention_head: 是否启用边界注意力头
        use_frequency_refinement: 是否启用高低频结构恢复
        frequency_refinement_stages: 频率恢复作用的stage列表（默认[0,1]）
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc_RTHD'
    network_class = UMambaEnc_RTHD

    # 默认RTHD配置（消融实验 #5: 全局平铺版）
    default_rthd_config = {
        'view_mode': 'tri',
        'share_weights': True,
        'scan_mode': 'omni',
        'use_local_window': False,
        'window_size': 8,
    }

    # 如果提供了自定义配置，则合并
    if rthd_config is not None:
        default_rthd_config.update(rthd_config)

    if rthd_stages_encoder is None:
        rthd_stages_encoder = [0, 1, 2, 3, 4]

    kwargs = {
        'UMambaEnc_RTHD': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU,
            'nonlin_kwargs': {'inplace': True},
            'use_rthd': True,  # 启用RTHD
            'rthd_stages': rthd_stages_encoder,
            'rthd_config': default_rthd_config,  # 统一配置（向后兼容）
            'rthd_config_encoder': rthd_config_encoder,  # 编码器专用配置
            'rthd_config_decoder': rthd_config_decoder,  # 解码器专用配置
            'use_rthd_decoder': use_rthd_decoder,  # 向后兼容参数
            'decoder_rthd_mode': decoder_rthd_mode,  # 解码器模式
            'rthd_stages_decoder': rthd_stages_decoder,  # 解码器RTHD stage列表
            # 第二版增强参数
            'use_skip_fusion_gate': use_skip_fusion_gate,
            'skip_gate_stages': skip_gate_stages,
            'skip_gate_reduction': skip_gate_reduction,
            'skip_gate_type': skip_gate_type,
            'use_boundary_attention_head': use_boundary_attention_head,
            'use_frequency_refinement': use_frequency_refinement,
            'frequency_refinement_stages': frequency_refinement_stages,
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )

    model.apply(InitWeights_He(1e-2))

    return model
