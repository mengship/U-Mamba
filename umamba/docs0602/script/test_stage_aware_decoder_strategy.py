"""
测试脚本：Stage-Aware Decoder Strategy (第二版增强)

测试内容：
1. SemanticSkipFusionGate3d - 语义引导跳跃连接融合门控
2. BoundaryAttentionHead3d - 边界感知注意力头
3. HighLowFrequencyRefinement3d - 高低频结构恢复
4. UMambaEnc_RTHD - 完整网络前向传播测试

作者：研究生脑肿瘤分割项目
日期：2026-06-07
"""

import torch
import sys
import os

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
umamba_dir = os.path.dirname(os.path.dirname(current_dir))
if umamba_dir not in sys.path:
    sys.path.insert(0, umamba_dir)

from nnunetv2.nets.rthd_modules import (
    SemanticSkipFusionGate3d,
    BoundaryAttentionHead3d,
    HighLowFrequencyRefinement3d
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_semantic_skip_fusion_gate():
    """测试语义引导跳跃连接融合门控"""
    print("\n" + "=" * 80)
    print("测试 1: SemanticSkipFusionGate3d")
    print("=" * 80)

    dim = 64
    gate = SemanticSkipFusionGate3d(dim=dim, reduction=4).to(device)

    # 测试1.1: skip和decoder shape一致
    print("\n1.1 测试相同shape的skip和decoder")
    skip = torch.randn(2, dim, 8, 16, 16, device=device)
    decoder = torch.randn(2, dim, 8, 16, 16, device=device)
    output = gate(skip, decoder)
    print(f"  Skip shape: {skip.shape}")
    print(f"  Decoder shape: {decoder.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == skip.shape, f"输出shape错误: {output.shape} != {skip.shape}"
    print("  ✓ 输出shape正确")

    # 测试1.2: skip和decoder shape不一致（需要对齐）
    print("\n1.2 测试不同shape的skip和decoder（自动对齐）")
    skip = torch.randn(2, dim, 16, 32, 32, device=device)
    decoder = torch.randn(2, dim, 8, 16, 16, device=device)  # 更小的decoder
    output = gate(skip, decoder)
    print(f"  Skip shape: {skip.shape}")
    print(f"  Decoder shape: {decoder.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == skip.shape, f"输出shape错误: {output.shape} != {skip.shape}"
    print("  ✓ 自动对齐成功，输出shape正确")

    # 测试1.3: 检查残差式门控（输出不应该是零）
    print("\n1.3 测试残差式门控效果")
    skip = torch.randn(2, dim, 8, 16, 16, device=device)
    decoder = torch.randn(2, dim, 8, 16, 16, device=device)
    output = gate(skip, decoder)
    diff = (output - skip).abs().mean().item()
    print(f"  输出与输入差异（平均绝对值）: {diff:.6f}")
    assert diff > 0, "门控没有产生效果"
    print("  ✓ 门控正常工作")

    print("\n✓ SemanticSkipFusionGate3d 所有测试通过")


def test_boundary_attention_head():
    """测试边界感知注意力头"""
    print("\n" + "=" * 80)
    print("测试 2: BoundaryAttentionHead3d")
    print("=" * 80)

    dim = 32
    attn_head = BoundaryAttentionHead3d(dim=dim).to(device)

    # 测试2.1: 输入输出shape一致
    print("\n2.1 测试输入输出shape")
    x = torch.randn(2, dim, 16, 32, 32, device=device)
    output = attn_head(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, f"输出shape错误: {output.shape} != {x.shape}"
    print("  ✓ 输出shape正确")

    # 测试2.2: 检查注意力增强效果
    print("\n2.2 测试注意力增强效果")
    x = torch.randn(2, dim, 8, 16, 16, device=device)
    output = attn_head(x)
    diff = (output - x).abs().mean().item()
    print(f"  输出与输入差异（平均绝对值）: {diff:.6f}")
    assert diff > 0, "注意力头没有产生效果"
    print("  ✓ 注意力增强正常工作")

    print("\n✓ BoundaryAttentionHead3d 所有测试通过")


def test_high_low_frequency_refinement():
    """测试高低频结构恢复"""
    print("\n" + "=" * 80)
    print("测试 3: HighLowFrequencyRefinement3d")
    print("=" * 80)

    dim = 48
    freq_refiner = HighLowFrequencyRefinement3d(dim=dim).to(device)

    # 测试3.1: 输入输出shape一致
    print("\n3.1 测试输入输出shape")
    x = torch.randn(2, dim, 12, 24, 24, device=device)
    output = freq_refiner(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, f"输出shape错误: {output.shape} != {x.shape}"
    print("  ✓ 输出shape正确")

    # 测试3.2: 检查频率恢复效果
    print("\n3.2 测试频率恢复效果")
    x = torch.randn(2, dim, 8, 16, 16, device=device)
    output = freq_refiner(x)
    diff = (output - x).abs().mean().item()
    print(f"  输出与输入差异（平均绝对值）: {diff:.6f}")
    assert diff > 0, "频率恢复没有产生效果"
    print("  ✓ 频率恢复正常工作")

    print("\n✓ HighLowFrequencyRefinement3d 所有测试通过")


def test_umamba_enc_rthd_stage_aware():
    """测试完整的UMambaEnc_RTHD网络（第二版配置）"""
    print("\n" + "=" * 80)
    print("测试 4: UMambaEnc_RTHD with Stage-Aware Decoder")
    print("=" * 80)
    print(f"Using device for full network test: {device}")

    if not torch.cuda.is_available():
        print("\n⚠️  跳过完整网络测试")
        print("   原因: 真实 SS2D/Mamba CUDA kernel 需要 CUDA tensor")
        print("=" * 80)
        return

    from nnunetv2.nets.UMambaEnc_RTHD import UMambaEnc_RTHD
    from torch import nn

    # 小尺寸测试配置
    input_size = (32, 64, 64)
    input_channels = 4
    n_stages = 4  # 简化为4个stage
    features_per_stage = [32, 64, 128, 256]
    kernel_sizes = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2]
    num_classes = 4

    # 编码器RTHD配置（第一版增强）
    rthd_config_encoder = {
        "view_mode": "tri",
        "share_weights": True,
        "scan_mode": "omni",
        "use_local_window": True,
        "window_size": 8,
        "reconstruction_mode": "gated",
        "cross_view_interaction": True,
        "interaction_mode": "post",
        "interaction_type": "gate",
    }

    # 解码器RTHD配置（第一版增强）
    rthd_config_decoder = {
        "view_mode": "tri",
        "share_weights": True,
        "scan_mode": "omni",
        "use_local_window": False,
        "window_size": 8,
        "reconstruction_mode": "gated",
        "cross_view_interaction": True,
        "interaction_mode": "post",
        "interaction_type": "gate",
    }

    print("\n4.1 测试 decoder_rthd_mode='partial' + 第二版增强")
    model = UMambaEnc_RTHD(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
        use_rthd=True,
        rthd_stages=[0, 1, 2],  # 前3个encoder stage使用RTHD
        rthd_config_encoder=rthd_config_encoder,
        rthd_config_decoder=rthd_config_decoder,
        use_rthd_decoder=True,
        decoder_rthd_mode="partial",
        rthd_stages_decoder=[0, 1],  # 只在D3/D2使用RTHD
        # 第二版增强参数
        use_skip_fusion_gate=True,
        skip_gate_stages=[0, 1],
        skip_gate_reduction=4,
        use_boundary_attention_head=True,
        use_frequency_refinement=False,
        frequency_refinement_stages=None,
    ).to(device)

    # 测试前向传播
    x = torch.randn(1, input_channels, *input_size, device=device)
    print(f"\n  Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}")
    print(f"  Output type: {type(output)}")
    assert isinstance(output, torch.Tensor), "deep_supervision=False应该返回tensor"
    assert output.shape[0] == 1, "Batch size错误"
    assert output.shape[1] == num_classes, f"类别数错误: {output.shape[1]} != {num_classes}"
    print("  ✓ 输出格式正确")

    print("\n4.2 测试 deep_supervision=True")
    model_ds = UMambaEnc_RTHD(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,  # 启用深度监督
        use_rthd=True,
        rthd_stages=[0, 1, 2],
        rthd_config_encoder=rthd_config_encoder,
        rthd_config_decoder=rthd_config_decoder,
        decoder_rthd_mode="partial",
        rthd_stages_decoder=[0, 1],
        # 第二版增强参数
        use_skip_fusion_gate=True,
        skip_gate_stages=[0, 1],
        use_boundary_attention_head=True,
        use_frequency_refinement=False,
        frequency_refinement_stages=None,
    ).to(device)

    with torch.no_grad():
        output_ds = model_ds(x)

    print(f"  Output type: {type(output_ds)}")
    print(f"  Output length: {len(output_ds)}")
    assert isinstance(output_ds, list), "deep_supervision=True应该返回list"
    print("  ✓ Deep supervision输出格式正确")

    # 测试4.3: 向后兼容（第二版增强全部关闭）
    print("\n4.3 测试向后兼容（第二版增强全部关闭）")
    model_compat = UMambaEnc_RTHD(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
        use_rthd=True,
        rthd_stages=[0, 1, 2],
        rthd_config_encoder=rthd_config_encoder,
        rthd_config_decoder=rthd_config_decoder,
        decoder_rthd_mode="partial",
        rthd_stages_decoder=[0, 1],
        # 第二版增强全部关闭（默认行为）
        use_skip_fusion_gate=False,
        use_boundary_attention_head=False,
        use_frequency_refinement=False,
    ).to(device)

    with torch.no_grad():
        output_compat = model_compat(x)

    print(f"  Output shape: {output_compat.shape}")
    assert isinstance(output_compat, torch.Tensor), "向后兼容模式应该返回tensor"
    print("  ✓ 向后兼容测试通过")

    print("\n✓ UMambaEnc_RTHD Stage-Aware Decoder 所有测试通过")


def main():
    print("=" * 80)
    print("Stage-Aware Decoder Strategy 测试套件")
    print("=" * 80)
    print("测试第二版增强模块和完整网络")
    print(f"Using device: {device}")
    print()

    try:
        # 测试各个模块
        test_semantic_skip_fusion_gate()
        test_boundary_attention_head()
        test_high_low_frequency_refinement()
        test_umamba_enc_rthd_stage_aware()

        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n第二版增强模块总结:")
        print("  1. SemanticSkipFusionGate3d - 语义引导skip融合门控")
        print("  2. BoundaryAttentionHead3d - 边界感知注意力头")
        print("  3. HighLowFrequencyRefinement3d - 高低频结构恢复（可选消融）")
        print("  4. UMambaEnc_RTHD - 阶段感知解码器集成（默认不启用频率恢复）")
        print()
        print("建议模型名称: RTHD-StageAwareDecoder")
        print("论文描述: 本文在解码阶段设计阶段感知的结构恢复策略，")
        print("通过低分辨率RTHD refinement建模全局结构、语义引导跳连融合")
        print("恢复局部细节，并结合边界注意力增强肿瘤轮廓质量。")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
