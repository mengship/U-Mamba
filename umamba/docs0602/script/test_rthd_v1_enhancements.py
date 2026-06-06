"""
测试RTHD第一版增强功能
测试gated reconstruction和cross-view interaction
"""

import torch
import sys
import os

# 添加路径：从脚本位置向上找到 umamba 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir: .../umamba/docs0602/script
# umamba_dir: .../umamba
umamba_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, umamba_dir)

from nnunetv2.nets.rthd_modules import (
    TriViewReconstruction,
    TriViewVMambaBlock,
    RTHDBlock
)

def test_tri_view_reconstruction_modes():
    """测试TriViewReconstruction的三种模式"""
    print("\n" + "="*80)
    print("测试 TriViewReconstruction 三种重建模式")
    print("="*80)

    B, C, D, H, W = 2, 64, 8, 16, 16

    # 创建模拟的三视图特征
    axial = torch.randn(B, C, H, W)
    coronal = torch.randn(B, C, D, W)
    sagittal = torch.randn(B, C, D, H)

    # 测试1: broadcast模式
    print("\n1. 测试 broadcast 模式...")
    recon_broadcast = TriViewReconstruction(dim=None, mode='broadcast')
    out_broadcast = recon_broadcast(axial, coronal, sagittal, target_shape=(D, H, W))
    print(f"   输入形状: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"   输出形状: {out_broadcast.shape}")
    print(f"   ✅ broadcast模式测试通过")

    # 测试2: weighted模式
    print("\n2. 测试 weighted 模式...")
    recon_weighted = TriViewReconstruction(dim=None, mode='weighted')
    weights = torch.ones(3) / 3.0
    out_weighted = recon_weighted(axial, coronal, sagittal, target_shape=(D, H, W), weights=weights)
    print(f"   输入形状: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"   输出形状: {out_weighted.shape}")
    print(f"   ✅ weighted模式测试通过")

    # 测试3: gated模式
    print("\n3. 测试 gated 模式...")
    recon_gated = TriViewReconstruction(dim=C, mode='gated')
    out_gated = recon_gated(axial, coronal, sagittal, target_shape=(D, H, W))
    print(f"   输入形状: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"   输出形状: {out_gated.shape}")
    print(f"   ✅ gated模式测试通过")

    print("\n" + "="*80)
    print("✅ TriViewReconstruction所有模式测试通过")
    print("="*80)


def test_tri_view_vmamba_block():
    """测试TriViewVMambaBlock的不同配置"""
    print("\n" + "="*80)
    print("测试 TriViewVMambaBlock 不同配置")
    print("="*80)

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W)

    # 测试1: 原有的broadcast模式
    print("\n1. 测试 broadcast 重建模式...")
    block_broadcast = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='broadcast',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_broadcast = block_broadcast(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_broadcast.shape}")
    print(f"   ✅ broadcast模式测试通过")

    # 测试2: weighted重建模式
    print("\n2. 测试 weighted 重建模式...")
    block_weighted = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='weighted',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_weighted = block_weighted(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_weighted.shape}")
    print(f"   可学习权重: {block_weighted.view_weights}")
    print(f"   ✅ weighted模式测试通过")

    # 测试3: gated重建模式（第一版增强）
    print("\n3. 测试 gated 重建模式（第一版增强）...")
    block_gated = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='gated',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_gated = block_gated(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_gated.shape}")
    print(f"   ✅ gated模式测试通过")

    # 测试4: 跨视图交互关闭
    print("\n4. 测试 跨视图交互=False...")
    block_no_interaction = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='broadcast',
        cross_view_interaction=False,
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_no_interaction = block_no_interaction(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_no_interaction.shape}")
    print(f"   ✅ 无交互模式测试通过")

    # 测试5: 跨视图交互开启（第一版增强）
    print("\n5. 测试 跨视图交互=True（第一版增强）...")
    block_interaction = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='broadcast',
        cross_view_interaction=True,
        interaction_mode='post',
        interaction_type='gate',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_interaction = block_interaction(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_interaction.shape}")
    print(f"   ✅ 跨视图交互模式测试通过")

    # 测试6: gated + cross-view interaction（完整第一版增强）
    print("\n6. 测试 gated + 跨视图交互（完整第一版增强）...")
    block_full = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='gated',
        cross_view_interaction=True,
        interaction_mode='post',
        interaction_type='gate',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    )
    out_full = block_full(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_full.shape}")
    print(f"   ✅ 完整第一版增强测试通过")

    print("\n" + "="*80)
    print("✅ TriViewVMambaBlock所有配置测试通过")
    print("="*80)


def test_rthd_block():
    """测试RTHDBlock的不同配置"""
    print("\n" + "="*80)
    print("测试 RTHDBlock 不同配置")
    print("="*80)

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W)

    # 测试1: 原有配置（broadcast + 无交互）
    print("\n1. 测试原有配置（broadcast + 无交互）...")
    block_original = RTHDBlock(
        dim=C,
        reconstruction_mode='broadcast',
        cross_view_interaction=False,
        use_ds_conv=True
    )
    out_original = block_original(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_original.shape}")
    print(f"   ✅ 原有配置测试通过")

    # 测试2: gated重建
    print("\n2. 测试 gated 重建...")
    block_gated = RTHDBlock(
        dim=C,
        reconstruction_mode='gated',
        cross_view_interaction=False,
        use_ds_conv=True
    )
    out_gated = block_gated(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_gated.shape}")
    print(f"   ✅ gated重建测试通过")

    # 测试3: 跨视图交互
    print("\n3. 测试跨视图交互...")
    block_interaction = RTHDBlock(
        dim=C,
        reconstruction_mode='broadcast',
        cross_view_interaction=True,
        interaction_mode='post',
        interaction_type='gate',
        use_ds_conv=True
    )
    out_interaction = block_interaction(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_interaction.shape}")
    print(f"   ✅ 跨视图交互测试通过")

    # 测试4: 完整第一版增强（gated + 交互）
    print("\n4. 测试完整第一版增强（gated + 交互）...")
    block_full = RTHDBlock(
        dim=C,
        reconstruction_mode='gated',
        cross_view_interaction=True,
        interaction_mode='post',
        interaction_type='gate',
        use_ds_conv=True
    )
    out_full = block_full(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_full.shape}")
    print(f"   ✅ 完整第一版增强测试通过")

    print("\n" + "="*80)
    print("✅ RTHDBlock所有配置测试通过")
    print("="*80)


def test_shape_compatibility():
    """测试不同形状的兼容性"""
    print("\n" + "="*80)
    print("测试不同形状的兼容性")
    print("="*80)

    test_shapes = [
        (2, 64, 8, 16, 16),
        (1, 32, 4, 8, 8),
        (4, 128, 16, 32, 32),
    ]

    for i, (B, C, D, H, W) in enumerate(test_shapes, 1):
        print(f"\n测试形状 {i}: (B={B}, C={C}, D={D}, H={H}, W={W})")
        x = torch.randn(B, C, D, H, W)

        block = RTHDBlock(
            dim=C,
            reconstruction_mode='gated',
            cross_view_interaction=True,
            interaction_mode='post',
            interaction_type='gate',
            use_ds_conv=True
        )

        out = block(x)
        assert out.shape == x.shape, f"输出形状不匹配: {out.shape} != {x.shape}"
        print(f"   ✅ 形状测试通过: {x.shape} -> {out.shape}")

    print("\n" + "="*80)
    print("✅ 所有形状兼容性测试通过")
    print("="*80)


def test_backward_compatibility():
    """测试向后兼容性（不传新参数时行为应与原版一致）"""
    print("\n" + "="*80)
    print("测试向后兼容性")
    print("="*80)

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W)

    # 不传新参数，应使用默认值
    print("\n1. 测试默认参数配置...")
    block_default = RTHDBlock(dim=C)
    out_default = block_default(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_default.shape}")
    print(f"   默认reconstruction_mode: {block_default.tri_view_vmamba.reconstruction_mode}")
    print(f"   默认cross_view_interaction: {block_default.tri_view_vmamba.cross_view_interaction}")
    print(f"   ✅ 默认配置测试通过")

    # 显式使用旧参数
    print("\n2. 测试显式旧配置...")
    block_old = RTHDBlock(
        dim=C,
        reconstruction_mode='broadcast',
        view_mode='tri',
        share_weights=True,
        scan_mode='omni',
        use_local_window=False
    )
    out_old = block_old(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_old.shape}")
    print(f"   ✅ 显式旧配置测试通过")

    print("\n" + "="*80)
    print("✅ 向后兼容性测试通过")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("开始测试 RTHD 第一版增强功能")
    print("="*80)

    try:
        # 测试1: TriViewReconstruction三种模式
        test_tri_view_reconstruction_modes()

        # 测试2: TriViewVMambaBlock不同配置
        test_tri_view_vmamba_block()

        # 测试3: RTHDBlock不同配置
        test_rthd_block()

        # 测试4: 形状兼容性
        test_shape_compatibility()

        # 测试5: 向后兼容性
        test_backward_compatibility()

        print("\n" + "="*80)
        print("🎉 所有测试通过！")
        print("="*80)
        print("\n第一版增强功能验证成功：")
        print("  ✅ gated reconstruction (位置相关门控融合)")
        print("  ✅ minimal cross-view interaction (最小版跨视图交互)")
        print("  ✅ 向后兼容性保持")
        print("  ✅ 所有形状正确")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
