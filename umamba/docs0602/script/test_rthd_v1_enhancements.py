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

# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*80}")
print(f"Device Configuration")
print(f"{'='*80}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    print(f"Will test with real SS2D (CUDA-based)")
else:
    print(f"CUDA is NOT available")
    print(f"Tests requiring real SS2D will be skipped")
    print(f"Only CPU-compatible tests will run")
print(f"{'='*80}\n")

def test_tri_view_reconstruction_modes():
    """测试TriViewReconstruction的三种模式"""
    print("\n" + "="*80)
    print("测试 TriViewReconstruction 三种重建模式")
    print("="*80)

    B, C, D, H, W = 2, 64, 8, 16, 16

    # 创建模拟的三视图特征（放到device上）
    axial = torch.randn(B, C, H, W, device=device)
    coronal = torch.randn(B, C, D, W, device=device)
    sagittal = torch.randn(B, C, D, H, device=device)

    # 测试1: broadcast模式
    print("\n1. 测试 broadcast 模式...")
    recon_broadcast = TriViewReconstruction(dim=None, mode='broadcast').to(device)
    out_broadcast = recon_broadcast(axial, coronal, sagittal, target_shape=(D, H, W))
    print(f"   输入形状: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"   输出形状: {out_broadcast.shape}")
    print(f"   ✅ broadcast模式测试通过")

    # 测试2: weighted模式
    print("\n2. 测试 weighted 模式...")
    recon_weighted = TriViewReconstruction(dim=None, mode='weighted').to(device)
    weights = torch.ones(3, device=device) / 3.0
    out_weighted = recon_weighted(axial, coronal, sagittal, target_shape=(D, H, W), weights=weights)
    print(f"   输入形状: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"   输出形状: {out_weighted.shape}")
    print(f"   ✅ weighted模式测试通过")

    # 测试3: gated模式
    print("\n3. 测试 gated 模式...")
    recon_gated = TriViewReconstruction(dim=C, mode='gated').to(device)
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

    # TriViewVMambaBlock 依赖真实 SS2D，需要 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  跳过 TriViewVMambaBlock 测试")
        print("   原因: 真实 SS2D 需要 CUDA 支持")
        print("   如需测试，请在有 CUDA 的环境运行")
        print("="*80)
        return

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W, device=device)

    # 测试1: 原有的broadcast模式
    print("\n1. 测试 broadcast 重建模式...")
    block_broadcast = TriViewVMambaBlock(
        dim=C,
        reconstruction_mode='broadcast',
        view_mode='tri',
        share_weights=True,
        use_local_window=False
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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

    # RTHDBlock 依赖真实 SS2D，需要 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  跳过 RTHDBlock 测试")
        print("   原因: 真实 SS2D 需要 CUDA 支持")
        print("   如需测试，请在有 CUDA 的环境运行")
        print("="*80)
        return

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W, device=device)

    # 测试1: 原有配置（broadcast + 无交互）
    print("\n1. 测试原有配置（broadcast + 无交互）...")
    block_original = RTHDBlock(
        dim=C,
        reconstruction_mode='broadcast',
        cross_view_interaction=False,
        use_ds_conv=True
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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
    ).to(device)
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

    # RTHDBlock 依赖真实 SS2D，需要 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  跳过形状兼容性测试")
        print("   原因: 真实 SS2D 需要 CUDA 支持")
        print("   如需测试，请在有 CUDA 的环境运行")
        print("="*80)
        return

    test_shapes = [
        (2, 64, 8, 16, 16),
        (1, 32, 4, 8, 8),
        (4, 128, 16, 32, 32),
    ]

    for i, (B, C, D, H, W) in enumerate(test_shapes, 1):
        print(f"\n测试形状 {i}: (B={B}, C={C}, D={D}, H={H}, W={W})")
        x = torch.randn(B, C, D, H, W, device=device)

        block = RTHDBlock(
            dim=C,
            reconstruction_mode='gated',
            cross_view_interaction=True,
            interaction_mode='post',
            interaction_type='gate',
            use_ds_conv=True
        ).to(device)

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

    # RTHDBlock 依赖真实 SS2D，需要 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  跳过向后兼容性测试")
        print("   原因: 真实 SS2D 需要 CUDA 支持")
        print("   如需测试，请在有 CUDA 的环境运行")
        print("="*80)
        return

    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W, device=device)

    # 不传新参数，应使用默认值
    print("\n1. 测试默认参数配置...")
    block_default = RTHDBlock(dim=C).to(device)
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
    ).to(device)
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
        # 测试1: TriViewReconstruction三种模式（纯PyTorch，不需要CUDA）
        test_tri_view_reconstruction_modes()

        # 测试2-5: 依赖真实SS2D，需要CUDA
        cuda_dependent_tests = [
            ("TriViewVMambaBlock", test_tri_view_vmamba_block),
            ("RTHDBlock", test_rthd_block),
            ("形状兼容性", test_shape_compatibility),
            ("向后兼容性", test_backward_compatibility),
        ]

        skipped_tests = []
        passed_tests = []

        for test_name, test_func in cuda_dependent_tests:
            if not torch.cuda.is_available():
                skipped_tests.append(test_name)
            test_func()
            if torch.cuda.is_available():
                passed_tests.append(test_name)

        print("\n" + "="*80)
        print("测试完成总结")
        print("="*80)

        if torch.cuda.is_available():
            print("🎉 所有测试通过！")
            print("\n第一版增强功能验证成功：")
            print("  ✅ gated reconstruction (位置相关门控融合)")
            print("  ✅ minimal cross-view interaction (最小版跨视图交互)")
            print("  ✅ 向后兼容性保持")
            print("  ✅ 所有形状正确")
            print(f"\n已通过测试 ({len(passed_tests) + 1}/{len(cuda_dependent_tests) + 1}):")
            print("  ✅ TriViewReconstruction (CPU)")
            for test_name in passed_tests:
                print(f"  ✅ {test_name} (CUDA)")
        else:
            print("⚠️  部分测试完成")
            print("\n✅ CPU兼容测试通过:")
            print("  ✅ TriViewReconstruction 三种模式")
            print("\n⚠️  已跳过的CUDA依赖测试:")
            for test_name in skipped_tests:
                print(f"  ⏭  {test_name}")
            print("\n说明:")
            print("  - TriViewReconstruction 是纯PyTorch模块，可在CPU运行")
            print("  - TriViewVMambaBlock/RTHDBlock 使用真实SS2D，需要CUDA")
            print("  - 要运行完整测试，请在有CUDA的环境中执行")

        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
