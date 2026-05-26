"""
测试RTHD消融实验配置
验证所有5个消融实验的维度匹配和前向传播
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from umamba.nnunetv2.nets.rthd_modules import RTHDBlock, TriViewVMambaBlock, window_partition, window_reverse


def test_window_partition():
    """测试局部滑窗机制"""
    print("\n" + "="*80)
    print("测试 1: 局部滑窗机制 (window_partition & window_reverse)")
    print("="*80)

    B, C, H, W = 2, 64, 32, 32
    window_size = 8

    x = torch.randn(B, C, H, W)
    print(f"输入: {x.shape}")

    # 窗口分割
    windows, (H_pad, W_pad) = window_partition(x, window_size)
    print(f"窗口分割后: {windows.shape}, padding后尺寸: ({H_pad}, {W_pad})")

    # 窗口合并
    x_recon = window_reverse(windows, window_size, H_pad, W_pad, H, W)
    print(f"窗口合并后: {x_recon.shape}")

    # 验证维度
    assert x_recon.shape == x.shape, f"维度不匹配: {x_recon.shape} != {x.shape}"
    print("✓ 维度匹配正确")


def test_ablation_configs():
    """测试所有5个消融实验配置"""

    configs = {
        "#2 单视图降级版": {
            'view_mode': 'single',
            'share_weights': False,
            'scan_mode': 'standard',
            'use_local_window': False,
        },
        "#3 独立参数版": {
            'view_mode': 'tri',
            'share_weights': False,
            'scan_mode': 'omni',
            'use_local_window': True,
        },
        "#4 常规扫描版": {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'standard',
            'use_local_window': True,
        },
        "#5 全局平铺版": {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
            'use_local_window': False,
        },
        "#6 完整创新版": {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
            'use_local_window': True,
        },
    }

    # 测试输入
    B, C, D, H, W = 2, 64, 16, 32, 32
    x = torch.randn(B, C, D, H, W)

    for name, config in configs.items():
        print("\n" + "="*80)
        print(f"测试消融实验: {name}")
        print(f"配置: {config}")
        print("="*80)

        try:
            # 创建RTHDBlock
            block = RTHDBlock(
                dim=C,
                d_state=16,
                ssm_ratio=2.0,
                use_ds_conv=True,
                **config
            )

            # 前向传播
            with torch.no_grad():
                out = block(x)

            # 验证输出维度
            assert out.shape == x.shape, f"输出维度不匹配: {out.shape} != {x.shape}"

            print(f"输入: {x.shape}")
            print(f"输出: {out.shape}")
            print(f"✓ {name} 测试通过")

        except Exception as e:
            print(f"✗ {name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()


def test_triview_vmamba_block():
    """测试TriViewVMambaBlock的不同配置"""
    print("\n" + "="*80)
    print("测试 TriViewVMambaBlock 各种配置")
    print("="*80)

    B, C, D, H, W = 2, 64, 16, 32, 32
    x = torch.randn(B, C, D, H, W)

    test_cases = [
        ("三视图 + 参数共享 + 全局", {'view_mode': 'tri', 'share_weights': True, 'use_local_window': False}),
        ("三视图 + 参数共享 + 局部窗口", {'view_mode': 'tri', 'share_weights': True, 'use_local_window': True}),
        ("三视图 + 独立参数 + 局部窗口", {'view_mode': 'tri', 'share_weights': False, 'use_local_window': True}),
        ("单视图 + 全局", {'view_mode': 'single', 'share_weights': False, 'use_local_window': False}),
    ]

    for name, config in test_cases:
        print(f"\n测试: {name}")
        try:
            block = TriViewVMambaBlock(dim=C, **config)
            with torch.no_grad():
                out = block(x)
            assert out.shape == x.shape
            print(f"  ✓ 输入: {x.shape} -> 输出: {out.shape}")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RTHD 消融实验配置测试")
    print("="*80)

    # 测试1: 局部滑窗机制
    test_window_partition()

    # 测试2: TriViewVMambaBlock
    test_triview_vmamba_block()

    # 测试3: 所有消融实验配置
    test_ablation_configs()

    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)
