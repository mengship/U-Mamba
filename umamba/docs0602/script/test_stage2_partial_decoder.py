"""
阶段二第一步：Partial Decoder RTHD 和 Encoder/Decoder 分离配置测试
测试三种decoder模式：none, partial, full
测试encoder/decoder分离配置
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from nnunetv2.nets.UMambaEnc_RTHD import UMambaEnc_RTHD


def test_decoder_modes():
    """测试三种decoder模式：none, partial, full"""
    print("\n" + "="*80)
    print("测试阶段二功能：Partial Decoder RTHD")
    print("="*80)

    # 通用参数
    input_size = (32, 64, 64)
    input_channels = 4
    num_classes = 3
    n_stages = 5
    features_per_stage = [32, 64, 128, 256, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [2, 2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2, 2]

    # 基础配置
    base_config = {
        'input_size': input_size,
        'input_channels': input_channels,
        'n_stages': n_stages,
        'features_per_stage': features_per_stage,
        'conv_op': conv_op,
        'kernel_sizes': kernel_sizes,
        'strides': strides,
        'n_conv_per_stage': n_conv_per_stage,
        'num_classes': num_classes,
        'n_conv_per_stage_decoder': n_conv_per_stage_decoder,
        'conv_bias': True,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': False,
        'use_rthd': True,
        'rthd_stages': [0, 1, 2],
    }

    # 测试1：decoder_rthd_mode="none"
    print("\n" + "-"*80)
    print("测试1: decoder_rthd_mode='none' (不使用RTHD解码器)")
    print("-"*80)
    try:
        model_none = UMambaEnc_RTHD(
            **base_config,
            decoder_rthd_mode="none",
        )
        print(f"✓ 模型构造成功")
        print(f"  - Decoder类型: {type(model_none.decoder).__name__}")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    # 测试2：decoder_rthd_mode="partial", rthd_stages_decoder=[0, 1]
    print("\n" + "-"*80)
    print("测试2: decoder_rthd_mode='partial', rthd_stages_decoder=[0, 1]")
    print("-"*80)
    try:
        model_partial = UMambaEnc_RTHD(
            **base_config,
            decoder_rthd_mode="partial",
            rthd_stages_decoder=[0, 1],
        )
        print(f"✓ 模型构造成功")
        print(f"  - Decoder类型: {type(model_partial.decoder).__name__}")
        print(f"  - RTHD stages: {model_partial.decoder.rthd_stages_decoder}")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    # 测试3：decoder_rthd_mode="full"
    print("\n" + "-"*80)
    print("测试3: decoder_rthd_mode='full' (所有stage使用RTHD)")
    print("-"*80)
    try:
        model_full = UMambaEnc_RTHD(
            **base_config,
            decoder_rthd_mode="full",
        )
        print(f"✓ 模型构造成功")
        print(f"  - Decoder类型: {type(model_full.decoder).__name__}")
        print(f"  - RTHD stages: {model_full.decoder.rthd_stages_decoder}")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    # 测试4：向后兼容性 - use_rthd_decoder=False
    print("\n" + "-"*80)
    print("测试4: 向后兼容 - use_rthd_decoder=False")
    print("-"*80)
    try:
        model_compat = UMambaEnc_RTHD(
            **base_config,
            use_rthd_decoder=False,
        )
        print(f"✓ 模型构造成功")
        print(f"  - Decoder类型: {type(model_compat.decoder).__name__}")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    print("\n" + "="*80)
    print("✓ 所有decoder模式测试通过")
    print("="*80)
    return True


def test_separate_configs():
    """测试encoder/decoder分离配置"""
    print("\n" + "="*80)
    print("测试阶段二功能：Encoder/Decoder 分离配置")
    print("="*80)

    # 通用参数
    input_size = (32, 64, 64)
    input_channels = 4
    num_classes = 3
    n_stages = 5
    features_per_stage = [32, 64, 128, 256, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [2, 2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2, 2]

    # 基础配置
    base_config = {
        'input_size': input_size,
        'input_channels': input_channels,
        'n_stages': n_stages,
        'features_per_stage': features_per_stage,
        'conv_op': conv_op,
        'kernel_sizes': kernel_sizes,
        'strides': strides,
        'n_conv_per_stage': n_conv_per_stage,
        'num_classes': num_classes,
        'n_conv_per_stage_decoder': n_conv_per_stage_decoder,
        'conv_bias': True,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': False,
        'use_rthd': True,
        'rthd_stages': [0, 1, 2],
        'decoder_rthd_mode': 'partial',
        'rthd_stages_decoder': [0, 1],
    }

    # 测试1：只使用统一rthd_config
    print("\n" + "-"*80)
    print("测试1: 只使用统一rthd_config")
    print("-"*80)
    try:
        unified_config = {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
        }
        model = UMambaEnc_RTHD(
            **base_config,
            rthd_config=unified_config,
        )
        print(f"✓ 模型构造成功")
        print(f"  - Encoder和Decoder共用配置")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    # 测试2：使用分离的encoder和decoder配置
    print("\n" + "-"*80)
    print("测试2: 使用分离的rthd_config_encoder和rthd_config_decoder")
    print("-"*80)
    try:
        encoder_config = {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
        }
        decoder_config = {
            'view_mode': 'tri',
            'share_weights': False,
            'scan_mode': 'standard',
        }
        model = UMambaEnc_RTHD(
            **base_config,
            rthd_config_encoder=encoder_config,
            rthd_config_decoder=decoder_config,
        )
        print(f"✓ 模型构造成功")
        print(f"  - Encoder使用专用配置")
        print(f"  - Decoder使用专用配置")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    # 测试3：混合使用（encoder专用，decoder回退到统一配置）
    print("\n" + "-"*80)
    print("测试3: 混合配置 - encoder专用配置 + 统一配置作为decoder回退")
    print("-"*80)
    try:
        unified_config = {
            'view_mode': 'tri',
            'share_weights': True,
        }
        encoder_config = {
            'view_mode': 'tri',
            'share_weights': True,
            'scan_mode': 'omni',
        }
        model = UMambaEnc_RTHD(
            **base_config,
            rthd_config=unified_config,
            rthd_config_encoder=encoder_config,
        )
        print(f"✓ 模型构造成功")
        print(f"  - Encoder使用专用配置")
        print(f"  - Decoder回退到统一配置")
    except Exception as e:
        print(f"✗ 模型构造失败: {e}")
        return False

    print("\n" + "="*80)
    print("✓ 所有配置分离测试通过")
    print("="*80)
    return True


def test_forward_pass():
    """测试前向传播（最小smoke test）"""
    print("\n" + "="*80)
    print("测试前向传播")
    print("="*80)

    # 小尺寸输入用于快速测试
    batch_size = 1
    input_channels = 4
    input_size = (32, 64, 64)
    num_classes = 3
    n_stages = 5
    features_per_stage = [32, 64, 128, 256, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    # 创建输入
    x = torch.randn(batch_size, input_channels, *input_size)
    print(f"输入shape: {x.shape}")

    # 测试partial模式前向传播
    print("\n" + "-"*80)
    print("测试partial模式前向传播")
    print("-"*80)
    try:
        model = UMambaEnc_RTHD(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=[2, 2, 2, 2, 2],
            num_classes=num_classes,
            n_conv_per_stage_decoder=[2, 2, 2, 2],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
            use_rthd=True,
            rthd_stages=[0, 1, 2],
            decoder_rthd_mode='partial',
            rthd_stages_decoder=[0, 1],
        )
        model.eval()

        with torch.no_grad():
            output = model(x)

        print(f"✓ 前向传播成功")
        print(f"  - 输出shape: {output.shape}")
        print(f"  - 期望shape: ({batch_size}, {num_classes}, {input_size[0]}, {input_size[1]}, {input_size[2]})")

        # 验证输出shape
        expected_shape = (batch_size, num_classes, input_size[0], input_size[1], input_size[2])
        if output.shape == expected_shape:
            print(f"✓ 输出shape正确")
        else:
            print(f"✗ 输出shape不匹配")
            return False

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✓ 前向传播测试通过")
    print("="*80)
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("阶段二第一步实现验证测试")
    print("="*80)

    success = True

    # 测试1：decoder模式
    if not test_decoder_modes():
        success = False
        print("\n✗ Decoder模式测试失败")

    # 测试2：分离配置
    if not test_separate_configs():
        success = False
        print("\n✗ 分离配置测试失败")

    # 测试3：前向传播
    if not test_forward_pass():
        success = False
        print("\n✗ 前向传播测试失败")

    # 总结
    print("\n" + "="*80)
    if success:
        print("✓ 所有测试通过！")
        print("\n实现功能总结:")
        print("  1. ✓ Partial Decoder RTHD (none/partial/full三种模式)")
        print("  2. ✓ rthd_stages_decoder参数支持")
        print("  3. ✓ Encoder/Decoder配置分离")
        print("  4. ✓ 向后兼容性保持")
        print("  5. ✓ 前向传播正常")
    else:
        print("✗ 部分测试失败，请检查上述错误信息")
    print("="*80)

    sys.exit(0 if success else 1)
