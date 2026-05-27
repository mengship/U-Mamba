"""
测试解码器 RTHD 集成

验证 UNetResDecoder_RTHD 是否正确工作
"""

import torch
import sys
sys.path.insert(0, '/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/U-Mamba/umamba')

from nnunetv2.nets.UMambaEnc_RTHD import UMambaEnc_RTHD
from torch import nn

def test_decoder_rthd():
    """测试完全对称的 RTHD 解码器"""

    print("=" * 80)
    print("测试 1: 编码器+解码器都使用 RTHD（完全对称）")
    print("=" * 80)

    # RTHD 配置
    rthd_config = {
        'view_mode': 'tri',
        'share_weights': True,
        'scan_mode': 'omni',
        'use_local_window': True,
        'window_size': 8,
    }

    # 创建模型
    model = UMambaEnc_RTHD(
        input_size=(128, 128, 112),
        input_channels=4,
        n_stages=5,
        features_per_stage=[32, 64, 128, 256, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2],
        num_classes=3,
        n_conv_per_stage_decoder=[2, 2, 2, 2],
        deep_supervision=True,
        use_rthd=True,
        rthd_stages=[0, 1, 2, 3, 4],
        rthd_config=rthd_config,
        use_rthd_decoder=True,  # 使用 RTHD 解码器
    )

    print(f"\n模型创建成功！")
    print(f"编码器类型: {type(model.encoder).__name__}")
    print(f"解码器类型: {type(model.decoder).__name__}")

    # 测试前向传播
    x = torch.randn(1, 4, 128, 128, 112)
    print(f"\n输入形状: {x.shape}")

    try:
        with torch.no_grad():
            output = model(x)

        if isinstance(output, list):
            print(f"\n输出（深度监督）:")
            for i, out in enumerate(output):
                print(f"  Level {i}: {out.shape}")
        else:
            print(f"\n输出形状: {output.shape}")

        print("\n✅ 测试通过！编码器+解码器 RTHD 工作正常")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("测试 2: 编码器 RTHD + 原始卷积解码器（对比）")
    print("=" * 80)

    # 创建对比模型
    model_baseline = UMambaEnc_RTHD(
        input_size=(128, 128, 112),
        input_channels=4,
        n_stages=5,
        features_per_stage=[32, 64, 128, 256, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2],
        num_classes=3,
        n_conv_per_stage_decoder=[2, 2, 2, 2],
        deep_supervision=True,
        use_rthd=True,
        rthd_stages=[0, 1, 2, 3, 4],
        rthd_config=rthd_config,
        use_rthd_decoder=False,  # 不使用 RTHD 解码器
    )

    print(f"\n基线模型创建成功！")
    print(f"编码器类型: {type(model_baseline.encoder).__name__}")
    print(f"解码器类型: {type(model_baseline.decoder).__name__}")

    try:
        with torch.no_grad():
            output_baseline = model_baseline(x)

        if isinstance(output_baseline, list):
            print(f"\n输出（深度监督）:")
            for i, out in enumerate(output_baseline):
                print(f"  Level {i}: {out.shape}")
        else:
            print(f"\n输出形状: {output_baseline.shape}")

        print("\n✅ 基线模型测试通过！")

    except Exception as e:
        print(f"\n❌ 基线模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 参数量对比
    print("\n" + "=" * 80)
    print("参数量对比")
    print("=" * 80)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_full = count_parameters(model)
    params_baseline = count_parameters(model_baseline)

    print(f"完全 RTHD（编码器+解码器）: {params_full:,} 参数")
    print(f"部分 RTHD（仅编码器）:     {params_baseline:,} 参数")
    print(f"增加比例: {(params_full / params_baseline - 1) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！解码器 RTHD 集成成功")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_decoder_rthd()
    sys.exit(0 if success else 1)
