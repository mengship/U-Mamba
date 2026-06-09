from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full(nnUNetTrainer):
    """
    nnUNet Trainer for UMambaEnc RTHD with Full Stage-Aware Decoder Enhancement

    完整配置（Full Model）：
    - Encoder: RTHD enabled (tri-view, omni-scan, gated reconstruction)
    - Decoder: Stage-aware RTHD deployment (partial mode)
      - Low-resolution stages (D4/D3): RTHD
      - High-resolution stages (D2/D1): Standard convolution
    - Skip Fusion Gate: enabled (stages [0, 1])
    - Boundary Attention: enabled (final decoder stage)
    - Frequency Refinement: disabled (不作为主模型，仅保留为可选消融)

    实验目的：
    验证完整的阶段感知解码器恢复策略（Stage-Aware Decoder + Skip Calibration + Boundary Refinement）

    论文主模型名称: RTHD-StageAwareDecoder-Full

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # Full model: Stage-aware decoder + Skip calibration + Boundary refinement
            model = get_umamba_enc_rthd_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
                # 编码器RTHD配置
                rthd_config_encoder={
                    "view_mode": "tri",
                    "share_weights": True,
                    "scan_mode": "omni",
                    "use_local_window": True,
                    "window_size": 8,
                    "reconstruction_mode": "gated",
                    "cross_view_interaction": True,
                    "interaction_mode": "post",
                    "interaction_type": "gate",
                },
                # 解码器RTHD配置
                rthd_config_decoder={
                    "view_mode": "tri",
                    "share_weights": True,
                    "scan_mode": "omni",
                    "use_local_window": False,
                    "window_size": 8,
                    "reconstruction_mode": "gated",
                    "cross_view_interaction": True,
                    "interaction_mode": "post",
                    "interaction_type": "gate",
                },
                # 阶段感知RTHD部署：partial模式
                use_rthd_decoder=True,
                decoder_rthd_mode="partial",
                rthd_stages_decoder=[0, 1],
                # 完整增强配置
                use_skip_fusion_gate=True,
                skip_gate_stages=[0, 1],
                skip_gate_reduction=4,
                use_boundary_attention_head=True,
                use_frequency_refinement=False,  # 不作为主模型，仅保留为可选消融
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("UMambaEnc_RTHD: Full Stage-Aware Decoder Enhancement (主模型)")
        print("=" * 80)
        print("编码器配置:")
        print("  - RTHD stages: [0, 1, 2, 3, 4] (全局建模)")
        print("  - Gated reconstruction + Cross-view interaction")
        print("  - Local window: True (window_size=8)")
        print()
        print("解码器配置 (阶段感知部署):")
        print("  - Decoder mode: partial")
        print("  - RTHD stages: [0, 1] (D4/D3低分辨率使用RTHD)")
        print("  - Conv stages: [2, 3] (D2/D1高分辨率保持卷积)")
        print()
        print("完整增强模块:")
        print("  ✓ Semantic-guided Skip Fusion Gate (stages [0, 1], reduction=4)")
        print("  ✓ Boundary-aware Attention Head (final decoder stage)")
        print("  - Frequency refinement: disabled (仅保留为可选消融)")
        print()
        print("论文描述:")
        print("  阶段感知的解码器结构恢复策略，通过低分辨率RTHD refinement建模")
        print("  全局结构、语义引导跳连融合恢复局部细节，并结合边界注意力增强")
        print("  肿瘤轮廓质量。")
        print()
        print("模型名称: RTHD-StageAwareDecoder-Full")
        print("=" * 80)

        return model
