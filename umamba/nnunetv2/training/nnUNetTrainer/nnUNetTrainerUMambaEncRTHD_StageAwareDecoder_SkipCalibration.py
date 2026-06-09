from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration(nnUNetTrainer):
    """
    nnUNet Trainer for UMambaEnc RTHD with Stage-Aware Decoder + Skip Calibration

    配置：
    - Encoder: RTHD enabled (tri-view, omni-scan, gated reconstruction)
    - Decoder: Stage-aware RTHD deployment (partial mode)
      - Low-resolution stages (D4/D3): RTHD
      - High-resolution stages (D2/D1): Standard convolution
    - Skip Fusion Gate: enabled (stages [0, 1])
    - No boundary attention (留给Full Model)
    - No frequency refinement

    实验目的：
    验证语义引导的跳跃连接特征校准（Skip Calibration）的有效性

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # Stage-aware decoder + Skip calibration
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
                # Skip Calibration: 启用语义引导跳跃连接特征校准
                use_skip_fusion_gate=True,
                skip_gate_stages=[0, 1],
                skip_gate_reduction=4,
                # 其他增强模块：暂不启用
                use_boundary_attention_head=False,
                use_frequency_refinement=False,
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("UMambaEnc_RTHD: Stage-Aware Decoder + Skip Calibration")
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
        print("增强模块:")
        print("  ✓ Semantic-guided Skip Fusion Gate (stages [0, 1], reduction=4)")
        print("  - Boundary attention: disabled (留给Full Model)")
        print("  - Frequency refinement: disabled")
        print("=" * 80)

        return model
