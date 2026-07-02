from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow(nnUNetTrainer):
    """
    Decoder LocalWindow Ablation: 基于StageAwareDecoder_SkipCalibration，仅修改decoder use_local_window

    配置变化（相对于StageAwareDecoder_SkipCalibration）：
    - Decoder: use_local_window = True （唯一变化）

    其他配置完全相同：
    - Encoder use_local_window=True
    - Encoder cross_view_interaction=True
    - Decoder cross_view_interaction=True
    - Decoder mode: partial, stages [0, 1]
    - Skip Fusion Gate: enabled (stages [0, 1], reduction=4)
    - Boundary attention: disabled
    - Frequency refinement: disabled

    实验目的：
    验证解码器局部窗口对显存和性能的影响

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # 完全复制StageAwareDecoder_SkipCalibration，仅修改decoder use_local_window
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
                    "use_local_window": True,  # 修改点：启用局部窗口
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
        print("UMambaEnc_RTHD: Decoder LocalWindow Ablation")
        print("=" * 80)
        print("配置差异（相对于StageAwareDecoder_SkipCalibration）:")
        print("  - Decoder use_local_window: False → True")
        print()
        print("编码器配置:")
        print(f"  - use_local_window: True")
        print(f"  - cross_view_interaction: True")
        print("  - RTHD stages: [0, 1, 2, 3, 4]")
        print()
        print("解码器配置:")
        print(f"  - use_local_window: True ← 消融点")
        print(f"  - cross_view_interaction: True")
        print(f"  - decoder_rthd_mode: partial")
        print(f"  - rthd_stages_decoder: [0, 1]")
        print()
        print("增强模块:")
        print(f"  - use_skip_fusion_gate: True")
        print(f"  - skip_gate_stages: [0, 1]")
        print(f"  - skip_gate_reduction: 4")
        print(f"  - skip_gate_type: semantic (默认值)")
        print("=" * 80)

        return model
