from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_EncoderOnly(nnUNetTrainer):
    """
    nnUNet Trainer for UMambaEnc with RTHD in Encoder Only

    配置：
    - Encoder: RTHD enabled (tri-view, omni-scan, gated reconstruction)
    - Decoder: Standard convolution (no RTHD)
    - No skip fusion gate
    - No boundary attention
    - No frequency refinement

    实验目的：
    验证 Encoder RTHD 对全局建模的有效性（baseline for decoder enhancements）

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_EncoderOnly
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # Encoder RTHD only - decoder uses standard convolution
            model = get_umamba_enc_rthd_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
                # Encoder RTHD configuration
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
                # Decoder: no RTHD
                use_rthd_decoder=False,
                decoder_rthd_mode="none",
                # No second-version enhancements
                use_skip_fusion_gate=False,
                use_boundary_attention_head=False,
                use_frequency_refinement=False,
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("UMambaEnc_RTHD: Encoder Only Configuration")
        print("=" * 80)
        print("Encoder:")
        print("  - RTHD enabled (tri-view, omni-scan, gated reconstruction)")
        print("  - Local window: True (window_size=8)")
        print("  - Cross-view interaction: post-gate")
        print()
        print("Decoder:")
        print("  - Standard convolution (no RTHD)")
        print("  - No skip fusion gate")
        print("  - No boundary attention")
        print("  - No frequency refinement")
        print("=" * 80)

        return model
