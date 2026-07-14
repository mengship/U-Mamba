from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEnc_SkipCalibration(nnUNetTrainer):
    """U-Mamba baseline with semantic skip calibration only."""

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise NotImplementedError("Skip calibration ablation currently only supports 3D models")

        model = get_umamba_enc_rthd_3d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            rthd_stages_encoder=[],
            use_rthd_decoder=True,
            decoder_rthd_mode="partial",
            rthd_stages_decoder=[],
            use_skip_fusion_gate=True,
            skip_gate_stages=[0, 1],
            skip_gate_reduction=4,
            skip_gate_type="semantic",
            use_boundary_attention_head=False,
            use_frequency_refinement=False,
        )

        print("=" * 80)
        print("U-Mamba + Skip Calibration")
        print("=" * 80)
        print("Encoder: original U-Mamba MambaLayer at all stages")
        print("Decoder: original convolution blocks (no ETSM)")
        print("Skip calibration: semantic gate at decoder stages [0, 1]")
        print("Boundary attention: disabled")
        print("Frequency refinement: disabled")
        print("=" * 80)

        return model
