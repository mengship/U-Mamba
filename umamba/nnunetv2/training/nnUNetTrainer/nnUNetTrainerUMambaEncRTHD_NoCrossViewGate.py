from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_NoCrossViewGate(nnUNetTrainer):
    """
    C3 ablation: final ETSM pipeline without cross-view interaction gate.

    The encoder, stage-aware decoder, and semantic skip calibration are retained,
    but cross_view_interaction is disabled in both encoder and decoder ETSM blocks.
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) != 3:
            raise NotImplementedError("ETSM currently only supports 3D models")

        common_config = {
            "view_mode": "tri",
            "share_weights": True,
            "scan_mode": "omni",
            "window_size": 8,
            "reconstruction_mode": "gated",
            "cross_view_interaction": False,
        }
        encoder_config = {**common_config, "use_local_window": True}
        decoder_config = {**common_config, "use_local_window": False}

        return get_umamba_enc_rthd_3d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            rthd_config_encoder=encoder_config,
            rthd_config_decoder=decoder_config,
            use_rthd_decoder=True,
            decoder_rthd_mode="partial",
            rthd_stages_decoder=[0, 1],
            use_skip_fusion_gate=True,
            skip_gate_stages=[0, 1],
            skip_gate_reduction=4,
            skip_gate_type="semantic",
            use_boundary_attention_head=False,
            use_frequency_refinement=False,
        )
