from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_AttentionSkipGate(nnUNetTrainer):
    """
    C4 ablation: Attention U-Net-style 0-1 skip gate.

    This keeps ETSM and stage-aware decoder settings identical to the final
    method, but replaces semantic residual skip calibration with a traditional
    one-way attention gate.
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) != 3:
            raise NotImplementedError("ETSM currently only supports 3D models")

        return get_umamba_enc_rthd_3d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
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
            use_rthd_decoder=True,
            decoder_rthd_mode="partial",
            rthd_stages_decoder=[0, 1],
            use_skip_fusion_gate=True,
            skip_gate_stages=[0, 1],
            skip_gate_reduction=4,
            skip_gate_type="attention",
            use_boundary_attention_head=False,
            use_frequency_refinement=False,
        )
