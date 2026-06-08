from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder(nnUNetTrainer):
    """
    nnUNet Trainer for UMambaEnc RTHD with Stage-Aware Decoder Recovery Strategy (第二版)

    第二版增强策略：
    1. Stage-aware RTHD Decoder: 阶段感知的解码器RTHD部署（partial模式，D4/D3使用RTHD）
    2. Semantic-guided Skip Fusion Gate: 语义引导的跳跃连接融合门控
    3. Boundary-aware Segmentation Attention: 边界感知注意力头
    4. High-low Frequency Structure Recovery: 高低频结构恢复（可选消融，默认不启用）

    核心思想：
    - 编码器：全局建模（RTHD全部stage）
    - 解码器：阶段感知结构恢复（低分辨率D4/D3用RTHD，高分辨率D2/D1用卷积）
    - 跳跃连接：语义引导门控（D4/D3 skip gate）
    - 边界增强：最终输出前boundary attention

    论文描述：
    "本文在解码阶段设计阶段感知的结构恢复策略，通过低分辨率RTHD refinement建模全局结构、
    语义引导跳连融合恢复局部细节，并结合边界注意力增强肿瘤轮廓质量。"

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # 使用第二版增强配置
            model = get_umamba_enc_rthd_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
                # 编码器RTHD配置（第一版增强）
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
                # 解码器RTHD配置（第一版增强）
                rthd_config_decoder={
                    "view_mode": "tri",
                    "share_weights": True,
                    "scan_mode": "omni",
                    "use_local_window": False,  # 解码器不使用局部窗口
                    "window_size": 8,
                    "reconstruction_mode": "gated",
                    "cross_view_interaction": True,
                    "interaction_mode": "post",
                    "interaction_type": "gate",
                },
                # 阶段感知RTHD部署：partial模式，只在D4/D3使用RTHD
                use_rthd_decoder=True,
                decoder_rthd_mode="partial",
                rthd_stages_decoder=[0, 1],  # D4和D3使用RTHD，D2/D1保持卷积
                # 第二版增强参数
                use_skip_fusion_gate=True,
                skip_gate_stages=[0, 1],  # D4/D3使用skip gate
                skip_gate_reduction=4,
                use_boundary_attention_head=True,  # 最终输出前使用边界注意力
                use_frequency_refinement=False,  # B方案：频率恢复仅保留为可选消融
                frequency_refinement_stages=None,
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("UMambaEnc_RTHD with Stage-Aware Decoder Recovery Strategy (第二版)")
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
        print("第二版增强模块:")
        print("  ✓ Semantic-guided Skip Fusion Gate (stages [0, 1])")
        print("  ✓ Boundary-aware Attention Head (final output)")
        print("  - High-low Frequency Refinement disabled by default (ablation only)")
        print()
        print("模型名称: RTHD-StageAwareDecoder")
        print("=" * 80)

        return model
