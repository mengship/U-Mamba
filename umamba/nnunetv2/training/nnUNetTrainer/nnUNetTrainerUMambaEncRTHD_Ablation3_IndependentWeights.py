from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights(nnUNetTrainer):
    """
    消融实验 #3: 独立参数版

    配置:
    - view_mode='tri' (三视图)
    - share_weights=False (三个视图使用独立的Mamba参数)
    - scan_mode='omni' (全向扫描)
    - use_local_window=True (局部滑窗)

    目的: 验证参数共享的有效性
    预期: 参数量增加3倍，但性能提升有限，证明参数共享的高效性

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights
    """
    def configure_optimizers(self):
        """修复：添加configure_optimizers方法以兼容基类调用"""
        return self.configure_optimizer()


    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            # 使用自定义配置的RTHD版本
            model = get_umamba_enc_rthd_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
                # 消融实验 #3 配置
                rthd_config={
                    'view_mode': 'tri',
                    'share_weights': False,  # 独立参数
                    'scan_mode': 'omni',
                    'use_local_window': True,
                    'window_size': 8,
                },

                use_rthd_decoder=True,  # 解码器也使用RTHD（完全对称）

                )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("消融实验 #3: 独立参数版")
        print("配置: view_mode='tri', share_weights=False, scan_mode='omni', use_local_window=True")
        print("=" * 80)

        return model
