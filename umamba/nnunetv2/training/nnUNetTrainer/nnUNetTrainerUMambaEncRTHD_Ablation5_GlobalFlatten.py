from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten(nnUNetTrainer):
    """
    消融实验 #5: 全局平铺版

    配置:
    - view_mode='tri' (三视图)
    - share_weights=True (参数共享)
    - scan_mode='omni' (全向扫描)
    - use_local_window=False (全局平铺，不使用局部滑窗)

    目的: 验证局部滑窗机制的必要性
    预期: 全局平铺在大特征图上可能性能略差，证明局部滑窗的价值

    注意: 这是当前 nnUNetTrainerUMambaEncRTHD 的默认配置

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten
    """

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
                # 消融实验 #5 配置
                rthd_config={
                    'view_mode': 'tri',
                    'share_weights': True,
                    'scan_mode': 'omni',
                    'use_local_window': False,  # 全局平铺
                    'window_size': 8,
                }
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("消融实验 #5: 全局平铺版")
        print("配置: view_mode='tri', share_weights=True, scan_mode='omni', use_local_window=False")
        print("=" * 80)

        return model
