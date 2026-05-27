from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_Ablation4_StandardScan(nnUNetTrainer):
    """
    消融实验 #4: 常规扫描版

    配置:
    - view_mode='tri' (三视图)
    - share_weights=True (参数共享)
    - scan_mode='standard' (标准双向扫描，非全向)
    - use_local_window=True (局部滑窗)

    目的: 验证全向扫描的必要性
    预期: 相比全向扫描性能略有下降，证明多方向扫描的价值

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation4_StandardScan
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
                # 消融实验 #4 配置
                rthd_config={
                    'view_mode': 'tri',
                    'share_weights': True,
                    'scan_mode': 'standard',  # 标准扫描
                    'use_local_window': True,
                    'window_size': 8,
                },

                use_rthd_decoder=True,  # 解码器也使用RTHD（完全对称）

                )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("消融实验 #4: 常规扫描版")
        print("配置: view_mode='tri', share_weights=True, scan_mode='standard', use_local_window=True")
        print("=" * 80)

        return model
