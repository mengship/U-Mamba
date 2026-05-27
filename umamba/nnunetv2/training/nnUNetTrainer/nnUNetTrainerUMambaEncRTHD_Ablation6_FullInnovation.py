from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_RTHD import get_umamba_enc_rthd_3d_from_plans


class nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation(nnUNetTrainer):
    """
    消融实验 #6: 完整创新版（最佳配置）

    配置:
    - view_mode='tri' (三视图)
    - share_weights=True (参数共享)
    - scan_mode='omni' (全向扫描)
    - use_local_window=True (局部滑窗)

    目的: 完整的RTHD架构，集成所有创新点
    预期: 最佳性能，证明各个创新点的协同作用

    这是论文中提出的完整RTHD方法

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation
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
                # 消融实验 #6 配置（完整版）
                rthd_config={
                    'view_mode': 'tri',
                    'share_weights': True,
                    'scan_mode': 'omni',
                    'use_local_window': True,  # 固定窗口
                    'window_size': 8,
                },
                use_rthd_decoder=True,  # 解码器也使用RTHD（完全对称）
            )
        else:
            raise NotImplementedError("RTHD currently only supports 3D models")

        print("=" * 80)
        print("消融实验 #6: 完整创新版（最佳配置）")
        print("配置: view_mode='tri', share_weights=True, scan_mode='omni', use_local_window=True")
        print("编码器: ✓ 三视图分解 ✓ 参数共享 ✓ 全向扫描 ✓ 固定窗口")
        print("解码器: ✓ RTHD（完全对称架构）")
        print("=" * 80)

        return model
