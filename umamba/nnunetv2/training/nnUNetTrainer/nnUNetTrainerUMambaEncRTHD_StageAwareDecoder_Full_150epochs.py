import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full import nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full_150epochs(nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full):
    """
    nnUNet Trainer for UMambaEnc RTHD with Full Stage-Aware Decoder Enhancement
    训练配置: 150 epochs (第一轮快速筛选)

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        """Fix: configure_optimizers calls configure_optimizer in base class"""
        return self.configure_optimizer()
