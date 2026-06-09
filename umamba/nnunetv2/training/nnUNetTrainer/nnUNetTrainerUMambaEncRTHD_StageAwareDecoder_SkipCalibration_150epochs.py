import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration import nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs(nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration):
    """
    nnUNet Trainer for UMambaEnc RTHD with Stage-Aware Decoder + Skip Calibration
    训练配置: 150 epochs (第一轮快速筛选)

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        """Fix: configure_optimizers calls configure_optimizer in base class"""
        return self.configure_optimizer()
