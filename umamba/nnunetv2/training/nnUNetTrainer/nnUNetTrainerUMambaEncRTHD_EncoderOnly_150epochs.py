import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_EncoderOnly import nnUNetTrainerUMambaEncRTHD_EncoderOnly


class nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs(nnUNetTrainerUMambaEncRTHD_EncoderOnly):
    """
    nnUNet Trainer for UMambaEnc with RTHD in Encoder Only
    训练配置: 150 epochs (第一轮快速筛选)

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150
