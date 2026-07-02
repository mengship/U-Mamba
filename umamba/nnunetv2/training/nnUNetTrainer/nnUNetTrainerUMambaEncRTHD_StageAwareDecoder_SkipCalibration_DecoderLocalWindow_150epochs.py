import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow import nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow


class nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow_150epochs(nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow):
    """
    Decoder LocalWindow Ablation (150 epochs)

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_DecoderLocalWindow_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        return self.configure_optimizer()
