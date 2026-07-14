import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration import nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration


class nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration_150epochs(nnUNetTrainerUMambaEncRTHD_EncoderOnly_SkipCalibration):
    """Encoder-only ETSM + Skip Calibration ablation trained for 150 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        return self.configure_optimizer()
