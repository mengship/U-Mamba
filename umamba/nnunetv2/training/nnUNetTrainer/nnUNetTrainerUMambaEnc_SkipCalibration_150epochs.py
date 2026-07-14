import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc_SkipCalibration import nnUNetTrainerUMambaEnc_SkipCalibration


class nnUNetTrainerUMambaEnc_SkipCalibration_150epochs(nnUNetTrainerUMambaEnc_SkipCalibration):
    """U-Mamba + Skip Calibration ablation trained for 150 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        return self.configure_optimizer()
