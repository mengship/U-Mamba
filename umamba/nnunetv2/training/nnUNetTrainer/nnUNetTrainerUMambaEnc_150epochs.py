import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc


class nnUNetTrainerUMambaEnc_150epochs(nnUNetTrainerUMambaEnc):
    """
    nnUNet Trainer for original UMambaEnc baseline.
    Training config: 150 epochs for first-round screening.

    Usage:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEnc_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        """Fix: configure_optimizers calls configure_optimizer in base class."""
        return self.configure_optimizer()
