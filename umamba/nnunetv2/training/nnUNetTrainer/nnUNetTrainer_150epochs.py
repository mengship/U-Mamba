import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_150epochs(nnUNetTrainer):
    """
    Official nnU-Net baseline trainer with 150 epochs for fair comparison.

    This trainer inherits from the original nnUNetTrainer without any modifications
    except for the number of training epochs. It maintains the same network architecture,
    optimizer, learning rate schedule, data augmentation, loss function, and deep supervision
    settings as the vanilla nnU-Net.

    Usage:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainer_150epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        """Wrapper method to call configure_optimizer (singular) from base class."""
        return self.configure_optimizer()
