import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncNoAMP import nnUNetTrainerUMambaEncNoAMP


class nnUNetTrainerUMambaBot_50epochs(nnUNetTrainerUMambaBot):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnUNetTrainerUMambaBot_100epochs(nnUNetTrainerUMambaBot):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainerUMambaEncNoAMP_50epochs(nnUNetTrainerUMambaEncNoAMP):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnUNetTrainerUMambaEncNoAMP_100epochs(nnUNetTrainerUMambaEncNoAMP):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100