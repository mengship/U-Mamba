import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_FullDecoderETSM import nnUNetTrainerUMambaEncRTHD_FullDecoderETSM


class nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs(nnUNetTrainerUMambaEncRTHD_FullDecoderETSM):
    """150-epoch wrapper for C2 full decoder ETSM ablation."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        return self.configure_optimizer()
