import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM import nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM


class nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM_150epochs(nnUNetTrainerUMambaEncRTHD_PartialDecoderETSM):
    """150-epoch wrapper for partial decoder ETSM (memory-efficient version)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150

    def configure_optimizers(self):
        return self.configure_optimizer()
