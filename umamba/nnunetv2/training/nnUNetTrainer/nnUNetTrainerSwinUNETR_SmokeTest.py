"""
SwinUNETR SmokeTest Trainer - Quick validation before full training.

This trainer runs minimal iterations to verify:
- Network initialization
- Forward/backward passes
- AMP compatibility
- Gradient accumulation
- Checkpoint saving
- No OOM or NaN issues

Usage:
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerSwinUNETR_SmokeTest
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR_150epochs import nnUNetTrainerSwinUNETR_150epochs
import torch


class nnUNetTrainerSwinUNETR_SmokeTest(nnUNetTrainerSwinUNETR_150epochs):
    """
    SmokeTest version of SwinUNETR trainer.

    Runs only 1 epoch with minimal iterations to verify:
    - Model can be instantiated
    - 128x128x128 patches can be processed
    - AMP and gradient accumulation work
    - No memory errors
    - Checkpoints can be saved

    Configuration:
    - num_epochs: 1
    - num_iterations_per_epoch: 4 (2 optimizer steps with 2-step accumulation)
    - num_val_iterations_per_epoch: 1
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Minimal configuration for smoke test
        self.num_epochs = 1
        self.num_iterations_per_epoch = 4  # 4 batches = 2 optimizer steps with 2-step accumulation
        self.num_val_iterations_per_epoch = 1

    def initialize(self):
        """Override to add SmokeTest banner."""
        super().initialize()

        self.print_to_log_file("\n" + "!"*80)
        self.print_to_log_file("!!! SMOKE TEST MODE !!!")
        self.print_to_log_file("!!! This is NOT a full training run !!!")
        self.print_to_log_file("!!! Only 1 epoch with 4 training iterations and 1 validation iteration !!!")
        self.print_to_log_file("!!! Purpose: Verify model initialization, forward/backward, and no OOM !!!")
        self.print_to_log_file("!"*80 + "\n")
