import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNETR import nnUNetTrainerUNETR


class nnUNetTrainerUNETR_150epochs(nnUNetTrainerUNETR):
    """UNETR trained for 150 epochs with a fixed, audited sample budget."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 150
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        expected_patch_size = (128, 128, 128)
        actual_patch_size = tuple(int(i) for i in self.configuration_manager.patch_size)
        if self.is_ddp:
            raise RuntimeError("This audited UNETR trainer currently supports single-GPU training only.")
        if actual_patch_size != expected_patch_size:
            raise RuntimeError(
                f"Expected patch_size={expected_patch_size}, got {actual_patch_size}. "
                "Refusing to start a non-comparable experiment."
            )
        if self.configuration_manager.batch_size != 2 or self.batch_size != 2:
            raise RuntimeError(
                f"Expected plans/actual batch size 2, got "
                f"{self.configuration_manager.batch_size}/{self.batch_size}. "
                "Refusing to start a non-comparable experiment."
            )

    def initialize(self):
        super().initialize()
        self.print_to_log_file("\n" + "=" * 80)
        self.print_to_log_file("Audited UNETR 150-epoch configuration")
        self.print_to_log_file("=" * 80)
        self.print_to_log_file(f"Patch size: {self.configuration_manager.patch_size}")
        self.print_to_log_file(f"Actual batch size: {self.batch_size}")
        self.print_to_log_file(f"Epochs: {self.num_epochs}")
        self.print_to_log_file(f"Training iterations per epoch: {self.num_iterations_per_epoch}")
        self.print_to_log_file(
            f"Training samples per epoch: {self.batch_size * self.num_iterations_per_epoch}"
        )
        self.print_to_log_file("Gradient accumulation: disabled")
        self.print_to_log_file("Deep supervision: disabled (original UNETR architecture)")
        self.print_to_log_file("Optimizer: AdamW, lr=1e-4, weight_decay=0.01")
        self.print_to_log_file("=" * 80 + "\n")
