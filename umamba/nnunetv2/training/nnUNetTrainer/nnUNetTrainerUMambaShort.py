import torch
import numpy as np

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


class nnUNetTrainerUMambaBot_350epochs_patience50(nnUNetTrainerUMambaBot):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 350
        self.patience = 50
        self.patience_counter = 0
        self._early_stop = False

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._best_ema is not None and \
           self.logger.my_fantastic_logging['ema_fg_dice'][-1] < self._best_ema:
            self.patience_counter += 1
            self.print_to_log_file(
                f'No improvement in EMA dice. Patience: {self.patience_counter}/{self.patience}')
            if self.patience_counter >= self.patience:
                self.print_to_log_file(
                    f'Early stopping triggered at epoch {self.current_epoch}')
                self._early_stop = True
        else:
            self.patience_counter = 0

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            if self._early_stop:
                break

        self.on_train_end()


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