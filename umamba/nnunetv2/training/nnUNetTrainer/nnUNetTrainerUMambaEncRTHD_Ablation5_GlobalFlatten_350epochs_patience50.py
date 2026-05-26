import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten import nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten


class nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten_350epochs_patience50(nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten):
    """
    消融实验 #5: 全局平铺版
    训练配置: 350 epochs, patience 50 (早停)

    配置:
    - view_mode='tri' (三视图)
    - share_weights=True (参数共享)
    - scan_mode='omni' (全向扫描)
    - use_local_window=False (全局平铺，不使用局部滑窗)

    注意: 这是当前 nnUNetTrainerUMambaEncRTHD 的默认配置

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation5_GlobalFlatten_350epochs_patience50
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 350
        self.patience = 50
        self.patience_counter = 0
        self._early_stop = False

    def on_epoch_end(self):
        old_best_ema = self._best_ema
        super().on_epoch_end()

        if old_best_ema is not None and self._best_ema == old_best_ema:
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
