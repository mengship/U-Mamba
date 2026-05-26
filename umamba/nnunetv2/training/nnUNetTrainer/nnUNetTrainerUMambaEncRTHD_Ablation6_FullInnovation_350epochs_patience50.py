import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation import nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation


class nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation_350epochs_patience50(nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation):
    """
    消融实验 #6: 完整创新版（最佳配置）
    训练配置: 350 epochs, patience 50 (早停)

    配置:
    - view_mode='tri' (三视图)
    - share_weights=True (参数共享)
    - scan_mode='omni' (全向扫描)
    - use_local_window=True (局部滑窗)

    这是论文中提出的完整RTHD方法，集成所有创新点

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation6_FullInnovation_350epochs_patience50
    """
    def configure_optimizers(self):
        """修复：添加configure_optimizers方法以兼容基类调用"""
        return self.configure_optimizer()


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
