import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights import nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights


class nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights_350epochs_patience50(nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights):
    """
    消融实验 #3: 独立参数版
    训练配置: 350 epochs, patience 50 (早停)

    配置:
    - view_mode='tri' (三视图)
    - share_weights=False (三个视图使用独立的Mamba参数)
    - scan_mode='omni' (全向扫描)
    - use_local_window=True (局部滑窗)

    使用方法:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerUMambaEncRTHD_Ablation3_IndependentWeights_350epochs_patience50
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
        # 在调用 super() 之前保存旧的 best_ema
        old_best_ema = self._best_ema
        # 调用父类方法（会更新 self._best_ema）
        super().on_epoch_end()

        # 修复：使用数值比较而不是相等比较，避免浮点数精度问题
        if old_best_ema is None:
            # 第一个 epoch，初始化 best_ema，不算改进
            self.patience_counter = 0
        elif self._best_ema > old_best_ema:
            # 有改进（EMA dice 提升了），重置计数器
            self.patience_counter = 0
            self.print_to_log_file(f'EMA dice improved from {old_best_ema:.4f} to {self._best_ema:.4f}')
        else:
            # 没有改进
            self.patience_counter += 1
            self.print_to_log_file(
                f'No improvement in EMA dice (current: {self._best_ema:.4f}, best: {old_best_ema:.4f}). '
                f'Patience: {self.patience_counter}/{self.patience}')
            if self.patience_counter >= self.patience:
                # 注意：此时 current_epoch 已经被 super() 增加了 1
                self.print_to_log_file(
                    f'Early stopping triggered after epoch {self.current_epoch - 1}')
                self._early_stop = True

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
