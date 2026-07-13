"""
SwinUNETR Trainer for BraTS2020 with 150 epochs and gradient accumulation.

This trainer implements:
- 150 epochs training
- AdamW optimizer with CosineAnnealingLR
- Gradient accumulation (batch_size=1, accumulate 2 steps for effective batch_size=2)
- Gradient clipping (threshold=12)
- AMP for CUDA devices
- No deep supervision
- Compatible with BraTS region-based labels

Usage:
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerSwinUNETR_150epochs
"""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

import torch
from torch import nn, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

from monai.networks.nets import SwinUNETR
from typing import List


class nnUNetTrainerSwinUNETR_150epochs(nnUNetTrainerNoDeepSupervision):
    """
    SwinUNETR trainer with 150 epochs and gradient accumulation.

    Configuration:
    - num_epochs: 150
    - optimizer: AdamW (lr=8e-4, weight_decay=0.01, eps=1e-5)
    - scheduler: CosineAnnealingLR (eta_min=1e-6)
    - batch_size: 1 (real) with 2-step gradient accumulation (effective batch_size=2)
    - gradient clipping: 12
    - AMP: enabled for CUDA, disabled for CPU
    - num_iterations_per_epoch: 500 (to maintain same samples/epoch as baseline batch_size=2)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Adjust patch size to be divisible by 32 (2^5 for 5 pooling layers)
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2**5) < 1 or ((original_patch_size[i] / 2**5) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2**5 + 0.5) * 2**5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

        # Training configuration
        self.num_epochs = 150
        self.initial_lr = 8e-4
        self.weight_decay = 0.01

        # Gradient accumulation configuration
        self.gradient_accumulation_steps = 2
        self._grad_accum_counter = 0

        # Adjust iterations to maintain same samples per epoch
        # Original: batch_size=2, 250 iters = 500 samples, 250 optimizer steps
        # New: batch_size=1, need 500 iters = 500 samples, 250 optimizer steps (with 2-step accumulation)
        plans_batch_size = self.configuration_manager.batch_size
        if plans_batch_size != 2:
            self.print_to_log_file(
                f"WARNING: Expected plans batch_size=2, but got {plans_batch_size}. "
                f"Effective batch size may differ from baseline!"
            )

        self.num_iterations_per_epoch = 500  # 500 samples = 250 optimizer updates with 2-step accumulation
        self.num_val_iterations_per_epoch = 50  # Keep validation samples consistent

        # GradScaler for AMP (only for CUDA)
        if self.device.type == 'cuda':
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = None

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        """
        Build SwinUNETR network with specified configuration.

        Note: enable_deep_supervision is ignored as SwinUNETR doesn't support it.
        """
        label_manager = plans_manager.get_label_manager(dataset_json)
        img_size = configuration_manager.patch_size
        spatial_dims = len(img_size)

        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            img_size=img_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=True,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False
        )

        return model

    def initialize(self):
        """Override to log configuration details."""
        super().initialize()

        # Log detailed configuration
        self.print_to_log_file("\n" + "="*80)
        self.print_to_log_file("SwinUNETR 150-epoch Training Configuration")
        self.print_to_log_file("="*80)

        # MONAI and PyTorch versions
        try:
            import monai
            self.print_to_log_file(f"MONAI version: {monai.__version__}")
        except:
            self.print_to_log_file("MONAI version: Unable to detect")
        self.print_to_log_file(f"PyTorch version: {torch.__version__}")

        # Network configuration
        self.print_to_log_file(f"\nNetwork Configuration:")
        self.print_to_log_file(f"  Patch size: {self.configuration_manager.patch_size}")
        self.print_to_log_file(f"  Spatial dims: {len(self.configuration_manager.patch_size)}")
        self.print_to_log_file(f"  Input channels: {self.num_input_channels}")
        self.print_to_log_file(f"  Output channels: {self.label_manager.num_segmentation_heads}")
        self.print_to_log_file(f"  Feature size: 48")
        self.print_to_log_file(f"  Depths: (2, 2, 2, 2)")
        self.print_to_log_file(f"  Num heads: (3, 6, 12, 24)")
        self.print_to_log_file(f"  Use checkpoint: True")

        # Training configuration
        self.print_to_log_file(f"\nTraining Configuration:")
        self.print_to_log_file(f"  Num epochs: {self.num_epochs}")
        self.print_to_log_file(f"  Plans batch size: {self.configuration_manager.batch_size}")
        self.print_to_log_file(f"  Actual batch size: 1")
        self.print_to_log_file(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        self.print_to_log_file(f"  Effective batch size: {self.gradient_accumulation_steps}")
        self.print_to_log_file(f"  Num iterations per epoch: {self.num_iterations_per_epoch}")
        self.print_to_log_file(f"  Optimizer steps per epoch: {self.num_iterations_per_epoch // self.gradient_accumulation_steps}")
        self.print_to_log_file(f"  Total samples per epoch: ~{self.num_iterations_per_epoch}")
        self.print_to_log_file(f"  Num validation iterations: {self.num_val_iterations_per_epoch}")

        # Optimizer configuration
        self.print_to_log_file(f"\nOptimizer Configuration:")
        self.print_to_log_file(f"  Optimizer: AdamW")
        self.print_to_log_file(f"  Initial learning rate: {self.initial_lr}")
        self.print_to_log_file(f"  Weight decay: {self.weight_decay}")
        self.print_to_log_file(f"  Epsilon: 1e-5")
        self.print_to_log_file(f"  Scheduler: CosineAnnealingLR")
        self.print_to_log_file(f"  Eta min: 1e-6")
        self.print_to_log_file(f"  Gradient clipping: 12")

        # AMP configuration
        self.print_to_log_file(f"\nAMP Configuration:")
        self.print_to_log_file(f"  Device type: {self.device.type}")
        self.print_to_log_file(f"  AMP enabled: {self.grad_scaler is not None}")

        self.print_to_log_file("="*80 + "\n")

    def train_step(self, batch: dict) -> dict:
        """
        Training step with gradient accumulation.

        Implements proper 2-step gradient accumulation:
        - Only zero_grad at the start of accumulation window
        - Scale loss by 1/accumulation_steps before backward
        - Only step optimizer after accumulation_steps
        - Return unscaled loss for logging
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Only zero gradients at the start of accumulation window
        if self._grad_accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Forward pass with autocast for CUDA
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # Compute loss and scale by accumulation steps
            loss = self.loss(output, target)
            loss_scaled = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.grad_scaler is not None:
            # AMP path
            self.grad_scaler.scale(loss_scaled).backward()
        else:
            # Non-AMP path
            loss_scaled.backward()

        # Increment accumulation counter
        self._grad_accum_counter += 1

        # Perform optimizer step after accumulation_steps
        if self._grad_accum_counter >= self.gradient_accumulation_steps:
            if self.grad_scaler is not None:
                # AMP: unscale before clipping
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                # Non-AMP
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

            # Reset accumulation counter
            self._grad_accum_counter = 0

        # Return original unscaled loss for logging
        return {'loss': loss.detach().cpu().numpy()}

    def on_train_epoch_start(self):
        """Reset gradient accumulation counter at epoch start."""
        super().on_train_epoch_start()
        self._grad_accum_counter = 0

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        Handle end of training epoch.

        If there are residual gradients (epoch not divisible by accumulation_steps),
        they are discarded to avoid contaminating the next epoch.
        """
        # Warn if there are residual gradients
        if self._grad_accum_counter > 0:
            self.print_to_log_file(
                f"WARNING: Epoch ended with {self._grad_accum_counter} residual gradient(s). "
                f"These gradients are discarded."
            )
            # Reset counter to ensure clean start next epoch
            self._grad_accum_counter = 0

        super().on_train_epoch_end(train_outputs)

    def validation_step(self, batch: dict) -> dict:
        """
        Validation step with proper no_grad and autocast handling.

        Note: No optimizer.zero_grad() in validation as it's wasteful and unnecessary.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Forward pass with autocast (already in torch.no_grad context from run_training)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # Online evaluation
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            # Remove background channel
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def configure_optimizers(self):
        """Configure AdamW optimizer and CosineAnnealingLR scheduler."""
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )

        self.print_to_log_file(f"Using optimizer: {optimizer}")
        self.print_to_log_file(f"Using scheduler: {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """SwinUNETR doesn't support deep supervision."""
        pass
