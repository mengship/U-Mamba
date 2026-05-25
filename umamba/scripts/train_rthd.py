"""
RTHD模型训练脚本
用于在云GPU上训练UMambaEnc_RTHD模型进行脑肿瘤分割

使用方法:
    python train_rthd.py --dataset_path /path/to/dataset --output_dir /path/to/output

作者: 研究生脑肿瘤分割项目
日期: 2026-05-26
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnunetv2.nets.UMambaEnc_RTHD import UMambaEnc_RTHD


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train RTHD model for brain tumor segmentation')

    # 数据相关
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints and logs')

    # 模型配置
    parser.add_argument('--input_channels', type=int, default=4,
                        help='Number of input channels (e.g., 4 for BraTS)')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                        help='Patch size for training (D H W)')
    parser.add_argument('--use_rthd', action='store_true', default=True,
                        help='Use RTHD blocks')
    parser.add_argument('--rthd_stages', type=int, nargs='+', default=[0, 1, 2],
                        help='Which stages to use RTHD (default: first 3 stages)')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=12.0,
                        help='Gradient clipping value')

    # 优化相关
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 保存和日志
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log training info every N iterations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # GPU设置
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device id (e.g., "0" or "0,1")')

    return parser.parse_args()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) - predicted logits
            target: (B, C, D, H, W) - one-hot encoded target
        """
        pred = torch.softmax(pred, dim=1)

        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (B, C, N)
        target = target.view(target.size(0), target.size(1), -1)  # (B, C, N)

        # Calculate Dice coefficient
        intersection = (pred * target).sum(dim=2)  # (B, C)
        union = pred.sum(dim=2) + target.sum(dim=2)  # (B, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Dice + Cross Entropy Loss"""
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) - predicted logits
            target: (B, D, H, W) - target labels (not one-hot)
        """
        # For Dice loss, convert target to one-hot
        target_onehot = torch.nn.functional.one_hot(
            target.long(),
            num_classes=pred.size(1)
        ).permute(0, 4, 1, 2, 3).float()

        dice_loss = self.dice_loss(pred, target_onehot)
        ce_loss = self.ce_loss(pred, target.long())

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return total_loss, dice_loss, ce_loss


def create_model(args):
    """创建RTHD模型"""
    print(f"\n{'='*60}")
    print("Creating UMambaEnc_RTHD model...")
    print(f"{'='*60}")

    model = UMambaEnc_RTHD(
        input_size=tuple(args.patch_size),
        input_channels=args.input_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1, 2, 2, 2, 2, 2],
        n_conv_per_stage=2,
        num_classes=args.num_classes,
        n_conv_per_stage_decoder=2,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
        use_rthd=args.use_rthd,
        rthd_stages=args.rthd_stages,
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"RTHD enabled: {args.use_rthd}")
    print(f"RTHD stages: {args.rthd_stages}")
    print(f"{'='*60}\n")

    return model


def setup_training(model, args):
    """设置训练相关组件"""
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # 损失函数
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)

    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None

    return optimizer, scheduler, criterion, scaler


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return start_epoch


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, args, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if args.use_amp:
            with autocast():
                outputs = model(images)
                # 处理深度监督输出
                if isinstance(outputs, list):
                    loss, dice_loss, ce_loss = criterion(outputs[0], labels)
                    for i in range(1, len(outputs)):
                        l, d, c = criterion(outputs[i], labels)
                        loss += l * 0.5 ** i  # 深度监督权重递减
                else:
                    loss, dice_loss, ce_loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, list):
                loss, dice_loss, ce_loss = criterion(outputs[0], labels)
                for i in range(1, len(outputs)):
                    l, d, c = criterion(outputs[i], labels)
                    loss += l * 0.5 ** i
            else:
                loss, dice_loss, ce_loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss += ce_loss.item()

        # 日志输出
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_dice = total_dice_loss / (batch_idx + 1)
            avg_ce = total_ce_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}/{args.num_epochs}] "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} (Dice: {avg_dice:.4f}, CE: {avg_ce:.4f})")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """主训练函数"""
    args = parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}\n")

    # 创建模型
    model = create_model(args)
    model = model.to(device)

    # 设置训练组件
    optimizer, scheduler, criterion, scaler = setup_training(model, args)

    # 加载检查点（如果有）
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)

    # TODO: 加载数据集
    # 这里需要根据你的数据集格式实现DataLoader
    # 示例：
    # train_dataset = YourDataset(args.dataset_path, ...)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, ...)

    print("WARNING: DataLoader not implemented. Please implement your dataset loading logic.")
    print("For now, creating dummy data for demonstration...\n")

    # 创建虚拟数据用于演示（实际使用时删除）
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 生成随机数据
            image = torch.randn(4, 128, 128, 128)
            label = torch.randint(0, 3, (128, 128, 128))
            return image, label

    train_dataset = DummyDataset(num_samples=100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 训练循环
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    training_start_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        # 训练一个epoch
        avg_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch + 1
        )

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        # 打印epoch总结
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}] Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"{'='*60}\n")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}\n")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)

        # 重置GPU内存统计
        torch.cuda.reset_peak_memory_stats()

    # 训练完成
    total_time = time.time() - training_start_time
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
