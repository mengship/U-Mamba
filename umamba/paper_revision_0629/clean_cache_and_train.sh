#!/bin/bash
# 清理 Python 缓存并重新训练 FullDecoderETSM

echo "=== Step 1: 清理 Python 缓存 ==="
find /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer -name "*.pyc" -delete
find /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /hy-tmp/U-Mamba/umamba/nnunetv2/nets -name "*.pyc" -delete
find /hy-tmp/U-Mamba/umamba/nnunetv2/nets -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Step 2: 验证配置文件 ==="
echo "检查 use_local_window 配置："
grep -A 2 "use_local_window" /hy-tmp/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncRTHD_FullDecoderETSM.py

echo ""
echo "=== Step 3: 重新训练（建议先验证配置） ==="
echo "请手动运行以下命令："
echo ""
echo "nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_FullDecoderETSM_150epochs"
echo ""
echo "训练开始后，检查日志中的配置输出："
echo "  Encoder RTHD config: {...'use_local_window': True...}"
echo "  Decoder RTHD config: {...'use_local_window': True...}  # 应该是 True"
echo "  Decoder mode: full"
echo ""
echo "如果仍然 OOM，考虑使用 partial 模式（见下方备选方案）"
