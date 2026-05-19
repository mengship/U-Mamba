#!/bin/bash
set -euo pipefail

# Config — 修改为你的路径和 OSS 桶
RESULTS_DIR="/hy-tmp/U-Mamba/data/nnUNet_results/"
ZIP_BASE_DIR="$(dirname "$RESULTS_DIR")"
OSS_BUCKET="oss://backup/"
OSS_CLI="oss" # 或者填写完整路径到 oss 可执行文件

# Ensure results directory exists so the logfile can be written into it
mkdir -p "$RESULTS_DIR"

# 可通过环境变量覆盖以下命令，例如在 tmux 中：
# 使用官方 trainer（默认）——如需 NoAMP 或短 epoch 变体，可通过环境变量覆盖 `TRAIN_CMD`
# 示例覆盖：
# TRAIN_CMD='nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncNoAMP_50epochs' bash /root/train_and_backup.sh
TRAIN_CMD="nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEnc"
LOGFILE="$RESULTS_DIR/train_and_backup_$(date '+%Y%m%d-%H%M%S').log"

echo "[train_and_backup] Log: $LOGFILE"

# Run training
echo "[train_and_backup] Starting training sequence at $(date)" | tee -a "$LOGFILE"

# Previously we ran a single fold via TRAIN_CMD. That logic is preserved as comments
# so you can still override via TRAIN_CMD environment variable if desired.
# eval "$TRAIN_CMD" 2>&1 | tee -a "$LOGFILE"

# New behavior: run folds 1-4 sequentially (fold 0 assumed already completed).
# Keep the flow simple: train, then backup, then shutdown.
# Using nnUNetTrainerUMambaBot_350epochs_patience50 for max_epoch=350, patience=50
for F in 1 2 3 4; do
  echo "[train_and_backup] Starting training for fold $F at $(date)" | tee -a "$LOGFILE"
  TRAIN_CMD_F="nnUNetv2_train 705 3d_fullres $F -tr nnUNetTrainerUMambaBot_350epochs_patience50"
  echo "[train_and_backup] Running: $TRAIN_CMD_F" | tee -a "$LOGFILE"
  set -o pipefail
  eval "$TRAIN_CMD_F" 2>&1 | tee -a "$LOGFILE"
  TRAIN_EXIT=${PIPESTATUS[0]:-$?}
  if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "[train_and_backup] Training fold $F exited with code $TRAIN_EXIT — aborting further folds." | tee -a "$LOGFILE"
    exit $TRAIN_EXIT
  fi
  echo "[train_and_backup] Fold $F completed at $(date)" | tee -a "$LOGFILE"
done

echo "[train_and_backup] All folds 1-4 finished at $(date)" | tee -a "$LOGFILE"

# Backup
echo "[train_and_backup] Training finished at $(date). Preparing backup..." | tee -a "$LOGFILE"
cd "$ZIP_BASE_DIR"
ZIP_NAME="result-$(date '+%Y%m%d-%H%M%S').zip"
# Ensure the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
  echo "[train_and_backup] Results directory not found: $RESULTS_DIR" | tee -a "$LOGFILE"
  exit 1
fi

# Create zip
echo "[train_and_backup] Creating archive $ZIP_NAME (this may take time)..." | tee -a "$LOGFILE"
zip -q -r "$ZIP_NAME" "$(basename "$RESULTS_DIR")"

# Upload
if ! command -v "$OSS_CLI" >/dev/null 2>&1; then
  echo "[train_and_backup] OSS CLI '$OSS_CLI' not found in PATH. Aborting." | tee -a "$LOGFILE"
  exit 1
fi

echo "[train_and_backup] Uploading $ZIP_NAME to $OSS_BUCKET" | tee -a "$LOGFILE"
$OSS_CLI cp "$ZIP_NAME" "$OSS_BUCKET" 2>&1 | tee -a "$LOGFILE"
UPLOAD_EXIT=${PIPESTATUS[0]:-$?}
if [ "$UPLOAD_EXIT" -ne 0 ]; then
  echo "[train_and_backup] Upload failed with exit $UPLOAD_EXIT — leaving system running for inspection." | tee -a "$LOGFILE"
  exit $UPLOAD_EXIT
fi

# Cleanup and shutdown
rm -f "$ZIP_NAME"
echo "[train_and_backup] Backup successful at $(date). Removing local archive and shutting down." | tee -a "$LOGFILE"
sleep 5
shutdown
