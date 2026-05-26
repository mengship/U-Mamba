#!/bin/bash

TRAIN_NAME="nnUNetTrainerUMambaEncRTHD_350epochs_patience50"
RESULTS_DIR="/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2018/"$TRAIN_NAME"__nnUNetPlans__3d_fullres/"
ZIP_BASE_DIR="$(dirname "$RESULTS_DIR")"
OSS_BUCKET="oss://backup/"
OSS_CLI="oss" # 或者填写完整路径到 oss 可执行文件

LOGFILE="$RESULTS_DIR/train_and_backup_$(date '+%Y%m%d-%H%M%S').log"
# TRAIN_CMD="nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEncRTHD_350epochs_patience50"
TRAIN_CMD="nnUNetv2_train 705 3d_fullres 0 -tr "$TRAIN
# eval "$TRAIN_CMD"

echo "$(dirname "$RESULTS_DIR")"
ZIP_NAME="result-$TRAIN_NAME-$(date '+%Y%m%d-%H%M%S').zip"
cd "$(dirname "$RESULTS_DIR")" || exit
zip -q -r "$ZIP_NAME" "$(basename "$RESULTS_DIR")"

$OSS_CLI cp "$ZIP_NAME" "$OSS_BUCKET"

# shutdown