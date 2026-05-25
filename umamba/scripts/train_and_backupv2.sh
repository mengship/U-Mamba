#!/bin/bash
set -euo pipefail

RESULTS_DIR="/hy-tmp/U-Mamba/data/nnUNet_results/"
ZIP_BASE_DIR="$(dirname "$RESULTS_DIR")"
OSS_BUCKET="oss://backup/"
OSS_CLI="oss" # 或者填写完整路径到 oss 可执行文件

LOGFILE="$RESULTS_DIR/train_and_backup_$(date '+%Y%m%d-%H%M%S').log"
TRAIN_CMD="nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerUMambaEnc_350epochs_patience50"

ZIP_NAME="result-$(date '+%Y%m%d-%H%M%S').zip"
zip -q -r "$ZIP_NAME" "$(basename "$RESULTS_DIR")"

$OSS_CLI cp "$ZIP_NAME" "$OSS_BUCKET"