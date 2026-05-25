#!/bin/bash
# RTHD模型云GPU训练部署脚本
# 使用方法: bash deploy_cloud_gpu.sh

set -e

echo "=========================================="
echo "RTHD Cloud GPU Deployment Script"
echo "=========================================="

# 1. 检查CUDA
echo -e "\n[1/6] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ CUDA available"
else
    echo "✗ CUDA not found. Please check GPU setup."
    exit 1
fi

# 2. 创建Python虚拟环境（如果不存在）
echo -e "\n[2/6] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

source venv/bin/activate

# 3. 安装依赖
echo -e "\n[3/6] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn
pip install mamba-ssm
pip install timm einops
pip install tensorboard

echo "✓ Dependencies installed"

# 4. 验证PyTorch和CUDA
echo -e "\n[4/6] Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 5. 测试RTHD模块导入
echo -e "\n[5/6] Testing RTHD module..."
python3 -c "
import sys
sys.path.insert(0, '.')
from nnunetv2.nets.rthd_modules import RTHDBlock
from nnunetv2.nets.UMambaEnc_RTHD import UMambaEnc_RTHD
print('✓ RTHD modules imported successfully')
"

# 6. 创建输出目录
echo -e "\n[6/6] Creating output directories..."
mkdir -p output/checkpoints
mkdir -p output/logs
mkdir -p data
echo "✓ Directories created"

echo -e "\n=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Place your dataset in ./data/"
echo "2. Run training:"
echo "   python train_rthd.py --dataset_path ./data --output_dir ./output --batch_size 2 --gpu 0"
echo ""
echo "For multi-GPU training:"
echo "   python train_rthd.py --dataset_path ./data --output_dir ./output --batch_size 4 --gpu 0,1"
echo ""
