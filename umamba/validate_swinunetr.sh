#!/bin/bash
# SwinUNETR Training Validation Script
# Run this on the cloud server with full nnU-Net environment

echo "=================================="
echo "SwinUNETR Trainer Validation"
echo "=================================="
echo ""

# 1. Check Python and dependencies
echo "[1/6] Checking Python environment..."
python3 --version
echo ""

# 2. Check MONAI version
echo "[2/6] Checking MONAI version..."
python3 -c "import monai; print(f'MONAI version: {monai.__version__}')"
echo ""

# 3. Check PyTorch and CUDA
echo "[3/6] Checking PyTorch and CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 4. Test imports
echo "[4/6] Testing trainer imports..."
python3 << 'EOF'
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR_150epochs import nnUNetTrainerSwinUNETR_150epochs
    print("✓ nnUNetTrainerSwinUNETR_150epochs")
except Exception as e:
    print(f"✗ nnUNetTrainerSwinUNETR_150epochs: {e}")
    exit(1)

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR_SmokeTest import nnUNetTrainerSwinUNETR_SmokeTest
    print("✓ nnUNetTrainerSwinUNETR_SmokeTest")
except Exception as e:
    print(f"✗ nnUNetTrainerSwinUNETR_SmokeTest: {e}")
    exit(1)

print("✓ All imports successful")
EOF
echo ""

# 5. Check Dataset705 exists
echo "[5/6] Checking Dataset705_BraTS2020..."
if [ -d "$nnUNet_preprocessed/Dataset705_BraTS2020" ]; then
    echo "✓ Preprocessed data found"
    ls -lh "$nnUNet_preprocessed/Dataset705_BraTS2020" | head -5
else
    echo "✗ Preprocessed data not found at $nnUNet_preprocessed/Dataset705_BraTS2020"
    echo "  Please run preprocessing first"
fi
echo ""

# 6. Verify original file unchanged
echo "[6/6] Verifying original nnUNetTrainerSwinUNETR.py unchanged..."
if git diff --quiet nnunetv2/training/nnUNetTrainer/nnUNetTrainerSwinUNETR.py 2>/dev/null; then
    echo "✓ Original file unchanged"
else
    echo "⚠ Original file has changes (check git diff)"
fi
echo ""

echo "=================================="
echo "Validation complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run SmokeTest: CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerSwinUNETR_SmokeTest"
echo "2. Check logs and memory usage"
echo "3. If successful, run full training: CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 3d_fullres 0 -tr nnUNetTrainerSwinUNETR_150epochs"
