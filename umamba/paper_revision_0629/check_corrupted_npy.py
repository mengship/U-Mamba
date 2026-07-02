#!/usr/bin/env python3
"""
检查 nnUNet 预处理数据中损坏的 .npy 文件
"""
import numpy as np
from pathlib import Path
import sys

preprocessed_dir = Path("/hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020")

if not preprocessed_dir.exists():
    print(f"目录不存在: {preprocessed_dir}")
    sys.exit(1)

print(f"检查目录: {preprocessed_dir}")
print("=" * 80)

corrupted_files = []
checked_count = 0

for npy_file in preprocessed_dir.rglob("*.npy"):
    checked_count += 1
    try:
        # 尝试加载文件
        arr = np.load(str(npy_file), mmap_mode='r')
        # 尝试访问 shape（触发 mmap 验证）
        _ = arr.shape
    except Exception as e:
        corrupted_files.append((npy_file, str(e)))
        print(f"✗ 损坏: {npy_file.relative_to(preprocessed_dir)}")
        print(f"  错误: {e}")
        print()

print("=" * 80)
print(f"检查完成: 共检查 {checked_count} 个文件")
print(f"损坏文件: {len(corrupted_files)} 个")

if corrupted_files:
    print("\n损坏文件列表:")
    for f, _ in corrupted_files:
        print(f"  - {f}")
    print("\n修复命令:")
    for f, _ in corrupted_files:
        print(f"rm '{f}'")
else:
    print("\n✓ 所有文件正常")
