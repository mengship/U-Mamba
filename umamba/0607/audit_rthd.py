"""
RTHD实验代码审计脚本
目的: 提取详细的WT/TC/ET Dice并检查实验配置
"""

import json
import os
from pathlib import Path

def extract_dice_from_summary(summary_path):
    """
    从summary.json提取详细Dice指标

    Returns:
        dict: {
            'WT': Whole Tumor Dice (regions 1,2,3),
            'TC': Tumor Core Dice (regions 2,3),
            'ET': Enhancing Tumor Dice (region 3),
            'Mean': Foreground Mean Dice
        }
    """
    with open(summary_path, 'r') as f:
        data = json.load(f)

    mean_metrics = data.get('mean', {})

    # nnU-Net的region key格式: "(1, 2, 3)" 或 "(2, 3)" 或 "(3,)"
    result = {
        'WT': mean_metrics.get('(1, 2, 3)', {}).get('Dice', None),
        'TC': mean_metrics.get('(2, 3)', {}).get('Dice', None),
        'ET': mean_metrics.get('(3,)', {}).get('Dice', None),
        'Mean': data.get('foreground_mean', {}).get('Dice', None)
    }

    return result


def compare_experiments(results_root):
    """
    比较三个实验的Dice结果

    Args:
        results_root: nnUNet训练结果根目录，例如
                     /path/to/nnUNet_results/Dataset705_BraTS2018/
    """
    trainers = [
        'nnUNetTrainerUMambaEnc_150epochs',
        'nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs',
        'nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs',
    ]

    print("=" * 80)
    print("RTHD实验结果对比 (fold 0, 150 epochs)")
    print("=" * 80)
    print()

    all_results = {}

    for trainer in trainers:
        summary_path = Path(results_root) / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_0" / "validation" / "summary.json"

        if not summary_path.exists():
            print(f"⚠️  {trainer}: summary.json not found")
            print(f"   Expected: {summary_path}")
            continue

        dice_results = extract_dice_from_summary(summary_path)
        all_results[trainer] = dice_results

        print(f"{trainer}:")
        print(f"  WT (Whole Tumor):     {dice_results['WT']:.6f}" if dice_results['WT'] else "  WT: N/A")
        print(f"  TC (Tumor Core):      {dice_results['TC']:.6f}" if dice_results['TC'] else "  TC: N/A")
        print(f"  ET (Enhancing Tumor): {dice_results['ET']:.6f}" if dice_results['ET'] else "  ET: N/A")
        print(f"  Mean Foreground Dice: {dice_results['Mean']:.6f}" if dice_results['Mean'] else "  Mean: N/A")
        print()

    # 计算相对差异
    if len(all_results) >= 2:
        baseline_name = 'nnUNetTrainerUMambaEnc_150epochs'
        if baseline_name in all_results:
            baseline = all_results[baseline_name]
            print("=" * 80)
            print("相对Baseline的差异 (百分点)")
            print("=" * 80)
            print()

            for trainer in trainers[1:]:  # 跳过baseline自己
                if trainer not in all_results:
                    continue

                result = all_results[trainer]
                print(f"{trainer}:")

                for metric in ['WT', 'TC', 'ET', 'Mean']:
                    if baseline[metric] is not None and result[metric] is not None:
                        diff = (result[metric] - baseline[metric]) * 100
                        sign = "+" if diff > 0 else ""
                        print(f"  {metric}: {sign}{diff:.4f}%")
                    else:
                        print(f"  {metric}: N/A")
                print()


def check_ss2d_import_in_logs(log_path):
    """
    检查训练日志中SS2D导入状态

    关键标识:
    - "✅ Successfully imported SS2D" -> 成功导入真实VMamba
    - "❌ ERROR: Cannot import SS2D" -> 导入失败
    - "Using placeholder fallback" -> 使用占位符（严重性能退化）
    """
    if not os.path.exists(log_path):
        print(f"⚠️  Log file not found: {log_path}")
        return

    print("=" * 80)
    print(f"检查SS2D导入状态: {log_path}")
    print("=" * 80)

    with open(log_path, 'r') as f:
        content = f.read()

    if "✅ Successfully imported SS2D" in content:
        print("✅ SS2D导入成功 - 使用真实VMamba")
        # 计算成功导入的次数
        count = content.count("✅ Successfully imported SS2D")
        print(f"   导入次数: {count} (每个RTHDBlock stage一次)")
    elif "❌ ERROR: Cannot import SS2D" in content:
        print("❌ SS2D导入失败!")
        if "Using placeholder fallback" in content:
            print("   ⚠️  使用占位符fallback (LayerNorm + Linear + GELU)")
            print("   ⚠️  这意味着RTHD退化为普通MLP，没有VMamba的长距离建模能力!")
        print()
        print("原因可能是:")
        print("  1. instructions/vmamba.py文件不存在")
        print("  2. SS2D类在vmamba.py中定义有误")
        print("  3. sys.path配置问题")
    else:
        print("⚠️  日志中未找到SS2D导入相关信息")

    print()


if __name__ == "__main__":
    # 示例用法
    print("RTHD实验代码审计脚本")
    print()
    print("用法1: 比较实验结果")
    print("  python audit_rthd.py --results /path/to/nnUNet_results/Dataset705_BraTS2018/")
    print()
    print("用法2: 检查SS2D导入状态")
    print("  python audit_rthd.py --check-log /path/to/training.log")
    print()
    print("=" * 80)
    print()

    # 如果提供了参数，执行相应功能
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--results' and len(sys.argv) > 2:
            compare_experiments(sys.argv[2])
        elif sys.argv[1] == '--check-log' and len(sys.argv) > 2:
            check_ss2d_import_in_logs(sys.argv[2])
        else:
            print("无效参数")
            print("使用 --results <path> 或 --check-log <path>")
    else:
        print("请提供参数")
        print("  --results <nnUNet_results_path>")
        print("  --check-log <training_log_path>")
