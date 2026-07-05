#!/usr/bin/env python3
"""
只读统计脚本：汇总nnUNetv2不同trainer的五折验证Dice结果
用于C1消融实验结果汇总

作者：自动生成
日期：2026-07-05
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import pandas as pd
import numpy as np


# 配置
RESULTS_ROOT = Path("/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020")
OUTPUT_CSV = Path("/hy-tmp/c1_five_fold_ablation_dice.csv")
OUTPUT_JSON = Path("/hy-tmp/c1_five_fold_ablation_dice_summary.json")

TRAINERS = [
    "nnUNetTrainerUMambaEnc_150epochs",
    "nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs",
]

FOLDS = [0, 1, 2, 3, 4]
REGIONS = ["WT", "TC", "ET", "Mean"]


def extract_dice_from_summary(summary_path: Path) -> Optional[Dict[str, float]]:
    """
    从summary.json中提取WT、TC、ET和Mean Dice
    兼容多种nnUNetv2输出格式
    """
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)

        dice_scores = {}

        # 方法1: 查找 "mean" 或 "foreground_mean" 字段
        if "mean" in data:
            mean_data = data["mean"]
            if isinstance(mean_data, dict):
                # 如果mean是字典，尝试提取Dice字段
                if "Dice" in mean_data:
                    dice_scores["Mean"] = float(mean_data["Dice"])
                # 尝试按类别提取 (假设类别1=WT, 2=TC, 3=ET)
                for idx, region in enumerate(["WT", "TC", "ET"], start=1):
                    key = f"{idx}"
                    if key in mean_data and "Dice" in mean_data[key]:
                        dice_scores[region] = float(mean_data[key]["Dice"])
            elif isinstance(mean_data, (list, tuple)):
                # 如果mean是列表，假设顺序是[背景, WT, TC, ET]或[WT, TC, ET]
                if len(mean_data) == 4:  # [背景, WT, TC, ET]
                    dice_scores["WT"] = float(mean_data[1])
                    dice_scores["TC"] = float(mean_data[2])
                    dice_scores["ET"] = float(mean_data[3])
                    dice_scores["Mean"] = float(np.mean(mean_data[1:]))
                elif len(mean_data) == 3:  # [WT, TC, ET]
                    dice_scores["WT"] = float(mean_data[0])
                    dice_scores["TC"] = float(mean_data[1])
                    dice_scores["ET"] = float(mean_data[2])
                    dice_scores["Mean"] = float(np.mean(mean_data))

        # 方法2: 查找 "foreground_mean" 字段
        if "foreground_mean" in data and not dice_scores:
            fg_mean = data["foreground_mean"]
            if isinstance(fg_mean, dict) and "Dice" in fg_mean:
                dice_scores["Mean"] = float(fg_mean["Dice"])

        # 方法3: 查找 "results" 或 "metric_per_case" 字段
        if "results" in data and not dice_scores:
            results = data["results"]
            if "mean" in results:
                mean_scores = results["mean"]
                if isinstance(mean_scores, dict):
                    for region in REGIONS:
                        if region in mean_scores:
                            dice_scores[region] = float(mean_scores[region])

        # 方法4: 直接查找顶层的WT、TC、ET字段
        for region in REGIONS:
            if region not in dice_scores and region in data:
                if isinstance(data[region], (int, float)):
                    dice_scores[region] = float(data[region])
                elif isinstance(data[region], dict) and "Dice" in data[region]:
                    dice_scores[region] = float(data[region]["Dice"])

        # 如果没有找到Mean，但有WT、TC、ET，则计算Mean
        if "Mean" not in dice_scores and all(r in dice_scores for r in ["WT", "TC", "ET"]):
            dice_scores["Mean"] = float(np.mean([dice_scores["WT"], dice_scores["TC"], dice_scores["ET"]]))

        # 验证是否至少提取到了一些数据
        if not dice_scores:
            warnings.warn(f"无法从 {summary_path} 提取任何Dice分数")
            return None

        # 确保所有区域都有值（如果缺失则填充None）
        result = {region: dice_scores.get(region, None) for region in REGIONS}
        return result

    except FileNotFoundError:
        warnings.warn(f"文件不存在: {summary_path}")
        return None
    except json.JSONDecodeError as e:
        warnings.warn(f"JSON解析错误 {summary_path}: {e}")
        return None
    except Exception as e:
        warnings.warn(f"读取 {summary_path} 时出错: {e}")
        return None


def find_validation_summary(trainer_path: Path, fold: int) -> Optional[Path]:
    """
    查找指定fold的validation目录下的summary.json
    """
    validation_dir = trainer_path / f"fold_{fold}" / "validation"

    # 尝试多个可能的文件名
    possible_files = [
        validation_dir / "summary.json",
        validation_dir / "validation_summary.json",
        validation_dir / "metrics.json",
        validation_dir / "dice_scores.json",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    return None


def collect_trainer_results(trainer_name: str) -> Tuple[List[Dict], Dict[str, float]]:
    """
    收集单个trainer的所有fold结果
    返回: (每个fold的结果列表, 汇总统计)
    """
    trainer_path = RESULTS_ROOT / f"{trainer_name}__nnUNetPlans__3d_fullres"

    if not trainer_path.exists():
        warnings.warn(f"WARNING: Trainer目录不存在: {trainer_path}")
        return [], {}

    fold_results = []
    mean_dice_list = []

    for fold in FOLDS:
        summary_path = find_validation_summary(trainer_path, fold)

        if summary_path is None:
            warnings.warn(f"WARNING: 未找到 {trainer_name} fold {fold} 的summary文件")
            fold_results.append({
                "Trainer": trainer_name,
                "Fold": fold,
                "WT": None,
                "TC": None,
                "ET": None,
                "Mean": None,
            })
            continue

        dice_scores = extract_dice_from_summary(summary_path)

        if dice_scores is None:
            warnings.warn(f"WARNING: 无法从 {trainer_name} fold {fold} 提取Dice分数")
            fold_results.append({
                "Trainer": trainer_name,
                "Fold": fold,
                "WT": None,
                "TC": None,
                "ET": None,
                "Mean": None,
            })
            continue

        fold_result = {
            "Trainer": trainer_name,
            "Fold": fold,
            **dice_scores
        }
        fold_results.append(fold_result)

        if dice_scores.get("Mean") is not None:
            mean_dice_list.append(dice_scores["Mean"])

    # 计算汇总统计
    summary = {}
    if mean_dice_list:
        summary["mean"] = float(np.mean(mean_dice_list))
        summary["std"] = float(np.std(mean_dice_list, ddof=1)) if len(mean_dice_list) > 1 else 0.0
        summary["n_folds"] = len(mean_dice_list)
    else:
        summary["mean"] = None
        summary["std"] = None
        summary["n_folds"] = 0

    return fold_results, summary


def format_markdown_table(df: pd.DataFrame, summary_dict: Dict) -> str:
    """
    生成Markdown格式的表格
    """
    lines = []
    lines.append("\n## C1消融实验 - 五折验证Dice结果\n")

    # 详细结果表
    lines.append("### 每个Fold的详细结果\n")
    lines.append("| Trainer | Fold | WT | TC | ET | Mean |")
    lines.append("|---------|------|----|----|----|----- |")

    for _, row in df.iterrows():
        trainer_short = row["Trainer"].replace("nnUNetTrainerUMambaEncRTHD_", "").replace("nnUNetTrainerUMambaEnc_", "Baseline_")
        fold = row["Fold"]
        wt = f"{row['WT']:.4f}" if pd.notna(row['WT']) else "N/A"
        tc = f"{row['TC']:.4f}" if pd.notna(row['TC']) else "N/A"
        et = f"{row['ET']:.4f}" if pd.notna(row['ET']) else "N/A"
        mean = f"{row['Mean']:.4f}" if pd.notna(row['Mean']) else "N/A"
        lines.append(f"| {trainer_short} | {fold} | {wt} | {tc} | {et} | {mean} |")

    # 汇总统计表
    lines.append("\n### 五折平均结果\n")
    lines.append("| Trainer | Mean Dice (5-fold) | Std Dev |")
    lines.append("|---------|-------------------|---------|")

    for trainer_name, stats in summary_dict.items():
        trainer_short = trainer_name.replace("nnUNetTrainerUMambaEncRTHD_", "").replace("nnUNetTrainerUMambaEnc_", "Baseline_")
        mean = f"{stats['mean']:.4f}" if stats['mean'] is not None else "N/A"
        std = f"{stats['std']:.4f}" if stats['std'] is not None else "N/A"
        n_folds = stats['n_folds']
        lines.append(f"| {trainer_short} | {mean} ± {std} | (n={n_folds}) |")

    return "\n".join(lines)


def main():
    """
    主函数：收集所有trainer的结果并生成输出
    """
    print("=" * 80)
    print("C1消融实验 - nnUNetv2五折验证Dice结果汇总")
    print("=" * 80)
    print(f"结果根目录: {RESULTS_ROOT}")
    print(f"输出CSV: {OUTPUT_CSV}")
    print(f"输出JSON: {OUTPUT_JSON}")
    print("-" * 80)

    all_results = []
    summary_dict = {}

    for trainer in TRAINERS:
        print(f"\n处理 {trainer}...")
        fold_results, summary = collect_trainer_results(trainer)
        all_results.extend(fold_results)
        summary_dict[trainer] = summary

        if summary.get("mean") is not None:
            print(f"  五折平均Dice: {summary['mean']:.4f} ± {summary['std']:.4f} (n={summary['n_folds']})")
        else:
            print(f"  未找到有效结果")

    # 创建DataFrame
    df = pd.DataFrame(all_results)

    # 保存CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"\n已保存CSV: {OUTPUT_CSV}")

    # 保存JSON
    output_data = {
        "metadata": {
            "dataset": "Dataset705_BraTS2020",
            "n_folds": len(FOLDS),
            "regions": REGIONS,
            "date": "2026-07-05",
        },
        "fold_results": all_results,
        "summary": summary_dict,
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"已保存JSON: {OUTPUT_JSON}")

    # 生成并打印Markdown表格
    markdown_table = format_markdown_table(df, summary_dict)
    print("\n" + "=" * 80)
    print("Markdown表格（可直接复制到论文）:")
    print("=" * 80)
    print(markdown_table)
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
