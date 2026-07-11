#!/usr/bin/env python3
"""Audit empty BraTS region masks and the corresponding HD95 status.

This script does not change the existing HD95 evaluation. It supports either
prediction NIfTI files or previously generated HD95 CSV/JSON files. Combined
with GT files, both modes distinguish complete misses from false positives
when one side of an HD95 comparison is empty.

Example for five nnU-Net folds:
    python umamba/0607/audit_brats_empty_masks.py \
        --model nnUNet \
        --pred-dir '/path/to/trainer/fold_{fold}/validation' \
        --folds 0 1 2 3 4 \
        --gt-dir /path/to/gt_segmentations \
        --csv /tmp/nnunet_empty_masks.csv \
        --json /tmp/nnunet_empty_masks.json

Example using existing HD95 CSV files and GT only:
    python umamba/0607/audit_brats_empty_masks.py \
        --model nnUNet \
        --hd95-file '/path/to/nnUNetTrainer_fold{fold}_hd95.csv' \
        --folds 0 1 2 3 4 \
        --gt-dir /path/to/gt_segmentations \
        --csv /tmp/nnunet_empty_masks.csv \
        --json /tmp/nnunet_empty_masks.json
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

from compute_brats_hd95 import REGIONS, hd95, load_seg, region_mask


MASK_STATUSES = (
    "both_nonempty",
    "both_empty",
    "gt_nonempty_pred_empty",
    "gt_empty_pred_nonempty",
)


def classify_masks(gt_any: bool, pred_any: bool) -> str:
    if gt_any and pred_any:
        return "both_nonempty"
    if not gt_any and not pred_any:
        return "both_empty"
    if gt_any:
        return "gt_nonempty_pred_empty"
    return "gt_empty_pred_nonempty"


def hd95_status(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return "finite"


def audit_case(pred_path: Path, gt_path: Path, model: str, fold: Optional[int]) -> list[dict]:
    pred, pred_spacing = load_seg(pred_path)
    gt, gt_spacing = load_seg(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred_path.name}: pred={pred.shape}, gt={gt.shape}")

    spacing = gt_spacing or pred_spacing
    rows = []
    for region, labels in REGIONS.items():
        gt_mask = region_mask(gt, labels)
        pred_mask = region_mask(pred, labels)
        value = hd95(gt_mask, pred_mask, spacing)
        status = hd95_status(value)
        rows.append({
            "model": model,
            "fold": "" if fold is None else fold,
            "case": pred_path.name,
            "region": region,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
            "mask_status": classify_masks(bool(gt_mask.any()), bool(pred_mask.any())),
            "hd95_status": status,
            "hd95_mm": value if status == "finite" else None,
        })
    return rows


def read_hd95_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        rows = data.get("per_case")
        if not isinstance(rows, list):
            raise ValueError(f"JSON has no per_case list: {path}")
        return rows

    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"HD95 input must be CSV or JSON: {path}")


def audit_hd95_row(metric_row: dict, gt_path: Path, model: str, fold: Optional[int]) -> list[dict]:
    gt, _ = load_seg(gt_path)
    rows = []
    for region, labels in REGIONS.items():
        if region not in metric_row:
            raise ValueError(f"Missing {region} HD95 for {metric_row.get('case', gt_path.name)}")

        value = float(metric_row[region])
        status = hd95_status(value)
        gt_mask = region_mask(gt, labels)
        gt_any = bool(gt_mask.any())

        if status == "finite":
            if not gt_any:
                raise ValueError(f"Finite {region} HD95 but GT is empty: {gt_path.name}")
            mask_status = "both_nonempty"
        elif status == "nan":
            if gt_any:
                raise ValueError(f"NaN {region} HD95 but GT is nonempty: {gt_path.name}")
            mask_status = "both_empty"
        elif gt_any:
            mask_status = "gt_nonempty_pred_empty"
        else:
            mask_status = "gt_empty_pred_nonempty"

        rows.append({
            "model": model,
            "fold": "" if fold is None else fold,
            "case": metric_row.get("case", gt_path.name),
            "region": region,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": None,
            "mask_status": mask_status,
            "hd95_status": status,
            "hd95_mm": value if status == "finite" else None,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for region in REGIONS:
        region_rows = [row for row in rows if row["region"] == region]
        finite_values = [row["hd95_mm"] for row in region_rows if row["hd95_status"] == "finite"]
        item = {
            "num_cases": len(region_rows),
            **{
                status: sum(row["mask_status"] == status for row in region_rows)
                for status in MASK_STATUSES
            },
            "finite_hd95": len(finite_values),
            "nan_hd95": sum(row["hd95_status"] == "nan" for row in region_rows),
            "inf_hd95": sum(row["hd95_status"] == "inf" for row in region_rows),
            "finite_hd95_mean_mm": (
                sum(finite_values) / len(finite_values) if finite_values else None
            ),
        }
        summary[region] = item
    return summary


def input_sources(path_pattern: str, folds: list[int]) -> list[tuple[Optional[int], Path]]:
    has_placeholder = "{fold}" in path_pattern
    if has_placeholder and not folds:
        raise ValueError("Input contains {fold}; provide --folds or one or more --fold values")
    if not has_placeholder and len(folds) > 1:
        raise ValueError("Multiple folds require {fold} in the input path")
    if has_placeholder:
        return [(fold, Path(path_pattern.format(fold=fold))) for fold in folds]
    return [(folds[0] if folds else None, Path(path_pattern))]


def print_summary(summary: dict) -> None:
    print("| Region | Both nonempty | Both empty | GT empty only | Pred empty only | Finite HD95 | Mean finite HD95 (mm) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for region in REGIONS:
        item = summary[region]
        mean_value = item["finite_hd95_mean_mm"]
        mean_text = "NA" if mean_value is None else f"{mean_value:.6f}"
        print(
            f"| {region} | {item['both_nonempty']} | {item['both_empty']} | "
            f"{item['gt_empty_pred_nonempty']} | {item['gt_nonempty_pred_empty']} | "
            f"{item['finite_hd95']} | {mean_text} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distinguish empty-mask cases in BraTS WT/TC/ET HD95 evaluation."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pred-dir", help="Prediction directory; may contain {fold}")
    inputs.add_argument("--hd95-file", help="Existing HD95 CSV/JSON; may contain {fold}")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--fold", dest="fold_values", action="append", type=int, default=[])
    parser.add_argument("--folds", nargs="+", type=int, default=[])
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    folds = list(dict.fromkeys(args.fold_values + args.folds))
    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    rows = []
    missing_gt = []
    source_counts = []
    if args.pred_dir:
        for fold, pred_path in input_sources(args.pred_dir, folds):
            if not pred_path.is_dir():
                raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
            pred_files = sorted(pred_path.glob("*.nii.gz"))
            if not pred_files:
                raise FileNotFoundError(f"No .nii.gz predictions found in {pred_path}")

            matched = 0
            for prediction in pred_files:
                gt_path = gt_dir / prediction.name
                if not gt_path.exists():
                    missing_gt.append({"fold": fold, "case": prediction.name})
                    continue
                rows.extend(audit_case(prediction, gt_path, args.model, fold))
                matched += 1
            source_counts.append({
                "fold": fold,
                "input": str(pred_path),
                "input_type": "prediction_directory",
                "input_cases": len(pred_files),
                "matched_cases": matched,
            })
    else:
        for fold, hd95_path in input_sources(args.hd95_file, folds):
            if not hd95_path.is_file():
                raise FileNotFoundError(f"HD95 file not found: {hd95_path}")
            metric_rows = read_hd95_rows(hd95_path)
            matched = 0
            for metric_row in metric_rows:
                case = metric_row.get("case")
                if not case:
                    raise ValueError(f"HD95 row has no case name: {hd95_path}")
                gt_path = gt_dir / Path(case).name
                if not gt_path.exists():
                    missing_gt.append({"fold": fold, "case": Path(case).name})
                    continue
                rows.extend(audit_hd95_row(metric_row, gt_path, args.model, fold))
                matched += 1
            source_counts.append({
                "fold": fold,
                "input": str(hd95_path),
                "input_type": "existing_hd95",
                "input_cases": len(metric_rows),
                "matched_cases": matched,
            })

    if not rows:
        raise RuntimeError("No prediction/GT pairs were audited")

    summary = summarize(rows)
    print_summary(summary)
    if missing_gt:
        print(f"\nWarning: {len(missing_gt)} predictions had no matching GT.")

    fieldnames = [
        "model",
        "fold",
        "case",
        "region",
        "gt_voxels",
        "pred_voxels",
        "mask_status",
        "hd95_status",
        "hd95_mm",
    ]
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "regions": {name: list(labels) for name, labels in REGIONS.items()},
            "sources": source_counts,
            "missing_gt": missing_gt,
            "summary": summary,
            "per_case_region": rows,
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
