#!/usr/bin/env python3
"""Compute five-fold BraTS normalized surface Dice (NSD).

This script uses DeepMind's ``surface-distance`` implementation so that NSD
is area-weighted rather than approximated by counting boundary voxels.

Install the extra dependency once in the evaluation environment:

    pip install surface-distance

Example:

    python umamba/0607/compute_brats_nsd.py \
        --model-name Ours \
        --pred-dir '/path/to/results/fold_{fold}/validation' \
        --folds 0 1 2 3 4 \
        --gt-dir /hy-tmp/gt_segmentations \
        --tolerances 1 2 3 \
        --csv /hy-tmp/nsd/Ours_nsd.csv \
        --json /hy-tmp/nsd/Ours_nsd.json

Empty-mask policy:
    - both GT and prediction empty: NaN (excluded from the mean)
    - only one side empty: 0 (included, penalizes misses/false positives)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REGIONS = {
    "WT": (1, 2, 3),
    "TC": (2, 3),
    "ET": (3,),
}


def load_surface_metrics():
    try:
        from surface_distance import metrics
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'surface-distance'. Install it with: "
            "pip install surface-distance"
        ) from exc
    return metrics


def load_seg(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Missing NIfTI dependencies. Install them with: pip install nibabel numpy"
        ) from exc
    image = nib.load(str(path))
    segmentation = np.asarray(image.get_fdata(), dtype=np.int16)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return segmentation, spacing


def region_mask(segmentation: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    import numpy as np

    return np.isin(segmentation, labels)


def mask_status(gt_mask: np.ndarray, pred_mask: np.ndarray) -> str:
    gt_any = bool(gt_mask.any())
    pred_any = bool(pred_mask.any())
    if gt_any and pred_any:
        return "both_nonempty"
    if not gt_any and not pred_any:
        return "both_empty"
    if gt_any:
        return "gt_nonempty_pred_empty"
    return "gt_empty_pred_nonempty"


def compute_nsd(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    spacing: tuple[float, float, float],
    tolerances: list[float],
    surface_metrics,
) -> tuple[str, dict[float, float]]:
    status = mask_status(gt_mask, pred_mask)
    if status == "both_empty":
        return status, {tolerance: math.nan for tolerance in tolerances}
    if status != "both_nonempty":
        return status, {tolerance: 0.0 for tolerance in tolerances}

    distances = surface_metrics.compute_surface_distances(
        gt_mask.astype(bool), pred_mask.astype(bool), spacing
    )
    values = {
        tolerance: float(
            surface_metrics.compute_surface_dice_at_tolerance(distances, tolerance)
        )
        for tolerance in tolerances
    }
    return status, values


def tolerance_key(tolerance: float) -> str:
    value = f"{tolerance:g}".replace(".", "p")
    return f"nsd_{value}mm"


def prediction_sources(pattern: str, folds: list[int]) -> list[tuple[int, Path]]:
    if "{fold}" not in pattern:
        if len(folds) != 1:
            raise ValueError("Multiple folds require {fold} in --pred-dir")
        return [(folds[0], Path(pattern))]
    return [(fold, Path(pattern.format(fold=fold))) for fold in folds]


def find_prediction_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.nii.gz"))
    if not files:
        files = sorted(directory.glob("*.nii"))
    return files


def finite_mean(values: list[float]) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def sample_std(values: list[float]) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return math.nan
    return float(array.std(ddof=1))


def summarize(
    rows: list[dict], folds: list[int], tolerances: list[float]
) -> dict:
    per_fold = {}
    five_fold = {}

    for tolerance in tolerances:
        key = tolerance_key(tolerance)
        fold_items = {}
        for fold in folds:
            fold_rows = [row for row in rows if row["fold"] == fold]
            region_means = {
                region: finite_mean(
                    [row[key] for row in fold_rows if row["region"] == region]
                )
                for region in REGIONS
            }
            region_means["Mean"] = finite_mean(list(region_means.values()))
            fold_items[str(fold)] = region_means

        aggregate = {}
        for region in [*REGIONS, "Mean"]:
            values = [fold_items[str(fold)][region] for fold in folds]
            aggregate[region] = {
                "mean": finite_mean(values),
                "std": sample_std(values),
                "fold_values": values,
            }
        per_fold[f"{tolerance:g}mm"] = fold_items
        five_fold[f"{tolerance:g}mm"] = aggregate

    status_counts = {}
    for region in REGIONS:
        region_rows = [row for row in rows if row["region"] == region]
        status_counts[region] = {
            status: sum(row["mask_status"] == status for row in region_rows)
            for status in (
                "both_nonempty",
                "both_empty",
                "gt_nonempty_pred_empty",
                "gt_empty_pred_nonempty",
            )
        }

    return {
        "per_fold": per_fold,
        "five_fold": five_fold,
        "mask_status_counts": status_counts,
    }


def print_summary(model: str, summary: dict, tolerances: list[float]) -> None:
    print(f"Model: {model}")
    print("| Tolerance | WT | TC | ET | Mean |")
    print("|---:|---:|---:|---:|---:|")
    for tolerance in tolerances:
        item = summary["five_fold"][f"{tolerance:g}mm"]
        values = []
        for region in [*REGIONS, "Mean"]:
            mean = item[region]["mean"]
            std = item[region]["std"]
            values.append(f"{mean:.4f}+/-{std:.4f}")
        print(f"| {tolerance:g} mm | " + " | ".join(values) + " |")


def write_csv(path: Path, rows: list[dict], tolerances: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "fold",
        "case",
        "region",
        "gt_voxels",
        "pred_voxels",
        "mask_status",
        *[tolerance_key(tolerance) for tolerance in tolerances],
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute five-fold BraTS NSD at fixed physical tolerances."
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model",
        required=True,
        help="Display label written to CSV/JSON; this is not a checkpoint path",
    )
    parser.add_argument("--pred-dir", required=True, help="May contain {fold}")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--tolerances", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    folds = list(dict.fromkeys(args.folds))
    tolerances = sorted(set(args.tolerances))
    if not tolerances or any(value <= 0 for value in tolerances):
        raise ValueError("All NSD tolerances must be positive")

    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    surface_metrics = load_surface_metrics()
    rows = []
    sources = []
    missing_gt = []

    for fold, pred_dir in prediction_sources(args.pred_dir, folds):
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        prediction_files = find_prediction_files(pred_dir)
        if not prediction_files:
            raise FileNotFoundError(f"No NIfTI predictions found in {pred_dir}")

        matched = 0
        for pred_path in prediction_files:
            gt_path = gt_dir / pred_path.name
            if not gt_path.exists():
                missing_gt.append({"fold": fold, "case": pred_path.name})
                continue

            pred, pred_spacing = load_seg(pred_path)
            gt, gt_spacing = load_seg(gt_path)
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {pred_path.name}: pred={pred.shape}, gt={gt.shape}"
                )
            spacing = gt_spacing or pred_spacing

            for region, labels in REGIONS.items():
                gt_mask = region_mask(gt, labels)
                pred_mask = region_mask(pred, labels)
                status, values = compute_nsd(
                    gt_mask, pred_mask, spacing, tolerances, surface_metrics
                )
                row = {
                    "model": args.model,
                    "fold": fold,
                    "case": pred_path.name,
                    "region": region,
                    "gt_voxels": int(gt_mask.sum()),
                    "pred_voxels": int(pred_mask.sum()),
                    "mask_status": status,
                }
                row.update(
                    {tolerance_key(value): values[value] for value in tolerances}
                )
                rows.append(row)
            matched += 1

        sources.append(
            {
                "fold": fold,
                "prediction_dir": str(pred_dir),
                "prediction_cases": len(prediction_files),
                "matched_cases": matched,
            }
        )

    if not rows:
        raise RuntimeError("No prediction/GT pairs were evaluated")

    summary = summarize(rows, folds, tolerances)
    print_summary(args.model, summary, tolerances)
    print(f"Matched cases: {sum(item['matched_cases'] for item in sources)}")
    print(f"Missing GT: {len(missing_gt)}")

    if args.csv:
        write_csv(Path(args.csv), rows, tolerances)

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": args.model,
            "tolerances_mm": tolerances,
            "empty_mask_policy": {
                "both_empty": "NaN_excluded",
                "one_side_empty": "0_included",
            },
            "sources": sources,
            "missing_gt": missing_gt,
            "summary": summary,
            "per_case_region": rows,
        }
        with output_path.open("w") as file:
            json.dump(output, file, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
