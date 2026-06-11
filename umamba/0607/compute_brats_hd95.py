#!/usr/bin/env python3
"""
Compute BraTS WT/TC/ET HD95 from nnU-Net validation predictions.

Requirements:
    nibabel
    scipy

Example:
    python umamba/0607/compute_brats_hd95.py \
        --pred-dir /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/TRAINER__nnUNetPlans__3d_fullres/fold_0/validation \
        --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
        --csv /tmp/hd95.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


REGIONS = {
    "WT": (1, 2, 3),
    "TC": (2, 3),
    "ET": (3,),
}


def region_mask(seg: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(seg, labels)


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)
    return mask ^ eroded


def hd95(mask_ref: np.ndarray, mask_pred: np.ndarray, spacing: tuple[float, float, float]) -> float:
    ref_any = bool(mask_ref.any())
    pred_any = bool(mask_pred.any())

    if not ref_any and not pred_any:
        return math.nan
    if ref_any != pred_any:
        return math.inf

    ref_surface = surface(mask_ref)
    pred_surface = surface(mask_pred)

    if not ref_surface.any() or not pred_surface.any():
        return math.inf

    dt_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    distances_pred_to_ref = dt_ref[pred_surface]
    distances_ref_to_pred = dt_pred[ref_surface]
    distances = np.concatenate([distances_pred_to_ref, distances_ref_to_pred])

    if distances.size == 0:
        return math.inf
    return float(np.percentile(distances, 95))


def load_seg(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    seg = np.asarray(img.get_fdata(), dtype=np.int16)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return seg, spacing


def compute_case(pred_path: Path, gt_path: Path) -> dict:
    pred, pred_spacing = load_seg(pred_path)
    gt, gt_spacing = load_seg(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred_path.name}: pred={pred.shape}, gt={gt.shape}")

    spacing = gt_spacing or pred_spacing
    row = {"case": pred_path.name}
    for region_name, labels in REGIONS.items():
        row[region_name] = hd95(region_mask(gt, labels), region_mask(pred, labels), spacing)
    return row


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr))


def value_counts(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "num_cases": int(arr.size),
        "finite": int(np.isfinite(arr).sum()),
        "nan": int(np.isnan(arr).sum()),
        "inf": int(np.isinf(arr).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {pred_dir}")

    rows = []
    missing = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            missing.append(pred_path.name)
            continue
        rows.append(compute_case(pred_path, gt_path))

    if missing:
        print(f"Warning: {len(missing)} predictions have no matching GT. First missing: {missing[:5]}")

    means = {region: finite_mean([row[region] for row in rows]) for region in REGIONS}
    counts = {region: value_counts([row[region] for row in rows]) for region in REGIONS}
    means["Mean"] = finite_mean(list(means.values()))

    print("| Region | HD95 |")
    print("|---|---:|")
    for region in ["WT", "TC", "ET", "Mean"]:
        value = means[region]
        value_str = "inf" if math.isinf(value) else f"{value:.6f}"
        print(f"| {region} | {value_str} |")

    print()
    print("| Region | finite | nan | inf | cases |")
    print("|---|---:|---:|---:|---:|")
    for region in ["WT", "TC", "ET"]:
        c = counts[region]
        print(f"| {region} | {c['finite']} | {c['nan']} | {c['inf']} | {c['num_cases']} |")
    if any(c["inf"] for c in counts.values()):
        print()
        print("Note: inf means one mask is empty while the other is non-empty for that region.")
        print("The reported HD95 means above average finite cases only; inspect inf counts separately.")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "WT", "TC", "ET"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump({"mean": means, "counts": counts, "per_case": rows}, f, indent=2)
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
