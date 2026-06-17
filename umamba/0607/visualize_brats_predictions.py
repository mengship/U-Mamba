#!/usr/bin/env python3
"""
Create paper-style BraTS segmentation visualizations.

The figure layout is:
    case rows x [MRI, GT, prediction_1, prediction_2, ...] columns

Examples:
    python umamba/0607/visualize_brats_predictions.py \
        --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
        --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
        --pred "nnU-Net=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
        --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
        --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
        --cases BraTS20_Training_011 BraTS20_Training_012 BraTS20_Training_019 \
        --out /hy-tmp/brats_visualization.png

Notes:
    - Default modality is FLAIR, which maps to nnU-Net channel _0003.
    - Labels are visualized as label 1/2/3 with green/yellow/red overlays.
    - Original BraTS labels 0/1/2/4 are converted to the nnU-Net convention 0/2/1/3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch


MODALITY_TO_CHANNEL = {
    "t1": "0000",
    "t1ce": "0001",
    "t2": "0002",
    "flair": "0003",
    "0": "0000",
    "1": "0001",
    "2": "0002",
    "3": "0003",
}

AXIS_TO_INDEX = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}

LABEL_COLORS = {
    1: (0.00, 0.85, 0.20),  # edema / WT-only area
    2: (1.00, 0.90, 0.00),  # tumor core-related area
    3: (1.00, 0.10, 0.10),  # enhancing tumor
}


def strip_nii_suffix(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def parse_pred_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    specs = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--pred must be NAME=DIR, got: {value}")
        name, directory = value.split("=", 1)
        name = name.strip()
        directory = directory.strip()
        if not name or not directory:
            raise ValueError(f"--pred must be NAME=DIR, got: {value}")
        specs.append((name, Path(directory)))
    return specs


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def normalize_image_slice(image_2d: np.ndarray) -> np.ndarray:
    arr = image_2d.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99.5])
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def normalize_seg_labels(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(np.rint(seg), dtype=np.int16)
    uniques = set(int(i) for i in np.unique(seg))
    if 4 in uniques:
        converted = np.zeros_like(seg, dtype=np.int16)
        converted[seg == 2] = 1
        converted[seg == 1] = 2
        converted[seg == 4] = 3
        return converted
    return seg


def find_image_path(image_dir: Path, case_id: str, modality: str) -> Path:
    channel = MODALITY_TO_CHANNEL.get(modality.lower())
    if channel is None:
        raise ValueError(f"Unknown modality {modality}. Use one of: {sorted(MODALITY_TO_CHANNEL)}")

    candidates = [
        image_dir / f"{case_id}_{channel}.nii.gz",
        image_dir / f"{case_id}_{channel}.nii",
        image_dir / case_id / f"{case_id}_{modality.lower()}.nii.gz",
        image_dir / case_id / f"{case_id}_{modality.lower()}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path

    suffixes = {
        "0000": "t1",
        "0001": "t1ce",
        "0002": "t2",
        "0003": "flair",
    }
    raw_suffix = suffixes[channel]
    extra_patterns = [
        f"**/{case_id}_{raw_suffix}.nii.gz",
        f"**/{case_id}_{raw_suffix}.nii",
        f"**/{case_id}_{channel}.nii.gz",
        f"**/{case_id}_{channel}.nii",
    ]
    for pattern in extra_patterns:
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find {modality} image for {case_id} in {image_dir}. "
        f"Tried nnU-Net channel _{channel} and raw BraTS suffix _{raw_suffix}."
    )


def find_seg_path(seg_dir: Path, case_id: str) -> Path:
    candidates = [
        seg_dir / f"{case_id}.nii.gz",
        seg_dir / f"{case_id}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(seg_dir.glob(f"**/{case_id}.nii*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find segmentation for {case_id} in {seg_dir}")


def auto_cases(gt_dir: Path, pred_specs: list[tuple[str, Path]], num_cases: int) -> list[str]:
    gt_cases = {strip_nii_suffix(p) for p in gt_dir.glob("*.nii*")}
    if not gt_cases:
        raise FileNotFoundError(f"No GT .nii/.nii.gz files found in {gt_dir}")

    common = set(gt_cases)
    for _, pred_dir in pred_specs:
        pred_cases = {strip_nii_suffix(p) for p in pred_dir.glob("*.nii*")}
        common &= pred_cases

    cases = sorted(common)
    if not cases:
        raise FileNotFoundError("No common cases found between GT and prediction directories")
    return cases[:num_cases]


def choose_slice(seg: np.ndarray, axis: int) -> int:
    mask = seg > 0
    if not mask.any():
        return int(seg.shape[axis] // 2)
    axes_to_sum = tuple(i for i in range(3) if i != axis)
    areas = mask.sum(axis=axes_to_sum)
    return int(np.argmax(areas))


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        arr = volume[index, :, :]
    elif axis == 1:
        arr = volume[:, index, :]
    elif axis == 2:
        arr = volume[:, :, index]
    else:
        raise ValueError(f"Invalid axis index: {axis}")
    return np.rot90(arr)


def crop_slices(slices: list[np.ndarray], masks: list[np.ndarray], margin: int) -> list[np.ndarray]:
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask > 0

    if not union.any():
        return slices

    rows, cols = np.where(union)
    r0 = max(int(rows.min()) - margin, 0)
    r1 = min(int(rows.max()) + margin + 1, union.shape[0])
    c0 = max(int(cols.min()) - margin, 0)
    c1 = min(int(cols.max()) + margin + 1, union.shape[1])

    height = r1 - r0
    width = c1 - c0
    side = max(height, width)
    r_center = (r0 + r1) // 2
    c_center = (c0 + c1) // 2
    r0 = max(r_center - side // 2, 0)
    c0 = max(c_center - side // 2, 0)
    r1 = min(r0 + side, union.shape[0])
    c1 = min(c0 + side, union.shape[1])
    r0 = max(r1 - side, 0)
    c0 = max(c1 - side, 0)

    return [arr[r0:r1, c0:c1] for arr in slices]


def overlay_seg(ax, image_2d: np.ndarray, seg_2d: np.ndarray, alpha: float, contour: bool) -> None:
    ax.imshow(image_2d, cmap="gray", vmin=0, vmax=1)
    for label, color in LABEL_COLORS.items():
        mask = seg_2d == label
        if not mask.any():
            continue
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = color
        rgba[..., 3] = mask.astype(np.float32) * alpha
        ax.imshow(rgba, interpolation="nearest")
        if contour:
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.8)


def build_case_data(
    case_id: str,
    image_dir: Path,
    gt_dir: Path,
    pred_specs: list[tuple[str, Path]],
    modality: str,
    axis: int,
    requested_slice: int | None,
    crop: bool,
    crop_margin: int,
) -> tuple[list[np.ndarray], int]:
    image = load_nifti(find_image_path(image_dir, case_id, modality))
    gt = normalize_seg_labels(load_nifti(find_seg_path(gt_dir, case_id)))
    preds = [normalize_seg_labels(load_nifti(find_seg_path(pred_dir, case_id))) for _, pred_dir in pred_specs]

    slice_index = requested_slice if requested_slice is not None else choose_slice(gt, axis)
    image_2d = normalize_image_slice(take_slice(image, axis, slice_index))
    gt_2d = take_slice(gt, axis, slice_index)
    pred_2d = [take_slice(pred, axis, slice_index) for pred in preds]

    panels = [image_2d, gt_2d, *pred_2d]
    if crop:
        masks = [gt_2d, *pred_2d]
        panels = crop_slices(panels, masks, crop_margin)
    return panels, slice_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="nnU-Net imagesTr folder or raw BraTS training root")
    parser.add_argument("--gt-dir", required=True, help="GT segmentation folder, for example gt_segmentations")
    parser.add_argument("--pred", action="append", default=[], help="Prediction spec NAME=DIR. Can be repeated.")
    parser.add_argument("--cases", nargs="+", default=None, help="Case ids, for example BraTS20_Training_011")
    parser.add_argument("--num-cases", type=int, default=3, help="Number of auto-selected cases when --cases is omitted")
    parser.add_argument("--modality", default="flair", help="t1/t1ce/t2/flair or 0/1/2/3. Default: flair")
    parser.add_argument("--axis", choices=tuple(AXIS_TO_INDEX), default="axial")
    parser.add_argument("--slice", type=int, default=None, help="Use a fixed slice index instead of max-tumor slice")
    parser.add_argument("--no-crop", action="store_true", help="Disable tumor-centered cropping")
    parser.add_argument("--crop-margin", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--no-contour", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    pred_specs = parse_pred_specs(args.pred)
    if not pred_specs:
        raise ValueError("At least one --pred NAME=DIR is required")

    cases = [strip_nii_suffix(i) for i in args.cases] if args.cases else auto_cases(gt_dir, pred_specs, args.num_cases)
    axis = AXIS_TO_INDEX[args.axis]
    column_titles = [args.modality.upper(), "GT", *[name for name, _ in pred_specs]]

    nrows = len(cases)
    ncols = len(column_titles)
    fig_width = max(2.0 * ncols, 7)
    fig_height = max(2.0 * nrows, 2.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    used_slices = {}
    for row_idx, case_id in enumerate(cases):
        panels, slice_index = build_case_data(
            case_id=case_id,
            image_dir=image_dir,
            gt_dir=gt_dir,
            pred_specs=pred_specs,
            modality=args.modality,
            axis=axis,
            requested_slice=args.slice,
            crop=not args.no_crop,
            crop_margin=args.crop_margin,
        )
        used_slices[case_id] = slice_index

        for col_idx, ax in enumerate(axes[row_idx]):
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=10, pad=6)
            if col_idx == 0:
                ax.imshow(panels[0], cmap="gray", vmin=0, vmax=1)
                ax.text(
                    -0.06,
                    0.5,
                    case_id,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
            else:
                overlay_seg(
                    ax,
                    panels[0],
                    panels[col_idx],
                    alpha=args.alpha,
                    contour=not args.no_contour,
                )

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[1], edgecolor=LABEL_COLORS[1], label="Edema / WT outer"),
        Patch(facecolor=LABEL_COLORS[2], edgecolor=LABEL_COLORS[2], label="Tumor core"),
        Patch(facecolor=LABEL_COLORS[3], edgecolor=LABEL_COLORS[3], label="Enhancing tumor"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.25, h_pad=0.4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {out_path}")
    print("Used slices:")
    for case_id, slice_index in used_slices.items():
        print(f"  {case_id}: {args.axis} slice {slice_index}")


if __name__ == "__main__":
    main()
