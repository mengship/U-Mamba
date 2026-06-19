# BraTS Visualization Script - Issues Fixed

## Summary
Fixed multiple issues in `visualize_brats_predictions.py` to improve robustness and error reporting.

## Issues Found and Fixed

### 1. Missing Dependency ❌→✅
**Problem**: `nibabel` package not installed
**Fix**: Added `nibabel>=5.0.0` to `umamba/scripts/requirements.txt`
**Action Required**: Run `pip install nibabel` or `pip install -r umamba/scripts/requirements.txt`

### 2. No Shape Validation ❌→✅
**Problem**: Script didn't validate that MRI images, GT, and predictions have matching shapes
**Fix**: Added shape validation in `build_case_data()` function (lines 283-290)
- Now raises `ValueError` with clear message if shapes don't match
- Validates all volumes before processing

### 3. Poor Error Messages ❌→✅
**Problem**: Generic error messages made debugging difficult
**Fix**: Enhanced error reporting in multiple functions:
- `auto_cases()`: Now shows which prediction directories are missing files
- `auto_cases()`: Warns when no overlap found between GT and predictions
- `auto_cases()`: Shows helpful message with case counts
- `build_case_data()`: Validates slice indices
- `main()`: Checks directory existence before processing
- `main()`: Added progress indicators for each case

### 4. Silent Failures ❌→✅
**Problem**: Errors during case processing weren't clearly identified
**Fix**: Added try-except block with case-specific error messages in `main()`
- Shows which case failed
- Displays progress: "Processing case 1/3: BraTS20_Training_011"

## Code Changes

### In `build_case_data()` (lines 270-306)
```python
# Added shape validation
expected_shape = image.shape
if gt.shape != expected_shape:
    raise ValueError(f"Shape mismatch for {case_id}: image {image.shape} vs GT {gt.shape}")
for i, pred in enumerate(preds):
    if pred.shape != expected_shape:
        pred_name = pred_specs[i][0]
        raise ValueError(f"Shape mismatch for {case_id}: image {image.shape} vs {pred_name} {pred.shape}")

# Added slice index validation
if not (0 <= slice_index < image.shape[axis]):
    raise ValueError(f"Invalid slice index {slice_index} for axis {axis} with shape {image.shape}")
```

### In `auto_cases()` (lines 182-207)
```python
# Added better diagnostics
for name, pred_dir in pred_specs:
    pred_cases = {strip_nii_suffix(p) for p in pred_dir.glob("*.nii*")}
    if not pred_cases:
        raise FileNotFoundError(f"No prediction .nii/.nii.gz files found in {pred_dir} for {name}")
    
    before = len(common)
    common &= pred_cases
    if len(common) == 0 and before > 0:
        missing = gt_cases - pred_cases
        print(f"Warning: No overlap between GT and {name}. Examples of missing cases: {sorted(missing)[:5]}")
```

### In `main()` (lines 310-424)
```python
# Added directory validation
if not image_dir.exists():
    raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
if not gt_dir.exists():
    raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")

for name, pred_dir in pred_specs:
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory for '{name}' does not exist: {pred_dir}")

# Added progress reporting
print(f"Processing {len(cases)} cases: {', '.join(cases)}")

for row_idx, case_id in enumerate(cases):
    print(f"Processing case {row_idx + 1}/{len(cases)}: {case_id}")
    try:
        # ... processing ...
    except Exception as e:
        print(f"Error processing case {case_id}: {e}")
        raise
```

## Testing

To test the fixes, first install the missing dependency:
```bash
pip install nibabel
```

Then run the visualization:
```bash
python umamba/0607/visualize_brats_predictions.py \
  --image-dir /hy-tmp/U-Mamba/data/nnUNet_raw/Dataset705_BraTS2020/imagesTr \
  --gt-dir /hy-tmp/U-Mamba/data/nnUNet_preprocessed/Dataset705_BraTS2020/gt_segmentations \
  --pred "U-Mamba=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEnc_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --pred "Ours=/hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020/nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
  --num-cases 3 \
  --out /hy-tmp/brats_visualization.png
```

## Expected Output

The script will now:
1. Validate all directories exist
2. Find common cases between GT and predictions
3. Show progress: "Found N common cases, selecting first 3"
4. For each case: "Processing case 1/3: BraTS20_Training_XXX"
5. Validate shapes match between volumes
6. Generate the visualization
7. Save to output path with summary of slices used

## Script Features

The script supports:
- Multiple prediction models (via repeated `--pred` flags)
- Different MRI modalities (t1, t1ce, t2, flair)
- Different slice views (axial, sagittal, coronal)
- Auto slice selection (max tumor area) or manual slice index
- Optional tumor-centered cropping
- Configurable overlay alpha and contours
- High-resolution output (default 300 DPI)
