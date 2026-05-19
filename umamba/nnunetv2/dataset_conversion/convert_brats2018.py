import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_segmentation(src_file: Path, dst_file: Path) -> None:
    img = sitk.ReadImage(str(src_file))
    arr = sitk.GetArrayFromImage(img)

    unique_values = np.unique(arr)
    for value in unique_values:
        if value not in [0, 1, 2, 4]:
            raise RuntimeError(f'unexpected label {value} in {src_file}')

    new_arr = np.zeros_like(arr)
    new_arr[arr == 4] = 3
    new_arr[arr == 2] = 1
    new_arr[arr == 1] = 2

    out = sitk.GetImageFromArray(new_arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst_file))


def collect_cases(src_root: Path):
    cases = []
    for grade in ['HGG', 'LGG']:
        grade_dir = src_root / grade
        if not grade_dir.is_dir():
            continue
        for case_dir in sorted(grade_dir.iterdir()):
            if case_dir.is_dir():
                cases.append((case_dir.name, case_dir))
    return cases


def convert_brats2018(src_root: Path, task_id: int, task_name: str) -> tuple[int, list[str]]:
    folder_name = f'Dataset{task_id:03d}_{task_name}'
    out_base = Path(nnUNet_raw) / folder_name
    images_tr = out_base / 'imagesTr'
    labels_tr = out_base / 'labelsTr'
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(src_root)
    skipped = []
    converted = 0

    for case_name, case_dir in cases:
        t1 = case_dir / f'{case_name}_t1.nii.gz'
        t1ce = case_dir / f'{case_name}_t1ce.nii.gz'
        t2 = case_dir / f'{case_name}_t2.nii.gz'
        flair = case_dir / f'{case_name}_flair.nii.gz'
        seg = case_dir / f'{case_name}_seg.nii.gz'

        required_files = [t1, t1ce, t2, flair, seg]
        if not all(path.is_file() for path in required_files):
            skipped.append(case_name)
            continue

        shutil.copy2(t1, images_tr / f'{case_name}_0000.nii.gz')
        shutil.copy2(t1ce, images_tr / f'{case_name}_0001.nii.gz')
        shutil.copy2(t2, images_tr / f'{case_name}_0002.nii.gz')
        shutil.copy2(flair, images_tr / f'{case_name}_0003.nii.gz')
        convert_segmentation(seg, labels_tr / f'{case_name}.nii.gz')
        converted += 1

    generate_dataset_json(
        out_base,
        channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
        labels={
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': (3,),
        },
        num_training_cases=converted,
        file_ending='.nii.gz',
        regions_class_order=(1, 2, 3),
        dataset_name=task_name,
        description='BraTS2018 converted to nnU-Net format',
    )

    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert BraTS2018 to nnU-Net format.')
    parser.add_argument(
        '--src-root',
        required=True,
        help='BraTS2018 raw training root, for example /hy-tmp/BraTS2018_raw/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training',
    )
    parser.add_argument('--task-id', type=int, default=705, help='nnU-Net dataset id, default: 705')
    parser.add_argument('--task-name', default='BraTS2018', help='nnU-Net dataset name, default: BraTS2018')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)

    if not src_root.is_dir():
        raise FileNotFoundError(f'source root not found: {src_root}')

    converted, skipped = convert_brats2018(src_root, args.task_id, args.task_name)
    print(f'Converted: {converted}')
    print(f'Skipped: {len(skipped)}')
    if skipped:
        print(f'First skipped: {skipped[:10]}')


if __name__ == '__main__':
    main()