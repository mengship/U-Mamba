#!/usr/bin/env python3
"""
Measure parameter count, single-patch forward time, and peak CUDA memory.

This script builds models through nnU-Net trainer classes so that patch size,
input channels, and architecture settings match the selected dataset/plans.

Examples:
    python umamba/0607/measure_model_complexity.py \
        --dataset 705 \
        --trainers nnUNetTrainerUMambaEnc_150epochs \
                   nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
        --csv /hy-tmp/model_complexity.csv

Optional FLOPs:
    python umamba/0607/measure_model_complexity.py --dataset 705 --trainers ... --flops

Notes:
    - Forward time is measured on one nnU-Net patch, not full sliding-window volume inference.
    - FLOPs may be unavailable or inaccurate for custom Mamba/SS2D ops.
"""

import argparse
import csv
import importlib
import time
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


DEFAULT_TRAINERS = [
    "nnUNetTrainerUMambaEnc_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs",
]


def import_trainer_class(trainer_name: str):
    module = importlib.import_module(f"nnunetv2.training.nnUNetTrainer.{trainer_name}")
    return getattr(module, trainer_name)


def load_dataset_json(dataset_name: str, preprocessed_folder: Path) -> dict:
    candidates = [
        preprocessed_folder / "dataset.json",
        Path(nnUNet_raw) / dataset_name / "dataset.json",
    ]
    for path in candidates:
        if path.exists():
            return load_json(str(path))
    raise FileNotFoundError(f"dataset.json not found. Checked: {candidates}")


def build_model(trainer_name: str, dataset: str, plans_name: str, config_name: str, deep_supervision: bool):
    dataset_name = maybe_convert_to_dataset_name(dataset)
    preprocessed_folder = Path(nnUNet_preprocessed) / dataset_name
    plans_file = preprocessed_folder / f"{plans_name}.json"
    if not plans_file.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_file}")

    plans_manager = PlansManager(str(plans_file))
    configuration_manager = plans_manager.get_configuration(config_name)
    dataset_json = load_dataset_json(dataset_name, preprocessed_folder)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

    trainer_cls = import_trainer_class(trainer_name)
    model = trainer_cls.build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision=deep_supervision,
    )

    patch_size = tuple(int(i) for i in configuration_manager.patch_size)
    return model, num_input_channels, patch_size, dataset_name


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward(model, dummy, device: torch.device, warmup: int, repeats: int) -> tuple[float, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        sync_if_cuda(device)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy)
        sync_if_cuda(device)
        elapsed = time.perf_counter() - start

    avg_time = elapsed / max(repeats, 1)
    peak_memory_gb = None
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return avg_time, peak_memory_gb


def try_measure_flops(model, dummy, enabled: bool):
    if not enabled:
        return None
    try:
        from thop import profile
    except Exception as e:
        print(f"FLOPs skipped: cannot import thop ({e})")
        return None

    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except Exception as e:
        print(f"FLOPs skipped: thop failed ({e})")
        return None


def format_optional(value, scale=1.0, digits=6):
    if value is None:
        return ""
    return f"{value / scale:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset id or name, for example 705 or Dataset705_BraTS2020")
    parser.add_argument("--trainers", nargs="+", default=DEFAULT_TRAINERS)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--deep-supervision", action="store_true", help="Measure training-style deep supervision output")
    parser.add_argument("--flops", action="store_true", help="Try to compute FLOPs with thop")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = []

    for trainer in args.trainers:
        print(f"\nMeasuring {trainer}")
        model, in_channels, patch_size, dataset_name = build_model(
            trainer,
            args.dataset,
            args.plans,
            args.config,
            args.deep_supervision,
        )
        model = model.to(device)

        dummy_shape = (args.batch_size, in_channels, *patch_size)
        dummy = torch.randn(dummy_shape, device=device)

        total_params, trainable_params = count_params(model)
        avg_time_s, peak_memory_gb = measure_forward(model, dummy, device, args.warmup, args.repeats)
        flops = try_measure_flops(model, dummy, args.flops)

        row = {
            "trainer": trainer,
            "dataset": dataset_name,
            "config": args.config,
            "input_shape": "x".join(str(i) for i in dummy_shape),
            "params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "flops_g": None if flops is None else flops / 1e9,
            "forward_time_s": avg_time_s,
            "peak_memory_gb": peak_memory_gb,
        }
        rows.append(row)

        print(f"  Input shape: {row['input_shape']}")
        print(f"  Params/M: {row['params_m']:.6f}")
        print(f"  Trainable Params/M: {row['trainable_params_m']:.6f}")
        print(f"  FLOPs/G: {format_optional(flops, 1e9)}")
        print(f"  Forward time/s: {row['forward_time_s']:.6f}")
        print(f"  Peak memory/GB: {format_optional(peak_memory_gb)}")

        del model, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("| Trainer | Params/M | FLOPs/G | Forward time/s | Peak memory/GB |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['trainer']} | "
            f"{row['params_m']:.6f} | "
            f"{format_optional(row['flops_g'])} | "
            f"{row['forward_time_s']:.6f} | "
            f"{format_optional(row['peak_memory_gb'])} |"
        )

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "trainer",
            "dataset",
            "config",
            "input_shape",
            "params_m",
            "trainable_params_m",
            "flops_g",
            "forward_time_s",
            "peak_memory_gb",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
# python umamba/0607/measure_model_complexity.py \
#  --dataset 705 \
#  --trainers \
#    nnUNetTrainerUMambaEnc_150epochs \
#    nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs \
#  --csv /hy-tmp/model_complexity_fold0.csv