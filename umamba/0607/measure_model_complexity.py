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

Optional MACs/FLOPs:
    python umamba/0607/measure_model_complexity.py --dataset 705 --trainers ... --flops

Notes:
    - Forward time is measured on one nnU-Net patch, not full sliding-window volume inference.
    - The fvcore backend follows the VMamba-style custom FLOPs handler for selective scan.
    - THOP reports MACs (multiply-accumulate operations). Estimated FLOPs are reported as 2 * MACs.
    - MACs/FLOPs may be unavailable or inaccurate for custom Mamba/SS2D selective-scan ops.
"""

import argparse
import csv
import importlib
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
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


def _num_elements(x) -> int:
    if torch.is_tensor(x):
        return x.numel()
    if isinstance(x, (list, tuple)):
        return sum(_num_elements(i) for i in x)
    return 0


def measure_supported_macs_with_hooks(model, dummy, enabled: bool, breakdown_topk: int = 0):
    """Count supported MACs with explicit hooks.

    This intentionally counts only common dense ops (Conv/ConvTranspose/Linear).
    It is more transparent than THOP for debugging, but it still misses custom
    selective-scan kernels used by Mamba/SS2D.
    """
    if not enabled:
        return None, []

    module_names = {module: name for name, module in model.named_modules()}
    records = []
    handles = []

    def add_record(module, macs):
        if macs <= 0:
            return
        records.append((
            module_names.get(module, module.__class__.__name__),
            module.__class__.__name__,
            int(macs),
        ))

    def conv_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return
        kernel_ops = int(module.weight[0].numel())
        output_elements = int(output.numel())
        add_record(module, output_elements * kernel_ops)

    def conv_transpose_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        kernel_spatial = 1
        for k in module.kernel_size:
            kernel_spatial *= int(k)
        output_channels_per_group = module.out_channels // module.groups
        input_positions = int(x.shape[0] * x.shape[1] * torch.tensor(x.shape[2:]).prod().item())
        add_record(module, input_positions * output_channels_per_group * kernel_spatial)

    def linear_hook(module, inputs, output):
        add_record(module, _num_elements(output) * int(module.in_features))

    for module in model.modules():
        if isinstance(module, _ConvTransposeNd):
            handles.append(module.register_forward_hook(conv_transpose_hook))
        elif isinstance(module, _ConvNd):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    if was_training:
        model.train()

    for handle in handles:
        handle.remove()

    total_macs = sum(i[2] for i in records)
    records.sort(key=lambda i: i[2], reverse=True)
    if breakdown_topk > 0:
        records = records[:breakdown_topk]
    else:
        records = []
    return total_macs, records


def try_measure_macs_thop(model, dummy, enabled: bool):
    if not enabled:
        return None
    try:
        from thop import profile
    except Exception as e:
        print(f"MACs skipped: cannot import thop ({e})")
        return None

    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs
    except Exception as e:
        print(f"MACs skipped: thop failed ({e})")
        return None


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False):
    """VMamba-style FLOPs estimate for selective scan."""
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs, verbose=False):
    """fvcore JIT handle for selective scan custom autograd ops."""
    if verbose:
        names = []
        for i in range(min(len(inputs), 10)):
            try:
                names.append(inputs[i].debugName())
            except Exception:
                pass
        print("selective_scan input params:", " ".join(names), flush=True)

    try:
        B, D, L = inputs[0].type().sizes()
        N = inputs[2].type().sizes()[1]
        if None in (B, D, L, N):
            return 0
        return flops_selective_scan_fn(B=int(B), L=int(L), D=int(D), N=int(N), with_D=True, with_Z=False)
    except Exception as e:
        if verbose:
            print(f"selective_scan FLOPs skipped: {e}", flush=True)
        return 0


def try_measure_flops_fvcore(model, dummy, enabled: bool, verbose: bool = False):
    if not enabled:
        return None, {}
    try:
        from fvcore.nn import flop_count
    except Exception as e:
        print(f"FLOPs skipped: cannot import fvcore ({e})")
        return None, {}

    supported_ops = {
        # Following VMamba's flops helper: ignore inexpensive elementwise and
        # rearrangement ops so the report focuses on dense ops and scan kernels.
        "aten::silu": None,
        "aten::neg": None,
        "aten::exp": None,
        "aten::flip": None,
        "aten::tanh": None,
        "aten::sigmoid": None,
        "aten::softmax": None,
        "aten::mean": None,
        "aten::sum": None,
        "aten::add": None,
        "aten::mul": None,
        "aten::div": None,
        "aten::sub": None,
        "aten::cat": None,
        "aten::permute": None,
        "aten::transpose": None,
        "aten::view": None,
        "aten::reshape": None,
        "aten::contiguous": None,
        "aten::expand": None,
        "aten::slice": None,
        "aten::select": None,
        "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, verbose=verbose),
        "prim::PythonOp.SelectiveScanFn": partial(selective_scan_flop_jit, verbose=verbose),
        "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, verbose=verbose),
        "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, verbose=verbose),
    }

    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            gflops_by_op, unsupported = flop_count(model=model, inputs=(dummy,), supported_ops=supported_ops)
            flops = sum(gflops_by_op.values()) * 1e9
        except Exception as e:
            print(f"FLOPs skipped: fvcore failed ({e})")
            return None, {}
    if was_training:
        model.train()

    if verbose:
        print("  fvcore FLOPs by op:")
        for op_name, value in sorted(gflops_by_op.items(), key=lambda item: item[1], reverse=True):
            print(f"    {value:.6f} G  {op_name}")
        if unsupported:
            print("  fvcore unsupported ops:")
            for op_name, count in unsupported.items():
                print(f"    {op_name}: {count}")

    return flops, dict(unsupported)


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
    parser.add_argument("--flops", action="store_true", help="Compute MACs and estimated FLOPs")
    parser.add_argument(
        "--flops-backend",
        choices=("hook", "thop", "fvcore"),
        default="hook",
        help="hook/thop report MACs; fvcore reports FLOPs and uses a custom selective-scan handler",
    )
    parser.add_argument("--breakdown-topk", type=int, default=0, help="Print top-k MAC contributors for hook backend")
    parser.add_argument("--flops-verbose", action="store_true", help="Print fvcore per-op FLOPs and unsupported ops")
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
        macs = None
        flops = None
        mac_records = []
        unsupported_ops = {}
        if args.flops and args.flops_backend == "hook":
            macs, mac_records = measure_supported_macs_with_hooks(model, dummy, True, args.breakdown_topk)
            flops = None if macs is None else 2 * macs
        elif args.flops and args.flops_backend == "thop":
            macs = try_measure_macs_thop(model, dummy, True)
            flops = None if macs is None else 2 * macs
        elif args.flops:
            flops, unsupported_ops = try_measure_flops_fvcore(model, dummy, True, args.flops_verbose)

        row = {
            "trainer": trainer,
            "dataset": dataset_name,
            "config": args.config,
            "flops_backend": args.flops_backend if args.flops else "",
            "input_shape": "x".join(str(i) for i in dummy_shape),
            "params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "macs_g": None if macs is None else macs / 1e9,
            "flops_g": None if flops is None else flops / 1e9,
            "forward_time_s": avg_time_s,
            "peak_memory_gb": peak_memory_gb,
        }
        rows.append(row)

        print(f"  Input shape: {row['input_shape']}")
        print(f"  Params/M: {row['params_m']:.6f}")
        print(f"  Trainable Params/M: {row['trainable_params_m']:.6f}")
        print(f"  MACs/G: {format_optional(macs, 1e9)}")
        print(f"  FLOPs/G: {format_optional(flops, 1e9)}")
        print(f"  Forward time/s: {row['forward_time_s']:.6f}")
        print(f"  Peak memory/GB: {format_optional(peak_memory_gb)}")
        if mac_records:
            print("  Top MAC contributors:")
            for name, module_type, module_macs in mac_records:
                print(f"    {module_macs / 1e9:.6f} G  {name} ({module_type})")
        if unsupported_ops and not args.flops_verbose:
            top_unsupported = sorted(unsupported_ops.items(), key=lambda item: item[1], reverse=True)[:10]
            print("  fvcore unsupported ops (top 10):")
            for op_name, count in top_unsupported:
                print(f"    {op_name}: {count}")

        del model, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("| Trainer | Params/M | MACs/G | FLOPs/G | Forward time/s | Peak memory/GB |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['trainer']} | "
            f"{row['params_m']:.6f} | "
            f"{format_optional(row['macs_g'])} | "
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
            "flops_backend",
            "input_shape",
            "params_m",
            "trainable_params_m",
            "macs_g",
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
