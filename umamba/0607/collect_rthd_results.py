#!/usr/bin/env python3
"""
Collect WT/TC/ET/Mean Dice from nnU-Net summary.json files.

Example:
    python umamba/0607/collect_rthd_results.py \
        --results-root /hy-tmp/U-Mamba/data/nnUNet_results/Dataset705_BraTS2020 \
        --folds 0 1 2 \
        --csv /tmp/rthd_dice.csv
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_TRAINERS = [
    "nnUNetTrainerUMambaEnc_150epochs",
    "nnUNetTrainerUMambaEncRTHD_EncoderOnly_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_SkipCalibration_150epochs",
    "nnUNetTrainerUMambaEncRTHD_StageAwareDecoder_Full_150epochs",
]


REGION_KEYS = {
    "WT": "(1, 2, 3)",
    "TC": "(2, 3)",
    "ET": "(3,)",
}


def read_summary(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    mean_metrics = data["mean"]
    return {
        "WT": mean_metrics[REGION_KEYS["WT"]]["Dice"],
        "TC": mean_metrics[REGION_KEYS["TC"]]["Dice"],
        "ET": mean_metrics[REGION_KEYS["ET"]]["Dice"],
        "Mean": data["foreground_mean"]["Dice"],
    }


def summary_path(results_root: Path, trainer: str, fold: int, plans: str, config: str) -> Path:
    return results_root / f"{trainer}__{plans}__{config}" / f"fold_{fold}" / "validation" / "summary.json"


def collect(results_root: Path, trainers: list[str], folds: list[int], plans: str, config: str) -> list[dict]:
    rows = []
    for fold in folds:
        for trainer in trainers:
            path = summary_path(results_root, trainer, fold, plans, config)
            if not path.exists():
                rows.append({
                    "trainer": trainer,
                    "fold": fold,
                    "WT": None,
                    "TC": None,
                    "ET": None,
                    "Mean": None,
                    "summary": str(path),
                    "status": "missing",
                })
                continue

            metrics = read_summary(path)
            rows.append({
                "trainer": trainer,
                "fold": fold,
                **metrics,
                "summary": str(path),
                "status": "ok",
            })
    return rows


def print_table(rows: list[dict]) -> None:
    print("| Trainer | Fold | WT | TC | ET | Mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["status"] != "ok":
            print(f"| {row['trainer']} | {row['fold']} | missing | missing | missing | missing |")
            continue
        print(
            f"| {row['trainer']} | {row['fold']} | "
            f"{row['WT']:.6f} | {row['TC']:.6f} | {row['ET']:.6f} | {row['Mean']:.6f} |"
        )


def print_fold_deltas(rows: list[dict], baseline_name: str) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    by_fold = {}
    for row in ok_rows:
        by_fold.setdefault(row["fold"], {})[row["trainer"]] = row

    print()
    print("Delta vs baseline (percentage points):")
    print("| Trainer | Fold | WT | TC | ET | Mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for fold, fold_rows in sorted(by_fold.items()):
        baseline = fold_rows.get(baseline_name)
        if baseline is None:
            continue
        for trainer, row in fold_rows.items():
            if trainer == baseline_name:
                continue
            print(
                f"| {trainer} | {fold} | "
                f"{(row['WT'] - baseline['WT']) * 100:+.4f} | "
                f"{(row['TC'] - baseline['TC']) * 100:+.4f} | "
                f"{(row['ET'] - baseline['ET']) * 100:+.4f} | "
                f"{(row['Mean'] - baseline['Mean']) * 100:+.4f} |"
            )


def print_averages(rows: list[dict]) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    by_trainer = {}
    for row in ok_rows:
        by_trainer.setdefault(row["trainer"], []).append(row)

    print()
    print("Average over available folds:")
    print("| Trainer | Folds | WT | TC | ET | Mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for trainer, trainer_rows in by_trainer.items():
        folds = ",".join(str(r["fold"]) for r in sorted(trainer_rows, key=lambda x: x["fold"]))
        print(
            f"| {trainer} | {folds} | "
            f"{mean(r['WT'] for r in trainer_rows):.6f} | "
            f"{mean(r['TC'] for r in trainer_rows):.6f} | "
            f"{mean(r['ET'] for r in trainer_rows):.6f} | "
            f"{mean(r['Mean'] for r in trainer_rows):.6f} |"
        )


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trainer", "fold", "WT", "TC", "ET", "Mean", "status", "summary"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Dataset result root, for example Dataset705_BraTS2020")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--trainers", nargs="+", default=DEFAULT_TRAINERS)
    parser.add_argument("--baseline", default="nnUNetTrainerUMambaEnc_150epochs")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    rows = collect(Path(args.results_root), args.trainers, args.folds, args.plans, args.config)
    print_table(rows)
    print_fold_deltas(rows, args.baseline)
    print_averages(rows)

    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
