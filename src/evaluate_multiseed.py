"""Retrain and summarize the four-seed stability experiment (Appendix A2).

This is intentionally separate from the quick frozen-artifact verification.
Neural-network retraining is slow and may not be bit-wise identical across
hardware backends; the expected scientific comparison is stored in
``results/multiseed_metrics.csv``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 123, 2024, 3407]
MODEL_ORDER = ["BLSTM-XGB", "CNN-BLSTM", "BLSTM-SA", "LCA-Stacking"]
METRICS = ["R2", "MSE", "RMSE", "MAE", "MAPE_percent"]


def train_seed(seed: int, output_root: Path, smoke: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, check=True)


def table_paths(seed_root: Path, smoke: bool) -> dict[str, tuple[str, Path]]:
    suffix = "_smoke" if smoke else ""
    external_suffix = "_smoke" if smoke else ""
    result_root = seed_root / "results"
    return {
        "Table6": ("Tout_C", result_root / f"Table6{suffix}.csv"),
        "Table7": ("log1p_Q_J", result_root / f"Table7{suffix}.csv"),
        "Table8": (
            "Tout_C_external",
            result_root / f"external_mainline{external_suffix}" / "Table8.csv",
        ),
        "Table9": (
            "log1p_Q_J_external",
            result_root / f"external_mainline{external_suffix}" / "Table9.csv",
        ),
    }


def summarize(seeds: list[int], root: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        for table, (target, path) in table_paths(root / f"seed_{seed}", smoke).items():
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "target", target)
            frame.insert(0, "table", table)
            frames.append(frame)
    per_seed = pd.concat(frames, ignore_index=True)
    long = per_seed.melt(
        id_vars=["table", "target", "seed", "model"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long.groupby(["table", "target", "model", "metric"], sort=False)["value"]
        .agg(mean="mean", sd="std", n="count", min="min", max="max")
        .reset_index()
    )
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "multiseed"
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a sample SD")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for seed in seeds:
            print(f"Training seed {seed} ...", flush=True)
            train_seed(seed, output_root / f"seed_{seed}", args.smoke)

    per_seed, summary = summarize(seeds, output_root, args.smoke)
    suffix = "_smoke" if args.smoke else ""
    per_seed.to_csv(output_root / f"multiseed_per_seed{suffix}.csv", index=False)
    summary.to_csv(output_root / f"multiseed_metrics{suffix}.csv", index=False)
    print(f"Four-seed summary: {output_root / f'multiseed_metrics{suffix}.csv'}")


if __name__ == "__main__":
    main()
