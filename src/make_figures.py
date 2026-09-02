"""Generate the scientific post-processing figures retained in the repository."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures"
MODEL_LABELS = {
    "BLSTM_XGB": "BLSTM-XGB",
    "CNN_BLSTM": "CNN-BLSTM",
    "BLSTM_SA": "BLSTM-SA",
    "LCA_Stacking": "LCA-Stacking",
}


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def external_predictions(output: Path) -> None:
    data = pd.read_csv(SOURCE / "external_predictions.csv.gz")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    axes[0].plot(data["time_min"], data["true_Tout_C"], color="black", lw=2, label="Reference")
    axes[1].plot(data["time_min"], data["true_Q_cum_J"] / 1e6, color="black", lw=2, label="Reference")
    for slug, label in MODEL_LABELS.items():
        axes[0].plot(data["time_min"], data[f"pred_Tout_C__{slug}"], lw=1.1, label=label)
        axes[1].plot(data["time_min"], data[f"pred_Q_cum_J__{slug}"] / 1e6, lw=1.1, label=label)
    axes[0].set_ylabel("Outlet temperature (°C)")
    axes[1].set_ylabel("Cumulative heat (MJ)")
    axes[1].set_xlabel("Time (min)")
    axes[0].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, output / "external_predictions.png")


def loss_curves(output: Path) -> None:
    data = pd.read_csv(SOURCE / "figures" / "loss_curves.csv")
    targets = [("Tout_C", "Outlet temperature"), ("logQ", "log1p cumulative heat")]
    base_models = [("lstm", "BiLSTM"), ("cnn", "CNN"), ("sa", "SA")]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharex=True)
    for row, (target, target_label) in enumerate(targets):
        for col, (base_model, model_label) in enumerate(base_models):
            axis = axes[row, col]
            group = data.loc[
                data["target"].eq(target) & data["base_model"].eq(base_model)
            ]
            summary = group.groupby("epoch")[["training_loss", "validation_loss"]].agg(["mean", "std"])
            epochs = summary.index.to_numpy()
            for variable, label, color in [
                ("training_loss", "Training", "#2563EB"),
                ("validation_loss", "Validation", "#DC2626"),
            ]:
                mean = summary[(variable, "mean")].to_numpy()
                sd = summary[(variable, "std")].fillna(0).to_numpy()
                axis.plot(epochs, mean, label=label, color=color)
                axis.fill_between(epochs, np.maximum(mean - sd, 0), mean + sd, color=color, alpha=0.15)
            axis.set_title(f"{target_label}: {model_label}", fontsize=9)
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
            if row == 1:
                axis.set_xlabel("Epoch")
            if col == 0:
                axis.set_ylabel("MSE loss")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, output / "loss_curves.png")


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.pivot(
        index="query_time_step", columns="key_time_step", values="mean_attention_score"
    ).sort_index().sort_index(axis=1).to_numpy()


def attention_overall(output: Path) -> None:
    data = pd.read_csv(SOURCE / "figures" / "attention_overall.csv")
    targets = [("Tout_C", "Outlet temperature"), ("logQ", "log1p cumulative heat")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for axis, (target, label) in zip(axes, targets):
        image = axis.imshow(matrix(data.loc[data["target"].eq(target)]), origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("Key time step")
        axis.set_ylabel("Query time step")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, output / "attention_overall.png")


def attention_stages(output: Path) -> None:
    data = pd.read_csv(SOURCE / "figures" / "attention_stages.csv")
    targets = [("Tout_C", "Outlet temperature"), ("logQ", "log1p cumulative heat")]
    stages = ["Early", "Late"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=True, sharey=True)
    for row, (target, target_label) in enumerate(targets):
        for col, stage in enumerate(stages):
            axis = axes[row, col]
            subset = data.loc[data["target"].eq(target) & data["stage"].eq(stage)]
            image = axis.imshow(matrix(subset), origin="lower", aspect="auto", cmap="viridis")
            axis.set_title(f"{target_label}: {stage}")
            axis.set_xlabel("Key time step")
            axis.set_ylabel("Query time step")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, output / "attention_stages.png")


def run(output: Path = DEFAULT_OUTPUT) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    external_predictions(output)
    loss_curves(output)
    attention_overall(output)
    attention_stages(output)
    return sorted(output.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = run(args.output_dir.resolve())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
