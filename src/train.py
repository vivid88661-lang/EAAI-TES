"""Train the grouped baselines and LCA-Stacking publication pipeline.

Outputs are always written below ``outputs/`` (or an explicit ``--output-root``)
so retraining cannot overwrite the distributed frozen artifacts or reference
CSV files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-final-fit",
        action="store_true",
        help="produce grouped OOF Tables 6-7 only",
    )
    return parser.parse_args()


def aggregate_internal(results_dir: Path, model_order: list[str], smoke: bool) -> None:
    import pandas as pd

    suffix = "_smoke" if smoke else ""
    frames = [
        pd.read_csv(results_dir / f"{model}{suffix}" / "pooled_metrics.csv")
        for model in model_order
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics["model"] = pd.Categorical(metrics["model"], model_order, ordered=True)
    metrics = metrics.sort_values(["target", "model"]).reset_index(drop=True)
    metrics.to_csv(results_dir / f"all_models_pooled_metrics{suffix}.csv", index=False)
    columns = ["model", "R2", "MSE", "RMSE", "MAE", "MAPE_percent"]
    metrics.loc[metrics["target"].eq("Tout_C"), columns].to_csv(
        results_dir / f"Table6{suffix}.csv", index=False
    )
    metrics.loc[metrics["target"].eq("log1p_Q_J"), columns].to_csv(
        results_dir / f"Table7{suffix}.csv", index=False
    )


def evaluate_external(results_dir: Path, smoke: bool) -> None:
    import joblib
    import numpy as np
    import pandas as pd

    from models import (
        MODEL_ORDER,
        MODEL_SLUGS,
        artifact_root,
        external_result_root,
        load_external_cases,
        predict_frozen_model,
    )
    from preprocessing import make_windows, metrics

    manifest, frames = load_external_cases()
    case_ids = manifest["case_id"].tolist()
    output = external_result_root(smoke)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: pd.DataFrame | None = None
    for model_name in MODEL_ORDER:
        model_dir = artifact_root(smoke) / MODEL_SLUGS[model_name]
        preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        windows = make_windows(frames, case_ids, preprocessor)
        pred_t, pred_q = predict_frozen_model(model_name, windows, preprocessor, smoke)
        if predictions is None:
            predictions = pd.DataFrame(
                {
                    "case_id": windows.case_ids,
                    "time_min": windows.times_min,
                    "true_Tout_C": windows.y_true[:, 0],
                    "true_log1p_Q_J": windows.y_true[:, 1],
                    "true_Q_cum_J": np.expm1(windows.y_true[:, 1]),
                }
            )
        slug = MODEL_SLUGS[model_name]
        predictions[f"pred_Tout_C__{slug}"] = pred_t
        predictions[f"pred_log1p_Q_J__{slug}"] = pred_q
        predictions[f"pred_Q_cum_J__{slug}"] = np.expm1(pred_q)
        rows.extend(
            [
                {"model": model_name, "target": "Tout_C", **metrics(windows.y_true[:, 0], pred_t)},
                {"model": model_name, "target": "log1p_Q_J", **metrics(windows.y_true[:, 1], pred_q)},
            ]
        )
    if predictions is None:
        raise RuntimeError("No external predictions")
    metrics_frame = pd.DataFrame(rows)
    metrics_frame["model"] = pd.Categorical(
        metrics_frame["model"], MODEL_ORDER, ordered=True
    )
    metrics_frame = metrics_frame.sort_values(["target", "model"]).reset_index(drop=True)
    predictions.to_csv(output / "external_predictions.csv.gz", index=False, compression="gzip")
    metrics_frame.to_csv(output / "external_pooled_metrics.csv", index=False)
    columns = ["model", "R2", "MSE", "RMSE", "MAE", "MAPE_percent"]
    metrics_frame.loc[metrics_frame["target"].eq("Tout_C"), columns].to_csv(
        output / "Table8.csv", index=False
    )
    metrics_frame.loc[metrics_frame["target"].eq("log1p_Q_J"), columns].to_csv(
        output / "Table9.csv", index=False
    )


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repository_root / "outputs" / "retraining" / f"seed_{args.seed}"
    )
    results_dir = output_root / "results"
    artifacts_dir = output_root / "artifacts"
    os.environ["TES_TRAIN_SEED"] = str(args.seed)
    os.environ["TES_RESULTS_DIR"] = str(results_dir)
    os.environ["TES_ARTIFACTS_DIR"] = str(artifacts_dir)

    from models import MODEL_ORDER, finalize_artifact_manifest, fit_final_model
    from preprocessing import (
        build_blstm_sa,
        build_cnn_blstm,
        train_blstm_xgb,
        train_lca,
        train_neural_baseline,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    train_blstm_xgb(args.smoke)
    train_neural_baseline("CNN-BLSTM", build_cnn_blstm, 10, 128, args.smoke)
    train_neural_baseline("BLSTM-SA", build_blstm_sa, 10, 128, args.smoke)
    train_lca(args.smoke)
    aggregate_internal(results_dir, MODEL_ORDER, args.smoke)

    if not args.skip_final_fit:
        for model_name in MODEL_ORDER:
            fit_final_model(model_name, args.smoke)
        finalize_artifact_manifest(args.smoke)
        evaluate_external(results_dir, args.smoke)

    print(f"Training outputs: {output_root}")


if __name__ == "__main__":
    main()
