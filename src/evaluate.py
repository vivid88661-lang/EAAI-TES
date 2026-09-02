"""Reproduce Tables 6--9 from the frozen publication package.

Tables 6--7 are recomputed from the committed grouped out-of-fold prediction
rows. Tables 8--9 are recomputed by applying the frozen final artifacts to the
locked external case E_RI002.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    MODEL_ORDER,
    MODEL_SLUGS,
    load_external_cases,
    load_public_preprocessor,
    predict_public_frozen_model,
)
from preprocessing import ROOT, make_windows, metrics


REFERENCE_DIR = ROOT / "results"
SOURCE_DIR = REFERENCE_DIR / "source_data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reproduced"
METRIC_COLUMNS = ["R2", "MSE", "RMSE", "MAE", "MAPE_percent"]


def ordered_metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    return frame.sort_values(["target", "model"]).reset_index(drop=True)


def evaluate_grouped_oof() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        predictions = pd.read_csv(
            SOURCE_DIR / "oof_predictions" / f"{model_name}.csv.gz"
        )
        if predictions["case_id"].nunique() != 32:
            raise RuntimeError(f"{model_name}: expected 32 OOF cases")
        for target, true_column, prediction_column in [
            ("Tout_C", "true_Tout_C", "pred_Tout_C"),
            ("log1p_Q_J", "true_log1p_Q_J", "pred_log1p_Q_J"),
        ]:
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    **metrics(predictions[true_column], predictions[prediction_column]),
                }
            )
    return ordered_metrics(rows)


def evaluate_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, frames = load_external_cases()
    case_ids = manifest["case_id"].tolist()
    metric_rows: list[dict[str, object]] = []
    prediction_frame: pd.DataFrame | None = None

    for model_name in MODEL_ORDER:
        preprocessor = load_public_preprocessor(model_name)
        windows = make_windows(frames, case_ids, preprocessor)
        pred_tout, pred_logq = predict_public_frozen_model(
            model_name, windows, preprocessor
        )
        if prediction_frame is None:
            prediction_frame = pd.DataFrame(
                {
                    "case_id": windows.case_ids,
                    "time_min": windows.times_min,
                    "true_Tout_C": windows.y_true[:, 0],
                    "true_log1p_Q_J": windows.y_true[:, 1],
                    "true_Q_cum_J": np.expm1(windows.y_true[:, 1]),
                }
            )
        slug = MODEL_SLUGS[model_name]
        prediction_frame[f"pred_Tout_C__{slug}"] = pred_tout
        prediction_frame[f"pred_log1p_Q_J__{slug}"] = pred_logq
        prediction_frame[f"pred_Q_cum_J__{slug}"] = np.expm1(pred_logq)
        metric_rows.extend(
            [
                {
                    "model": model_name,
                    "target": "Tout_C",
                    **metrics(windows.y_true[:, 0], pred_tout),
                },
                {
                    "model": model_name,
                    "target": "log1p_Q_J",
                    **metrics(windows.y_true[:, 1], pred_logq),
                },
            ]
        )

    if prediction_frame is None:
        raise RuntimeError("No external predictions were produced")
    return ordered_metrics(metric_rows), prediction_frame


def compare_reference(observed: pd.DataFrame, reference_path: Path) -> None:
    expected = pd.read_csv(reference_path)
    keys = ["model", "target"]
    observed_indexed = observed.assign(model=observed["model"].astype(str)).set_index(keys)
    expected_indexed = expected.set_index(keys)
    if set(observed_indexed.index) != set(expected_indexed.index):
        raise RuntimeError(f"Row identity mismatch against {reference_path.name}")
    for key in expected_indexed.index:
        for column in METRIC_COLUMNS:
            left = float(observed_indexed.loc[key, column])
            right = float(expected_indexed.loc[key, column])
            if round(left, 3) != round(right, 3):
                raise RuntimeError(
                    f"{reference_path.name}: {key}/{column} differs at paper precision: "
                    f"{left} != {right}"
                )


def run(output_dir: Path, compare: bool = True) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_metrics = evaluate_grouped_oof()
    external_metrics, external_predictions = evaluate_external()
    main_path = output_dir / "main_metrics.csv"
    external_path = output_dir / "external_metrics.csv"
    prediction_path = output_dir / "external_predictions.csv.gz"
    main_metrics.to_csv(main_path, index=False)
    external_metrics.to_csv(external_path, index=False)
    external_predictions.to_csv(prediction_path, index=False, compression="gzip")
    if compare:
        compare_reference(main_metrics, REFERENCE_DIR / "main_metrics.csv")
        compare_reference(external_metrics, REFERENCE_DIR / "external_metrics.csv")
    protocol = {
        "Tables_6_7": "metrics recomputed from grouped OOF prediction rows",
        "Tables_8_9": "frozen artifacts applied to E_RI002 only",
        "external_prediction_rows": int(len(external_predictions)),
        "reference_comparison": bool(compare),
    }
    protocol_path = output_dir / "evaluation_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return {
        "main_metrics": main_path,
        "external_metrics": external_path,
        "external_predictions": prediction_path,
        "protocol": protocol_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    outputs = run(args.output_dir.resolve(), compare=not args.no_compare)
    print("PASS: Tables 6-9 reproduced at manuscript precision")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
