"""Reproduce the restricted-input baseline and sensor-noise robustness results.

Table 11 is the zero-noise row. Table 12 evaluates the same frozen model and
same seven held-out cases at 0%, 0.5% and 1.0% multiplicative Gaussian noise.
Noise is applied to the inlet-temperature input ``Tin_C`` only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from preprocessing import ROOT


MODEL_DIR = ROOT / "models" / "frozen_models" / "restricted_input"
SCALER_DIR = ROOT / "models" / "scalers"
SPLIT_PATH = ROOT / "models" / "splits" / "heldout_cases.json"
REFERENCE_PATH = ROOT / "results" / "robustness_metrics.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "reproduced" / "robustness_metrics.csv"
NOISE_LEVELS = (0.0, 0.005, 0.01)
SEED = 42
N_PAST = 30


def make_windows(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for end_index in range(N_PAST, len(dataset)):
        window = dataset[end_index - N_PAST : end_index]
        q_history = window[:-1, 2]
        delta_q = q_history[-1] - q_history[0]
        x_rows.append(
            np.concatenate(
                [window, np.full((N_PAST, 1), delta_q)], axis=1
            )
        )
        y_rows.append(dataset[end_index, [1, 2]])
    return np.asarray(x_rows), np.asarray(y_rows)


def load_frozen() -> dict[str, object]:
    return {
        "feature_scaler": joblib.load(SCALER_DIR / "scaler_features.joblib"),
        "temperature_scaler": joblib.load(SCALER_DIR / "scaler_tf_out.joblib"),
        "heat_transformer": joblib.load(SCALER_DIR / "qt.joblib"),
        "meta_temperature": joblib.load(MODEL_DIR / "meta_model_t.joblib"),
        "temperature_models": [
            load_model(MODEL_DIR / "final_lstm_t.keras", compile=False),
            load_model(MODEL_DIR / "final_cnn_t.keras", compile=False),
            load_model(MODEL_DIR / "final_sa_t.keras", compile=False, safe_mode=False),
        ],
    }


def evaluate_noise_level(
    noise_fraction: float,
    case_ids: list[str],
    frozen: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    case_rows: list[dict[str, float | str]] = []

    for case_id in case_ids:
        frame = pd.read_csv(ROOT / "data" / "development_cases" / f"{case_id}.csv.gz")
        if noise_fraction > 0:
            frame["Tin_C"] = frame["Tin_C"] * (
                1.0 + rng.normal(0.0, noise_fraction, size=len(frame))
            )

        q_scaled = frozen["heat_transformer"].transform(
            np.log1p(frame[["Q_cum_J"]].to_numpy())
        )
        restricted_features = frame[["Tin_C", "Ri_m", "L_m", "Phi"]].rename(
            columns={"Tin_C": "Tf_in_dch,C", "Ri_m": "Ri_", "L_m": "L"}
        )
        restricted_temperature = frame[["Tout_C"]].rename(
            columns={"Tout_C": "Tf_out_dch,C"}
        )
        feature_scaled = frozen["feature_scaler"].transform(restricted_features)
        temperature_scaled = frozen["temperature_scaler"].transform(
            restricted_temperature
        )
        combined = np.column_stack(
            [
                feature_scaled[:, 0],
                temperature_scaled[:, 0],
                q_scaled[:, 0],
                feature_scaled[:, 1],
                feature_scaled[:, 2],
                feature_scaled[:, 3],
            ]
        )
        windows, targets = make_windows(combined)

        base_predictions: list[np.ndarray] = []
        for model in frozen["temperature_models"]:
            output = model.predict(windows, batch_size=256, verbose=0)
            if isinstance(output, (list, tuple)):
                output = output[0]
            base_predictions.append(np.asarray(output).reshape(-1))
        prediction_scaled = frozen["meta_temperature"].predict(
            np.column_stack(base_predictions)
        )
        true_temperature = frozen["temperature_scaler"].inverse_transform(
            targets[:, 0].reshape(-1, 1)
        ).reshape(-1)
        predicted_temperature = frozen["temperature_scaler"].inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).reshape(-1)

        mse = float(mean_squared_error(true_temperature, predicted_temperature))
        case_rows.append(
            {
                "case_id": case_id,
                "R2": float(r2_score(true_temperature, predicted_temperature)),
                "MSE": mse,
                "MAE": float(mean_absolute_error(true_temperature, predicted_temperature)),
            }
        )

    mean_mse = float(np.mean([float(row["MSE"]) for row in case_rows]))
    return {
        "supports": (
            "Table 11 and Table 12 (0%)"
            if noise_fraction == 0
            else f"Table 12 ({noise_fraction * 100:.1f}%)"
        ),
        "noise_percent": 100.0 * noise_fraction,
        "case_count": len(case_rows),
        "R2": float(np.mean([float(row["R2"]) for row in case_rows])),
        "MSE": mean_mse,
        "RMSE": float(np.sqrt(mean_mse)),
        "MAE": float(np.mean([float(row["MAE"]) for row in case_rows])),
        "noise_variable": "Tin_C",
        "aggregation": "macro case mean; RMSE=sqrt(mean case-wise MSE)",
    }


def compare_reference(observed: pd.DataFrame) -> None:
    expected = pd.read_csv(REFERENCE_PATH)
    for row_index in range(len(expected)):
        for column in ["R2", "MSE", "RMSE", "MAE"]:
            left = float(observed.loc[row_index, column])
            right = float(expected.loc[row_index, column])
            if not np.isclose(left, right, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"Robustness mismatch at row {row_index}/{column}: {left} != {right}"
                )


def run(output_path: Path = DEFAULT_OUTPUT, compare: bool = True) -> Path:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    case_ids = split["case_ids_in_evaluation_order"]
    frozen = load_frozen()
    frame = pd.DataFrame(
        [evaluate_noise_level(level, case_ids, frozen) for level in NOISE_LEVELS]
    )
    if compare:
        compare_reference(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    path = run(args.output.resolve(), compare=not args.no_compare)
    print("PASS: Tables 11-12 reproduced from the frozen restricted-input model")
    print(path)


if __name__ == "__main__":
    main()
