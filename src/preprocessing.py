from __future__ import annotations

"""Case-wise preprocessing, model builders and grouped OOF training routines."""

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from xgboost import XGBRegressor
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.regularizers import l2


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "development_cases"
MANIFEST_PATH = ROOT / "data" / "metadata.csv"
RESULTS_DIR = Path(
    os.environ.get("TES_RESULTS_DIR", ROOT / "outputs" / "training_results")
).resolve()

N_PAST = 30
N_SPLITS = 5
SPLIT_SEED = 42
SEED = int(os.environ.get("TES_TRAIN_SEED", "42"))
TARGETS = ("Tout_C", "logQ")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@dataclass
class FoldPreprocessor:
    feature_scaler: MinMaxScaler
    t_scaler: MinMaxScaler
    q_transformer: QuantileTransformer

    @classmethod
    def fit(cls, frames: dict[str, pd.DataFrame], train_case_ids: list[str]) -> "FoldPreprocessor":
        train = pd.concat([frames[c] for c in train_case_ids], ignore_index=True)
        feature_cols = ["f_liquid", "Tin_C", "Ri_m", "L_m", "Phi"]
        feature_scaler = MinMaxScaler().fit(train[feature_cols].to_numpy(dtype=float))
        t_scaler = MinMaxScaler().fit(train[["Tout_C"]].to_numpy(dtype=float))
        n_quantiles = min(500, len(train))
        q_transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            random_state=SPLIT_SEED,
        ).fit(train[["log1p_Q_J"]].to_numpy(dtype=float))
        return cls(feature_scaler, t_scaler, q_transformer)

    def transform_case(self, frame: pd.DataFrame) -> np.ndarray:
        feature_cols = ["f_liquid", "Tin_C", "Ri_m", "L_m", "Phi"]
        features = self.feature_scaler.transform(frame[feature_cols].to_numpy(dtype=float))
        tout = self.t_scaler.transform(frame[["Tout_C"]].to_numpy(dtype=float)).reshape(-1)
        logq = self.q_transformer.transform(frame[["log1p_Q_J"]].to_numpy(dtype=float)).reshape(-1)
        return np.column_stack(
            [features[:, 0], features[:, 1], tout, logq, features[:, 2], features[:, 3], features[:, 4]]
        )

    def inverse_t(self, scaled: np.ndarray) -> np.ndarray:
        return self.t_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).reshape(-1)

    def inverse_logq(self, scaled: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(scaled).reshape(-1, 1), 0.0, 1.0)
        return self.q_transformer.inverse_transform(clipped).reshape(-1)


@dataclass
class WindowSet:
    X: np.ndarray
    y_scaled: np.ndarray
    y_true: np.ndarray
    case_ids: np.ndarray
    times_min: np.ndarray


def load_development_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = (
        manifest.loc[manifest["dataset_role"].eq("development")]
        .copy()
        .sort_values("case_id")
    )
    if len(manifest) != 32:
        raise RuntimeError(f"Expected 32 development cases, found {len(manifest)}")
    frames: dict[str, pd.DataFrame] = {}
    required = {
        "case_id", "time_min", "f_liquid", "Tin_C", "Tout_C", "Q_cum_J",
        "log1p_Q_J", "Ri_m", "L_m", "Phi",
    }
    for case_id in manifest["case_id"]:
        path = DATA_DIR / f"{case_id}.csv.gz"
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
        if not frame["case_id"].eq(case_id).all():
            raise RuntimeError(f"Case ID mismatch in {path.name}")
        frames[case_id] = frame.reset_index(drop=True)
    return manifest, frames


def make_windows(
    frames: dict[str, pd.DataFrame], case_ids: list[str], preprocessor: FoldPreprocessor
) -> WindowSet:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    groups: list[str] = []
    times: list[float] = []
    for case_id in case_ids:
        frame = frames[case_id]
        scaled = preprocessor.transform_case(frame)
        if len(frame) <= N_PAST:
            continue
        for index in range(N_PAST, len(frame)):
            window = scaled[index - N_PAST:index]
            q_history = window[:-1, 3]
            delta_q = float(q_history[-1] - q_history[0])
            xs.append(np.column_stack([window, np.full(N_PAST, delta_q)]))
            ys.append(scaled[index, [2, 3]])
            truths.append(frame.loc[index, ["Tout_C", "log1p_Q_J"]].to_numpy(dtype=float))
            groups.append(case_id)
            times.append(float(frame.loc[index, "time_min"]))
    return WindowSet(
        X=np.asarray(xs, dtype=np.float32),
        y_scaled=np.asarray(ys, dtype=np.float32),
        y_true=np.asarray(truths, dtype=np.float64),
        case_ids=np.asarray(groups),
        times_min=np.asarray(times, dtype=np.float64),
    )


def build_bilstm_xgb_lstm() -> Model:
    model = Sequential(
        [
            layers.Input((N_PAST, 8)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.2),
            layers.Dense(1),
        ],
        name="blstm_xgb_stage1",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    return model


def build_cnn_blstm(target_index: int) -> Model:
    if target_index == 0:
        units, kernel, dropout, lr, activation = 128, 3, 0.1, 0.001, "relu"
    else:
        units, kernel, dropout, lr, activation = 64, 5, 0.3, 0.001, "sigmoid"
    model = Sequential(
        [
            layers.Input((N_PAST, 8)),
            layers.Conv1D(64, kernel, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
            layers.Dropout(dropout),
            layers.Bidirectional(layers.LSTM(units)),
            layers.Dense(64, activation=activation, kernel_regularizer=l2(0.001)),
            layers.Dense(1),
        ],
        name=f"cnn_blstm_{TARGETS[target_index]}",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_blstm_sa(target_index: int) -> Model:
    if target_index == 0:
        units, layers_count, dropout, lr, activation = 128, 2, 0.1, 0.0005, "relu"
    else:
        units, layers_count, dropout, lr, activation = 64, 1, 0.2, 0.0005, "sigmoid"
    inputs = layers.Input((N_PAST, 8))
    x = inputs
    for _ in range(layers_count):
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
        x = layers.Dropout(dropout)(x)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=units, dropout=dropout)(x, x, x)
    x = layers.LayerNormalization()(x + attention)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation=activation, kernel_regularizer=l2(0.001))(x)
    output = layers.Dense(1)(x)
    model = Model(inputs, output, name=f"blstm_sa_{TARGETS[target_index]}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_lca_lstm(target_index: int) -> Model:
    layer_count = 2 if target_index == 0 else 1
    model = Sequential(name=f"lca_lstm_{TARGETS[target_index]}")
    model.add(layers.Input((N_PAST, 8)))
    for index in range(layer_count):
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=index < layer_count - 1)))
        model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_lca_cnn(_: int) -> Model:
    model = Sequential(
        [
            layers.Input((N_PAST, 8)),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ],
        name="lca_cnn",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_lca_sa(_: int) -> Model:
    inputs = layers.Input((N_PAST, 8))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x, x)
    x = layers.LayerNormalization()(x + attention)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1)(x)
    model = Model(inputs, output, name="lca_sa")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def inverse_prediction(pred_scaled: np.ndarray, preprocessor: FoldPreprocessor, target_index: int) -> np.ndarray:
    return preprocessor.inverse_t(pred_scaled) if target_index == 0 else preprocessor.inverse_logq(pred_scaled)


def safe_mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_percent": safe_mape_percent(y_true, y_pred),
    }


def case_folds(manifest: pd.DataFrame, n_splits: int) -> list[tuple[list[str], list[str]]]:
    case_ids = manifest["case_id"].to_numpy()
    dummy = np.zeros(len(case_ids))
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_index, val_index in splitter.split(dummy, groups=case_ids):
        folds.append((case_ids[train_index].tolist(), case_ids[val_index].tolist()))
    return folds


def prediction_frame(
    model_name: str,
    fold_index: int,
    windows: WindowSet,
    pred_t: np.ndarray,
    pred_q: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "fold": fold_index,
            "case_id": windows.case_ids,
            "time_min": windows.times_min,
            "true_Tout_C": windows.y_true[:, 0],
            "pred_Tout_C": pred_t,
            "true_log1p_Q_J": windows.y_true[:, 1],
            "pred_log1p_Q_J": pred_q,
        }
    )


def save_outputs(model_name: str, predictions: pd.DataFrame, config: dict[str, object], smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    output_dir = RESULTS_DIR / f"{model_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.sort_values(["fold", "case_id", "time_min"]).reset_index(drop=True)
    predictions.to_csv(output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    table_rows = []
    for target, true_col, pred_col in [
        ("Tout_C", "true_Tout_C", "pred_Tout_C"),
        ("log1p_Q_J", "true_log1p_Q_J", "pred_log1p_Q_J"),
    ]:
        table_rows.append({"model": model_name, "target": target, **metrics(predictions[true_col], predictions[pred_col])})
    pd.DataFrame(table_rows).to_csv(output_dir / "pooled_metrics.csv", index=False)
    per_case = []
    for case_id, group in predictions.groupby("case_id", sort=True):
        for target, true_col, pred_col in [
            ("Tout_C", "true_Tout_C", "pred_Tout_C"),
            ("log1p_Q_J", "true_log1p_Q_J", "pred_log1p_Q_J"),
        ]:
            per_case.append(
                {"model": model_name, "case_id": case_id, "target": target, **metrics(group[true_col], group[pred_col])}
            )
    pd.DataFrame(per_case).to_csv(output_dir / "per_case_metrics.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir


def train_neural_baseline(
    model_name: str,
    builder: Callable[[int], Model],
    epochs: int,
    batch_size: int,
    smoke: bool,
) -> Path:
    manifest, frames = load_development_cases()
    n_splits = 2 if smoke else N_SPLITS
    max_epochs = 1 if smoke else epochs
    fold_predictions: list[pd.DataFrame] = []
    fold_definition: list[dict[str, object]] = []
    start = time.perf_counter()
    for fold_index, (train_cases, val_cases) in enumerate(case_folds(manifest, n_splits), start=1):
        preprocessor = FoldPreprocessor.fit(frames, train_cases)
        train = make_windows(frames, train_cases, preprocessor)
        val = make_windows(frames, val_cases, preprocessor)
        target_predictions = []
        for target_index in range(2):
            set_seed(SEED * 100 + fold_index * 10 + target_index)
            model = builder(target_index)
            model.fit(
                train.X,
                train.y_scaled[:, target_index],
                epochs=max_epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
            )
            pred_scaled = model.predict(val.X, batch_size=256, verbose=0).reshape(-1)
            target_predictions.append(inverse_prediction(pred_scaled, preprocessor, target_index))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        fold_predictions.append(
            prediction_frame(model_name, fold_index, val, target_predictions[0], target_predictions[1])
        )
        fold_definition.append({"fold": fold_index, "train_cases": train_cases, "validation_cases": val_cases})
        print(f"{model_name}: fold {fold_index}/{n_splits} complete", flush=True)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    config = {
        "model": model_name,
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_splits": n_splits,
        "n_past": N_PAST,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "scaler_scope": "training cases of each fold only",
        "evaluation": "pooled out-of-fold predictions",
        "folds": fold_definition,
        "runtime_seconds": time.perf_counter() - start,
        "smoke": smoke,
    }
    return save_outputs(model_name, predictions, config, smoke)


def train_blstm_xgb(smoke: bool) -> Path:
    model_name = "BLSTM-XGB"
    manifest, frames = load_development_cases()
    n_splits = 2 if smoke else N_SPLITS
    # Fixed publication budget; the grouped outer split and the disjoint XGB
    # calibration split are held constant across seeds.
    epochs = 1 if smoke else 20
    batch_size = 128
    fold_predictions: list[pd.DataFrame] = []
    fold_definition: list[dict[str, object]] = []
    start = time.perf_counter()
    for fold_index, (train_cases, val_cases) in enumerate(case_folds(manifest, n_splits), start=1):
        preprocessor = FoldPreprocessor.fit(frames, train_cases)
        train_full = make_windows(frames, train_cases, preprocessor)
        val = make_windows(frames, val_cases, preprocessor)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
        sub_index, calibration_index = next(
            splitter.split(train_full.X, groups=train_full.case_ids)
        )
        target_predictions = []
        for target_index in range(2):
            set_seed(SEED * 100 + fold_index * 10 + target_index)
            stage1 = build_bilstm_xgb_lstm()
            stage1.fit(
                train_full.X[sub_index],
                train_full.y_scaled[sub_index, target_index],
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
            )
            calibration_scaled = stage1.predict(train_full.X[calibration_index], batch_size=256, verbose=0).reshape(-1)
            calibration_pred = inverse_prediction(calibration_scaled, preprocessor, target_index)
            calibration_true = train_full.y_true[calibration_index, target_index]
            stage2 = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=4,
            )
            stage2.fit(calibration_pred.reshape(-1, 1), calibration_true)
            val_scaled = stage1.predict(val.X, batch_size=256, verbose=0).reshape(-1)
            val_stage1 = inverse_prediction(val_scaled, preprocessor, target_index)
            target_predictions.append(stage2.predict(val_stage1.reshape(-1, 1)))
            del stage1, stage2
            tf.keras.backend.clear_session()
            gc.collect()
        fold_predictions.append(
            prediction_frame(model_name, fold_index, val, target_predictions[0], target_predictions[1])
        )
        fold_definition.append(
            {
                "fold": fold_index,
                "train_cases": train_cases,
                "validation_cases": val_cases,
                "stage1_cases": sorted(set(train_full.case_ids[sub_index])),
                "xgb_calibration_cases": sorted(set(train_full.case_ids[calibration_index])),
            }
        )
        print(f"{model_name}: fold {fold_index}/{n_splits} complete", flush=True)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    config = {
        "model": model_name,
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_splits": n_splits,
        "n_past": N_PAST,
        "epochs": epochs,
        "batch_size": batch_size,
        "xgb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        "scaler_scope": "training cases of each outer fold only",
        "calibration": "group-disjoint 80/20 split inside each outer training fold",
        "evaluation": "pooled outer-fold predictions",
        "folds": fold_definition,
        "runtime_seconds": time.perf_counter() - start,
        "smoke": smoke,
    }
    return save_outputs(model_name, predictions, config, smoke)


def train_lca(smoke: bool) -> Path:
    model_name = "LCA-Stacking"
    manifest, frames = load_development_cases()
    n_splits = 2 if smoke else N_SPLITS
    epochs = 1 if smoke else 10
    folds = case_folds(manifest, n_splits)
    base_frames: list[pd.DataFrame] = []
    fold_definition: list[dict[str, object]] = []
    start = time.perf_counter()
    builders: list[tuple[str, Callable[[int], Model]]] = [
        ("lstm", build_lca_lstm),
        ("cnn", build_lca_cnn),
        ("sa", build_lca_sa),
    ]
    for fold_index, (train_cases, val_cases) in enumerate(folds, start=1):
        preprocessor = FoldPreprocessor.fit(frames, train_cases)
        train = make_windows(frames, train_cases, preprocessor)
        val = make_windows(frames, val_cases, preprocessor)
        payload: dict[str, object] = {
            "fold": fold_index,
            "case_id": val.case_ids,
            "time_min": val.times_min,
            "true_Tout_C": val.y_true[:, 0],
            "true_log1p_Q_J": val.y_true[:, 1],
        }
        for target_index, target_name in enumerate(TARGETS):
            for model_index, (base_name, builder) in enumerate(builders):
                set_seed(SEED * 1000 + fold_index * 100 + target_index * 10 + model_index)
                model = builder(target_index)
                model.fit(
                    train.X,
                    train.y_scaled[:, target_index],
                    epochs=epochs,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                )
                pred_scaled = model.predict(val.X, batch_size=256, verbose=0).reshape(-1)
                payload[f"{target_name}_{base_name}"] = inverse_prediction(pred_scaled, preprocessor, target_index)
                del model
                tf.keras.backend.clear_session()
                gc.collect()
        base_frames.append(pd.DataFrame(payload))
        fold_definition.append({"fold": fold_index, "train_cases": train_cases, "validation_cases": val_cases})
        print(f"{model_name}: base OOF fold {fold_index}/{n_splits} complete", flush=True)

    base_oof = pd.concat(base_frames, ignore_index=True)
    base_oof["pred_Tout_C"] = np.nan
    base_oof["pred_log1p_Q_J"] = np.nan
    meta_rows: list[dict[str, object]] = []
    for fold_index in range(1, n_splits + 1):
        train_mask = base_oof["fold"].ne(fold_index)
        val_mask = base_oof["fold"].eq(fold_index)
        for target_name, true_col, output_col in [
            ("Tout_C", "true_Tout_C", "pred_Tout_C"),
            ("logQ", "true_log1p_Q_J", "pred_log1p_Q_J"),
        ]:
            feature_cols = [f"{target_name}_{base}" for base, _ in builders]
            meta = LinearRegression().fit(base_oof.loc[train_mask, feature_cols], base_oof.loc[train_mask, true_col])
            base_oof.loc[val_mask, output_col] = meta.predict(base_oof.loc[val_mask, feature_cols])
            meta_rows.append(
                {
                    "fold": fold_index,
                    "target": target_name,
                    "intercept": float(meta.intercept_),
                    **{f"coef_{name}": float(value) for name, value in zip([b for b, _ in builders], meta.coef_)},
                }
            )
    if base_oof[["pred_Tout_C", "pred_log1p_Q_J"]].isna().any().any():
        raise RuntimeError("Cross-fitted LCA meta predictions are incomplete")
    predictions = base_oof[
        ["fold", "case_id", "time_min", "true_Tout_C", "pred_Tout_C", "true_log1p_Q_J", "pred_log1p_Q_J"]
    ].copy()
    predictions.insert(0, "model", model_name)
    suffix = "_smoke" if smoke else ""
    output_dir = RESULTS_DIR / f"{model_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_oof.to_csv(output_dir / "base_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(meta_rows).to_csv(output_dir / "meta_crossfit_coefficients.csv", index=False)
    config = {
        "model": model_name,
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_splits": n_splits,
        "n_past": N_PAST,
        "epochs": epochs,
        "batch_size": 32,
        "base_models": [name for name, _ in builders],
        "meta_model": "LinearRegression cross-fitted by the same outer fold labels",
        "scaler_scope": "training cases of each base-model fold only",
        "evaluation": "pooled cross-fitted meta predictions",
        "folds": fold_definition,
        "runtime_seconds": time.perf_counter() - start,
        "smoke": smoke,
    }
    return save_outputs(model_name, predictions, config, smoke)


def parser(description: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--smoke", action="store_true", help="Run 2 folds and 1 epoch for pipeline validation only")
    return value
