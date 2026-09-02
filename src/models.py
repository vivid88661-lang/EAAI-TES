from __future__ import annotations

"""Frozen-artifact loading, model refitting and deployment inference."""

import gc
import hashlib
import json
import os
import time
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
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import Model
from xgboost import XGBRegressor

from preprocessing import (
    DATA_DIR,
    MANIFEST_PATH,
    N_PAST,
    RESULTS_DIR,
    SEED,
    SPLIT_SEED,
    TARGETS,
    FoldPreprocessor,
    WindowSet,
    build_bilstm_xgb_lstm,
    build_blstm_sa,
    build_cnn_blstm,
    build_lca_cnn,
    build_lca_lstm,
    build_lca_sa,
    inverse_prediction,
    load_development_cases,
    make_windows,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(
    os.environ.get("TES_ARTIFACTS_DIR", ROOT / "outputs" / "training_artifacts")
).resolve()
EXTERNAL_RESULTS_DIR = RESULTS_DIR
MODEL_ORDER = ["BLSTM-XGB", "CNN-BLSTM", "BLSTM-SA", "LCA-Stacking"]
MODEL_SLUGS = {
    "BLSTM-XGB": "BLSTM_XGB",
    "CNN-BLSTM": "CNN_BLSTM",
    "BLSTM-SA": "BLSTM_SA",
    "LCA-Stacking": "LCA_Stacking",
}
PUBLIC_FROZEN_DIR = ROOT / "models" / "frozen_models"
PUBLIC_SCALERS_DIR = ROOT / "models" / "scalers"
LCA_BUILDERS: list[tuple[str, Callable[[int], Model]]] = [
    ("lstm", build_lca_lstm),
    ("cnn", build_lca_cnn),
    ("sa", build_lca_sa),
]


def artifact_root(smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return ARTIFACTS_DIR / f"frozen_mainline{suffix}"


def external_result_root(smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return EXTERNAL_RESULTS_DIR / f"external_mainline{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def selected_development_cases(smoke: bool) -> tuple[dict[str, pd.DataFrame], list[str]]:
    manifest, frames = load_development_cases()
    case_ids = manifest["case_id"].tolist()
    if smoke:
        case_ids = case_ids[:6]
        frames = {case_id: frames[case_id] for case_id in case_ids}
    return frames, case_ids


def _model_dir(model_name: str, smoke: bool) -> Path:
    path = artifact_root(smoke) / MODEL_SLUGS[model_name]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fit_neural_target(
    builder: Callable[[int], Model],
    windows: WindowSet,
    preprocessor: FoldPreprocessor,
    target_index: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Model:
    set_seed(seed)
    model = builder(target_index)
    model.fit(
        windows.X,
        windows.y_scaled[:, target_index],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    return model


def fit_blstm_xgb_final(smoke: bool) -> Path:
    model_name = "BLSTM-XGB"
    output = _model_dir(model_name, smoke)
    frames, case_ids = selected_development_cases(smoke)
    preprocessor = FoldPreprocessor.fit(frames, case_ids)
    windows = make_windows(frames, case_ids, preprocessor)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
    stage1_index, calibration_index = next(splitter.split(windows.X, groups=windows.case_ids))
    epochs = 1 if smoke else 20
    batch_size = 128
    started = time.perf_counter()

    for target_index, target_name in enumerate(TARGETS):
        set_seed(SEED * 100 + target_index)
        stage1 = build_bilstm_xgb_lstm()
        stage1.fit(
            windows.X[stage1_index],
            windows.y_scaled[stage1_index, target_index],
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
        )
        calibration_scaled = stage1.predict(
            windows.X[calibration_index], batch_size=256, verbose=0
        ).reshape(-1)
        calibration_pred = inverse_prediction(calibration_scaled, preprocessor, target_index)
        stage2 = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=4,
        )
        stage2.fit(
            calibration_pred.reshape(-1, 1),
            windows.y_true[calibration_index, target_index],
        )
        stage1.save(output / f"stage1_{target_name}.keras")
        joblib.dump(stage2, output / f"stage2_{target_name}.joblib")
        del stage1, stage2
        tf.keras.backend.clear_session()
        gc.collect()

    joblib.dump(preprocessor, output / "preprocessor.joblib")
    config = {
        "model": model_name,
        "role": "final external-inference artifact; not the Table 6-7 OOF estimator",
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_past": N_PAST,
        "development_cases": case_ids,
        "stage1_cases": sorted(set(windows.case_ids[stage1_index])),
        "xgb_calibration_cases": sorted(set(windows.case_ids[calibration_index])),
        "epochs": epochs,
        "batch_size": batch_size,
        "xgb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        "scaler_scope": "selected development cases only",
        "early_stopping": False,
        "smoke": smoke,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "artifact_config.json", config)
    return output


def fit_neural_baseline_final(model_name: str, smoke: bool) -> Path:
    if model_name not in {"CNN-BLSTM", "BLSTM-SA"}:
        raise ValueError(model_name)
    output = _model_dir(model_name, smoke)
    frames, case_ids = selected_development_cases(smoke)
    preprocessor = FoldPreprocessor.fit(frames, case_ids)
    windows = make_windows(frames, case_ids, preprocessor)
    builder = build_cnn_blstm if model_name == "CNN-BLSTM" else build_blstm_sa
    epochs = 1 if smoke else 10
    batch_size = 128
    started = time.perf_counter()

    for target_index, target_name in enumerate(TARGETS):
        model = _fit_neural_target(
            builder,
            windows,
            preprocessor,
            target_index,
            epochs,
            batch_size,
            SEED * 100 + target_index,
        )
        model.save(output / f"model_{target_name}.keras")
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    joblib.dump(preprocessor, output / "preprocessor.joblib")
    config = {
        "model": model_name,
        "role": "full-development refit for external inference",
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_past": N_PAST,
        "development_cases": case_ids,
        "epochs": epochs,
        "batch_size": batch_size,
        "scaler_scope": "selected development cases only",
        "early_stopping": False,
        "smoke": smoke,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "artifact_config.json", config)
    return output


def fit_lca_final(smoke: bool) -> Path:
    model_name = "LCA-Stacking"
    output = _model_dir(model_name, smoke)
    frames, case_ids = selected_development_cases(smoke)
    preprocessor = FoldPreprocessor.fit(frames, case_ids)
    windows = make_windows(frames, case_ids, preprocessor)
    epochs = 1 if smoke else 10
    batch_size = 32
    suffix = "_smoke" if smoke else ""
    oof_path = RESULTS_DIR / f"LCA-Stacking{suffix}" / "base_oof_predictions.csv.gz"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"LCA meta-training requires completed base OOF predictions: {oof_path}"
        )
    base_oof = pd.read_csv(oof_path)
    started = time.perf_counter()
    meta_summary: list[dict[str, object]] = []

    for target_name, true_col in [
        ("Tout_C", "true_Tout_C"),
        ("logQ", "true_log1p_Q_J"),
    ]:
        feature_cols = [f"{target_name}_{base}" for base, _ in LCA_BUILDERS]
        meta = LinearRegression().fit(base_oof[feature_cols], base_oof[true_col])
        joblib.dump(meta, output / f"meta_{target_name}.joblib")
        meta_summary.append(
            {
                "target": target_name,
                "intercept": float(meta.intercept_),
                "coefficients": {
                    base_name: float(value)
                    for (base_name, _), value in zip(LCA_BUILDERS, meta.coef_)
                },
            }
        )

    for target_index, target_name in enumerate(TARGETS):
        for base_index, (base_name, builder) in enumerate(LCA_BUILDERS):
            model = _fit_neural_target(
                builder,
                windows,
                preprocessor,
                target_index,
                epochs,
                batch_size,
                SEED * 1000 + target_index * 10 + base_index,
            )
            model.save(output / f"base_{target_name}_{base_name}.keras")
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    joblib.dump(preprocessor, output / "preprocessor.joblib")
    config = {
        "model": model_name,
        "role": "OOF-trained meta-model plus full-development base-model refits for external inference",
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_past": N_PAST,
        "development_cases": case_ids,
        "epochs": epochs,
        "batch_size": batch_size,
        "base_models": [name for name, _ in LCA_BUILDERS],
        "meta_model": "LinearRegression fitted once on grouped base OOF predictions",
        "meta_training_source": str(oof_path.relative_to(ROOT)).replace("\\", "/"),
        "meta_training_sha256": sha256(oof_path),
        "meta_parameters": meta_summary,
        "scaler_scope": "selected development cases only",
        "early_stopping": False,
        "smoke": smoke,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "artifact_config.json", config)
    return output


def fit_final_model(model_name: str, smoke: bool) -> Path:
    if model_name == "BLSTM-XGB":
        return fit_blstm_xgb_final(smoke)
    if model_name in {"CNN-BLSTM", "BLSTM-SA"}:
        return fit_neural_baseline_final(model_name, smoke)
    if model_name == "LCA-Stacking":
        return fit_lca_final(smoke)
    raise ValueError(model_name)


def finalize_artifact_manifest(smoke: bool) -> Path:
    root = artifact_root(smoke)
    root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MODEL_ORDER if not (_model_dir(name, smoke) / "artifact_config.json").exists()]
    if missing:
        raise RuntimeError(f"Final artifacts are incomplete: {missing}")
    checksum_rows = []
    for model_name in MODEL_ORDER:
        model_dir = root / MODEL_SLUGS[model_name]
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                checksum_rows.append(f"{sha256(path)}  {relative}")
    artifact_checksums = root / "artifact_checksums.sha256"
    artifact_checksums.write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    data_checksums = ROOT / "data" / "metadata.csv"
    payload = {
        "pipeline": "frozen grouped paper mainline",
        "evaluation": "five-fold GroupKFold OOF on D001-D032 (Table 6-7)",
        "deployment": "frozen-hyperparameter final fitting on development cases only",
        "external_test": "E_RI002, transform and inference only",
        "prediction_mode": "one-step-ahead using observed 30-step history within the external case",
        "models": MODEL_ORDER,
        "artifact_checksum_file": artifact_checksums.name,
        "artifact_checksum_file_sha256": sha256(artifact_checksums),
        "data_checksum_file": str(data_checksums.relative_to(ROOT)).replace("\\", "/"),
        "data_checksum_file_sha256": sha256(data_checksums),
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "smoke": smoke,
    }
    path = root / "artifact_manifest.json"
    _write_json(path, payload)
    return path


def verify_artifact_checksums(smoke: bool) -> None:
    root = artifact_root(smoke)
    checksum_path = root / "artifact_checksums.sha256"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Frozen artifact checksum list is missing: {checksum_path}")
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(checksum_path) != manifest["artifact_checksum_file_sha256"]:
        raise RuntimeError("Frozen artifact checksum-list hash does not match the manifest")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen artifact checksum mismatch: {relative.strip()}")


def load_external_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = pd.read_csv(MANIFEST_PATH)
    external = (
        manifest.loc[manifest["dataset_role"].eq("external")]
        .copy()
        .sort_values("case_id")
    )
    if external["case_id"].tolist() != ["E_RI002"]:
        raise RuntimeError(
            f"The frozen external test must contain only E_RI002, found {external['case_id'].tolist()}"
        )
    frames: dict[str, pd.DataFrame] = {}
    for row in external.itertuples(index=False):
        case_id = row.case_id
        frame = pd.read_csv(ROOT / "data" / row.file)
        if not frame["split"].eq("external_test").all():
            raise RuntimeError(f"External split mismatch for {case_id}")
        frames[case_id] = frame.reset_index(drop=True)
    return external, frames


def _load_keras(path: Path) -> Model:
    return tf.keras.models.load_model(path, compile=False)


def predict_frozen_model(
    model_name: str,
    windows: WindowSet,
    preprocessor: FoldPreprocessor,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model_dir = artifact_root(smoke) / MODEL_SLUGS[model_name]
    predictions: list[np.ndarray] = []

    if model_name == "BLSTM-XGB":
        for target_index, target_name in enumerate(TARGETS):
            stage1 = _load_keras(model_dir / f"stage1_{target_name}.keras")
            stage2 = joblib.load(model_dir / f"stage2_{target_name}.joblib")
            scaled = stage1.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
            stage1_prediction = inverse_prediction(scaled, preprocessor, target_index)
            predictions.append(stage2.predict(stage1_prediction.reshape(-1, 1)))
            del stage1, stage2
            tf.keras.backend.clear_session()
            gc.collect()
        return predictions[0], predictions[1]

    if model_name in {"CNN-BLSTM", "BLSTM-SA"}:
        for target_index, target_name in enumerate(TARGETS):
            model = _load_keras(model_dir / f"model_{target_name}.keras")
            scaled = model.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
            predictions.append(inverse_prediction(scaled, preprocessor, target_index))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        return predictions[0], predictions[1]

    if model_name == "LCA-Stacking":
        for target_index, target_name in enumerate(TARGETS):
            base_predictions = []
            for base_name, _ in LCA_BUILDERS:
                model = _load_keras(model_dir / f"base_{target_name}_{base_name}.keras")
                scaled = model.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
                base_predictions.append(inverse_prediction(scaled, preprocessor, target_index))
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            meta = joblib.load(model_dir / f"meta_{target_name}.joblib")
            feature_cols = [f"{target_name}_{base_name}" for base_name, _ in LCA_BUILDERS]
            meta_features = pd.DataFrame(np.column_stack(base_predictions), columns=feature_cols)
            predictions.append(meta.predict(meta_features))
        return predictions[0], predictions[1]

    raise ValueError(model_name)


def load_public_preprocessor(model_name: str) -> FoldPreprocessor:
    """Load the scaler bundle distributed with the frozen paper model."""
    return joblib.load(PUBLIC_SCALERS_DIR / f"{MODEL_SLUGS[model_name]}.joblib")


def predict_public_frozen_model(
    model_name: str,
    windows: WindowSet,
    preprocessor: FoldPreprocessor,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a distributed frozen model without writing to the repository."""
    model_dir = PUBLIC_FROZEN_DIR / MODEL_SLUGS[model_name]
    predictions: list[np.ndarray] = []

    if model_name == "BLSTM-XGB":
        for target_index, target_name in enumerate(TARGETS):
            stage1 = _load_keras(model_dir / f"stage1_{target_name}.keras")
            stage2 = joblib.load(model_dir / f"stage2_{target_name}.joblib")
            scaled = stage1.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
            stage1_prediction = inverse_prediction(scaled, preprocessor, target_index)
            predictions.append(stage2.predict(stage1_prediction.reshape(-1, 1)))
            del stage1, stage2
            tf.keras.backend.clear_session()
            gc.collect()
        return predictions[0], predictions[1]

    if model_name in {"CNN-BLSTM", "BLSTM-SA"}:
        for target_index, target_name in enumerate(TARGETS):
            model = _load_keras(model_dir / f"model_{target_name}.keras")
            scaled = model.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
            predictions.append(inverse_prediction(scaled, preprocessor, target_index))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        return predictions[0], predictions[1]

    if model_name == "LCA-Stacking":
        for target_index, target_name in enumerate(TARGETS):
            base_predictions = []
            for base_name, _ in LCA_BUILDERS:
                model = _load_keras(model_dir / f"base_{target_name}_{base_name}.keras")
                scaled = model.predict(windows.X, batch_size=256, verbose=0).reshape(-1)
                base_predictions.append(
                    inverse_prediction(scaled, preprocessor, target_index)
                )
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            meta = joblib.load(model_dir / f"meta_{target_name}.joblib")
            feature_cols = [f"{target_name}_{base_name}" for base_name, _ in LCA_BUILDERS]
            meta_features = pd.DataFrame(
                np.column_stack(base_predictions), columns=feature_cols
            )
            predictions.append(meta.predict(meta_features))
        return predictions[0], predictions[1]

    raise ValueError(model_name)
