"""Shared case-wise preprocessing and historical window construction."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get('TES_REPOSITORY_ROOT', HERE.parent)).resolve()
sys.path.insert(0, str(PACKAGE / 'src'))
import preprocessing as original
HORIZONS = [1, 10, 30]
BUILDERS = [('lstm', original.build_lca_lstm), ('cnn', original.build_lca_cnn), ('sa', original.build_lca_sa)]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(external=False):
    manifest = pd.read_csv(PACKAGE / 'data/metadata.csv')
    role = 'external' if external else 'development'
    rows = manifest.loc[manifest.dataset_role.eq(role)].sort_values('case_id')
    assert rows.case_id.tolist() == (['E_RI002'] if external else [f'D{i:03}' for i in range(1, 33)])
    frames = {}
    for row in rows.itertuples(index=False):
        path = PACKAGE / 'data' / row.file
        assert digest(path) == row.sha256, f'checksum mismatch {path}'
        frame = pd.read_csv(path)
        assert len(frame) == row.row_count and frame.case_id.eq(row.case_id).all()
        assert not frame.isna().any().any()
        assert not frame.time_min.duplicated().any()
        assert np.allclose(np.diff(frame.time_min), .5, rtol=0, atol=1e-12)
        assert np.allclose(np.log1p(frame.Q_cum_J), frame.log1p_Q_J, rtol=0, atol=1e-12)
        frames[row.case_id] = frame
    return frames


def folds(case_ids, n):
    case_ids = np.asarray(sorted(case_ids))
    splitter = GroupKFold(n_splits=n)
    return [(case_ids[a].tolist(), case_ids[b].tolist())
            for a, b in splitter.split(np.zeros(len(case_ids)), groups=case_ids)]


def origins(frame, smoke=False):
    result = np.arange(29, len(frame) - 30)
    if smoke:
        result = result[np.linspace(0, len(result)-1, 32).astype(int)]
    assert len(result) and result[-1] + max(HORIZONS) < len(frame)
    return result


def baseline_rows(frame, h, smoke=False):
    idx = origins(frame, smoke)
    hist_idx = idx[:, None] + np.arange(-29, 1)[None, :]
    t_hist = frame.Tout_C.to_numpy()[hist_idx]
    q_hist = frame.Q_cum_J.to_numpy()[hist_idx]
    pos = np.arange(-29, 1, dtype=float)
    centered = pos - pos.mean()
    slope_t = t_hist @ centered / np.sum(centered ** 2)
    slope_q = q_hist @ centered / np.sum(centered ** 2)
    trend_t = t_hist.mean(axis=1) + slope_t * (h - pos.mean())
    trend_q = q_hist.mean(axis=1) + slope_q * (h - pos.mean())
    assert (trend_q >= 0).all(), 'Frozen un-clipped physical-Q linear baseline returned negative Q'
    q_now = frame.Q_cum_J.to_numpy()[idx] / 1e6
    q_future = frame.Q_cum_J.to_numpy()[idx + h] / 1e6
    data = dict(case_id=frame.case_id.iloc[0], origin_index=idx,
                origin_time_min=frame.time_min.to_numpy()[idx], target_index=idx + h,
                target_time_min=frame.time_min.to_numpy()[idx + h], horizon_steps=h,
                origin_Q_cum_MJ=q_now, true_Tout_C=frame.Tout_C.to_numpy()[idx+h],
                true_log1p_Q_J=frame.log1p_Q_J.to_numpy()[idx+h], true_Q_cum_MJ=q_future,
                true_delta_Q_MJ=q_future-q_now,
                pred_Tout_C__Persistence=t_hist[:, -1], pred_Tout_C__Linear_trend=trend_t,
                pred_Q_cum_MJ__Persistence=q_now, pred_Q_cum_MJ__Linear_trend=trend_q/1e6)
    for model in ['Persistence', 'Linear_trend']:
        data[f'pred_log1p_Q_J__{model}'] = np.log1p(data[f'pred_Q_cum_MJ__{model}']*1e6)
        data[f'pred_delta_Q_MJ__{model}'] = data[f'pred_Q_cum_MJ__{model}'] - q_now
    rows = pd.DataFrame(data)
    assert np.allclose((rows.target_time_min-rows.origin_time_min)*60, h*30)
    return rows


@dataclass
class Windows:
    X: np.ndarray
    y_scaled: np.ndarray
    y_true: np.ndarray
    rows: pd.DataFrame


def windows(frames, ids, processor, h, smoke=False):
    X, scaled_y, true_y, metadata = [], [], [], []
    for cid in sorted(ids):
        frame = frames[cid]
        idx = origins(frame, smoke)
        scaled = processor.transform_case(frame)
        history = scaled[idx[:, None] + np.arange(-29, 1)[None, :]]
        delta = history[:, -2, 3] - history[:, 0, 3]
        X.append(np.concatenate([history, np.broadcast_to(delta[:, None, None], (len(idx), 30, 1))], axis=2))
        scaled_y.append(scaled[idx+h][:, [2, 3]])
        true_y.append(frame[['Tout_C', 'log1p_Q_J']].to_numpy()[idx+h])
        metadata.append(baseline_rows(frame, h, smoke))
    return Windows(np.concatenate(X).astype(np.float32), np.concatenate(scaled_y).astype(np.float32),
                   np.concatenate(true_y).astype(np.float64), pd.concat(metadata, ignore_index=True))
