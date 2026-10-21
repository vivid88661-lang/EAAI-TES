"""Reconstruct Table 8 and Appendix Tables A4-A6 from frozen predictions."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 3407]
TARGETS = ['Tout_C', 'Q_cum_MJ']


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(prediction, float)-np.asarray(y, float))**2)))


def historical_rows(frame, horizon):
    """Construct persistence and 30-point OLS forecasts using history only."""
    origins = np.arange(29, len(frame)-30)
    history_index = origins[:, None] + np.arange(-29, 1)[None, :]
    positions = np.arange(-29, 1, dtype=float)
    centered = positions-positions.mean()
    result = pd.DataFrame({'case_id': frame.case_id.iloc[0], 'origin_index': origins})
    for target, column, scale in [('Tout_C', 'Tout_C', 1), ('Q_cum_MJ', 'Q_cum_J', 1e6)]:
        values = frame[column].to_numpy(float)
        history = values[history_index]
        slope = history @ centered / np.sum(centered**2)
        result[f'true_{target}'] = values[origins+horizon]/scale
        result[f'pred_{target}__Persistence'] = history[:, -1]/scale
        result[f'pred_{target}__Linear_trend'] = (history.mean(axis=1)+slope*(horizon-positions.mean()))/scale
    return result


def validate_inputs(repository):
    manifest = json.loads((HERE/'checksums.json').read_text())
    for relative, expected in manifest['files'].items():
        path = HERE/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Checksum mismatch: {relative}')
    for relative, expected in manifest['repository_dependencies'].items():
        path = repository/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Repository dependency differs: {relative}')
    metadata = read(repository/'data/metadata.csv')
    frames = {}
    for row in metadata.itertuples(index=False):
        path = repository/'data'/row.file
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise RuntimeError(f'Data checksum mismatch: {row.case_id}')
        frame = read(path)
        assert len(frame) == row.row_count
        assert frame.case_id.eq(row.case_id).all()
        np.testing.assert_allclose(np.diff(frame.time_min), .5, rtol=0, atol=1e-12)
        frames[row.case_id] = frame
    for cid in ['E01', 'E02']:
        frames[cid] = read(HERE/f'data/external/{cid}.csv.gz')
        raw = pd.read_csv(HERE/f'data/source/{cid}_cycle03_discharge.csv', header=None)
        raw = raw.loc[raw[0] > 0].reset_index(drop=True)
        assert len(raw) == len(frames[cid])
        np.testing.assert_allclose(frames[cid][['time_min','f_liquid','Tin_C','Tout_C','Q_cum_J']], raw[[0,1,2,3,5]], rtol=1e-12, atol=1e-8)
    return frames


def keyed(frame):
    assert not frame.duplicated(['case_id','origin_index']).any()
    return frame.set_index(['case_id','origin_index']).sort_index()


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert list(actual.columns) == list(expected.columns)
    for column in actual:
        if pd.api.types.is_numeric_dtype(actual[column]):
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-10, atol=1e-11)
        else:
            assert actual[column].tolist() == expected[column].tolist(), column


def calculate(frames):
    outputs = {}
    direct = {}
    horizon_rows = []
    for h in [1, 10, 30]:
        for scope, n in [('oof',25599), ('external',895)]:
            frame = keyed(read(HERE/f'predictions/direct/h{h:02}_{scope}.csv.gz'))
            assert len(frame) == n
            ids = [f'D{i:03}' for i in range(1,33)] if scope == 'oof' else ['E_RI002']
            assert set(frame.index.get_level_values('case_id')) == set(ids)
            baseline = keyed(pd.concat([historical_rows(frames[cid], h) for cid in ids], ignore_index=True))
            assert frame.index.equals(baseline.index)
            for column in baseline:
                np.testing.assert_allclose(frame[column], baseline[column], rtol=1e-11, atol=1e-8)
            direct[h, scope] = frame
            for target in TARGETS:
                row = dict(target=target, evaluation=scope, horizon_seconds=h*30, n_origins=n)
                for method in ['Persistence','Linear_trend','LCA']:
                    row[method+'_RMSE'] = rmse(frame['true_'+target], frame[f'pred_{target}__{method}'])
                horizon_rows.append(row)
    outputs['table_A4.csv'] = pd.DataFrame(horizon_rows)
    state = {}
    seed_metrics = []
    external_metrics = []
    baseline_external = keyed(pd.concat([historical_rows(frames[c],1) for c in ['E01','E02']], ignore_index=True))
    for seed in SEEDS:
        for scope,n in [('oof',25599),('external',1160)]:
            frame = keyed(read(HERE/f'predictions/state_anchored/seed_{seed}_{scope}.csv.gz'))
            assert len(frame) == n
            baseline = direct[1,'oof'] if scope == 'oof' else baseline_external
            assert frame.index.equals(baseline.index)
            state[seed,scope] = frame
            for target in TARGETS:
                np.testing.assert_allclose(frame['true_'+target],baseline['true_'+target],rtol=1e-12,atol=1e-10)
                if scope == 'oof':
                    seed_metrics.append(dict(seed=seed, target=target, n_origins=n,
                                             State_anchored_LCA_RMSE=rmse(frame['true_'+target], frame[f'pred_{target}__State_anchored_LCA'])))
                else:
                    for case in ['E01','E02','Pooled']:
                        current = frame if case == 'Pooled' else frame.loc[[case]]
                        b = baseline.loc[current.index]
                        external_metrics.append(dict(seed=seed,case=case,target=target,n_origins=len(current),
                            State_anchored_LCA_RMSE=rmse(current['true_'+target],current[f'pred_{target}__State_anchored_LCA']),
                            Persistence_RMSE=rmse(current['true_'+target],b[f'pred_{target}__Persistence']),
                            Linear_trend_RMSE=rmse(current['true_'+target],b[f'pred_{target}__Linear_trend'])))
    table8=[]
    for method in ['Persistence','LCA','State_anchored_LCA']:
        row={'model': method,'seed':42,'n_origins':25599}
        for target in TARGETS:
            frame=state[42,'oof'] if method=='State_anchored_LCA' else direct[1,'oof']
            row[target+'_RMSE']=rmse(frame['true_'+target],frame[f'pred_{target}__{method}'])
        table8.append(row)
    outputs['table_8.csv']=pd.DataFrame(table8)
    per_seed=pd.DataFrame(seed_metrics)
    outputs['table_A5_per_seed.csv']=per_seed
    table5=[]
    for target in TARGETS:
        values=per_seed.loc[per_seed.target.eq(target),'State_anchored_LCA_RMSE']
        b=direct[1,'oof']
        persistence=rmse(b['true_'+target],b[f'pred_{target}__Persistence'])
        table5.append(dict(target=target,Persistence_RMSE=persistence,
            Linear_trend_RMSE=rmse(b['true_'+target],b[f'pred_{target}__Linear_trend']),
            State_anchored_LCA_mean_RMSE=float(values.mean()),State_anchored_LCA_sample_SD=float(values.std(ddof=1)),
            RMSE_reduction_vs_persistence_percent=100*(1-float(values.mean())/persistence)))
    outputs['table_A5.csv']=pd.DataFrame(table5)
    per_external=pd.DataFrame(external_metrics)
    outputs['table_A6_per_seed.csv']=per_external
    table6=[]
    for case in ['E01','E02','Pooled']:
        for target in TARGETS:
            rows=per_external.loc[per_external.case.eq(case)&per_external.target.eq(target)]
            table6.append(dict(case=case,target=target,n_origins=int(rows.n_origins.iloc[0]),
                Persistence_RMSE=float(rows.Persistence_RMSE.iloc[0]),Linear_trend_RMSE=float(rows.Linear_trend_RMSE.iloc[0]),
                State_anchored_LCA_mean_RMSE=float(rows.State_anchored_LCA_RMSE.mean()),
                State_anchored_LCA_sample_SD=float(rows.State_anchored_LCA_RMSE.std(ddof=1))))
    outputs['table_A6.csv']=pd.DataFrame(table6)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,default=HERE/'outputs/reproduced')
    args=parser.parse_args()
    outputs=calculate(validate_inputs(args.repository_root.resolve()))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():
        compare(frame,read(HERE/'reference'/name))
        frame.to_csv(args.output/name,index=False,float_format='%.17g')
    (args.output/'verification.json').write_text(json.dumps({'status':'PASS','tables':['8','A4','A5','A6'],
        'internal_origins':25599,'previous_external_origins':895,'E01_origins':714,'E02_origins':446,
        'seeds':SEEDS,'baseline_and_truth_checked_against_raw_history':True},indent=2)+'\n')
    print('PASS: checksums, source data, historical baselines and Tables 8/A4/A5/A6')
    print(outputs['table_8.csv'].to_string(index=False))
    print(outputs['table_A5.csv'].to_string(index=False))
    print(outputs['table_A6.csv'].to_string(index=False))


if __name__=='__main__':
    main()
