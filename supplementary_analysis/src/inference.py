"""Apply the frozen supplementary models without fitting any parameter."""
import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_DETERMINISTIC_OPS','1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from common import HERE, BUILDERS, read_cases, windows, original


def run(variant,output):
    output.mkdir(parents=True,exist_ok=True)
    if variant=='state_anchored':
        frames={cid:pd.read_csv(HERE/f'data/external/{cid}.csv.gz') for cid in ['E01','E02']}
        for seed in [42,123,2024,3407]:
            folder=HERE/f'models/state_anchored/seed_{seed}'
            merged=None
            for ti,target in enumerate(['Tout_C','Q_cum_MJ']):
                rows=None;pred=[]
                for base,builder in BUILDERS:
                    job=folder/f'{base}_{target}'
                    proc=joblib.load(job/'preprocessor.joblib')
                    ws=windows(frames,list(frames),proc,1)
                    model=builder(ti);model.load_weights(job/'model.keras')
                    scaled=model.predict(ws.X,batch_size=256,verbose=0).reshape(-1).astype(np.float64)
                    pred.append(scaled/proc.t_scaler.scale_[0] if ti==0 else scaled*proc.heat_increment_scale_MJ)
                    if rows is None:rows=ws.rows.copy()
                    assert rows[['case_id','origin_index']].equals(ws.rows[['case_id','origin_index']])
                    del model;tf.keras.backend.clear_session();gc.collect()
                meta=json.loads((folder/f'meta_{target}.json').read_text())
                increment=np.column_stack(pred)@np.array(meta['coefficients'])+meta['intercept']
                anchor=rows.pred_Tout_C__Persistence.to_numpy(float) if ti==0 else rows.origin_Q_cum_MJ.to_numpy(float)
                result=rows[['case_id','origin_index','true_'+target]].copy()
                result[f'pred_{target}__State_anchored_LCA']=anchor+increment
                merged=result if merged is None else merged.merge(result,on=['case_id','origin_index'],validate='one_to_one')
            merged.to_csv(output/f'seed_{seed}_external.csv.gz',index=False,compression='gzip')
            print('Completed state-anchored external inference:',seed,flush=True)
    else:
        frames=read_cases(True)
        for h in [1,10,30]:
            folder=HERE/f'models/direct/h{h:02}'
            proc=joblib.load(folder/'preprocessor.joblib');ws=windows(frames,list(frames),proc,h)
            rows=ws.rows.copy()
            for ti,target in enumerate(['Tout_C','log1p_Q_J']):
                predictions={}
                for base,_ in BUILDERS:
                    model=tf.keras.models.load_model(folder/f'{base}_{target}.keras',compile=False)
                    scaled=model.predict(ws.X,batch_size=256,verbose=0).reshape(-1)
                    predictions[target+'_'+base]=original.inverse_prediction(scaled,proc,ti)
                    del model;tf.keras.backend.clear_session();gc.collect()
                meta=joblib.load(folder/f'meta_{target}.joblib')
                value=meta.predict(pd.DataFrame(predictions).astype(np.float64))
                rows[f'pred_{"Tout_C" if ti==0 else "Q_cum_MJ"}__LCA']=value if ti==0 else np.expm1(value)/1e6
            cols=['case_id','origin_index']+['true_'+t for t in ['Tout_C','Q_cum_MJ']]+[f'pred_{t}__{m}' for t in ['Tout_C','Q_cum_MJ'] for m in ['Persistence','Linear_trend','LCA']]
            rows[cols].to_csv(output/f'h{h:02}_external.csv.gz',index=False,compression='gzip')
            print('Completed direct external inference:',h,flush=True)


def infer_trained_run(folder,variant,output):
    from train import physical, anchor
    protocol=json.loads((folder/'protocol.json').read_text())
    assert protocol['variant']==variant
    h=protocol['horizon_steps']
    frames=read_cases(True) if variant=='direct' else {cid:pd.read_csv(HERE/f'data/external/{cid}.csv.gz') for cid in ['E01','E02']}
    merged=None
    for ti,target in enumerate(['Tout_C','Q_cum_MJ']):
        rows=None;predictions={}
        for base,builder in BUILDERS:
            job=folder/f'outer0_final/{base}_{target}'
            proc=joblib.load(job/'preprocessor.joblib')
            ws=windows(frames,list(frames),proc,h)
            model=builder(ti);model.load_weights(job/'model.keras')
            predictions[base]=physical(model.predict(ws.X,batch_size=256,verbose=0),proc,ti,variant)
            if rows is None:rows=ws.rows.copy()
            assert rows[['case_id','origin_index']].equals(ws.rows[['case_id','origin_index']])
            del model;tf.keras.backend.clear_session();gc.collect()
        meta=joblib.load(folder/f'final_meta_{target}.joblib')
        pred=meta.predict(pd.DataFrame(predictions).astype(np.float64))
        if variant=='state_anchored':pred=pred+anchor(rows,ti)
        elif ti==1:pred=np.expm1(pred)/1e6
        result=rows[['case_id','origin_index','true_'+target]].copy()
        result['pred_'+target]=pred
        merged=result if merged is None else merged.merge(result,on=['case_id','origin_index'],validate='one_to_one')
    output.mkdir(parents=True,exist_ok=True)
    merged.to_csv(output/'external_predictions.csv.gz',index=False,compression='gzip')
    print('Completed inference from retrained model:',variant,protocol['seed'],h)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['direct','state_anchored'],required=True)
    parser.add_argument('--output',type=Path,default=HERE/'outputs/inference')
    parser.add_argument('--trained-run',type=Path,help='Optional output directory from train.py')
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.trained_run:infer_trained_run(args.trained_run,args.variant,args.output/args.variant)
    else:run(args.variant,args.output/args.variant)


if __name__=='__main__':main()
