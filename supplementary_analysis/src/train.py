"""Train the direct-horizon and state-anchored supplementary LCA models."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_DETERMINISTIC_OPS','1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from common import HERE, BUILDERS, original, read_cases, folds, origins, windows

BASES=['lstm','cnn','sa']
TARGETS=['Tout_C','Q_cum_MJ']


def processor(frames,ids,target,variant):
    proc=original.FoldPreprocessor.fit(frames,ids)
    if variant=='state_anchored' and target==1:
        delta=[]
        for cid in sorted(ids):
            q=frames[cid].Q_cum_J.to_numpy(float)/1e6
            idx=origins(frames[cid])
            delta.append(q[idx+1]-q[idx])
        proc.heat_increment_scale_MJ=float(np.sqrt(np.mean(np.concatenate(delta)**2)))
        assert proc.heat_increment_scale_MJ>0
    return proc


def label(ws,proc,target,variant):
    if variant=='direct':
        return ws.y_scaled[:,target]
    if target==0:
        delta=ws.rows.true_Tout_C.to_numpy()-ws.rows.pred_Tout_C__Persistence.to_numpy()
        return (delta*proc.t_scaler.scale_[0]).astype(np.float32)
    delta=ws.rows.true_Q_cum_MJ.to_numpy()-ws.rows.origin_Q_cum_MJ.to_numpy()
    return (delta/proc.heat_increment_scale_MJ).astype(np.float32)


def physical(scaled,proc,target,variant):
    scaled=np.asarray(scaled,dtype=np.float64).reshape(-1)
    if variant=='direct':
        return original.inverse_prediction(scaled,proc,target)
    return scaled/proc.t_scaler.scale_[0] if target==0 else scaled*proc.heat_increment_scale_MJ


def anchor(rows,target):
    return rows.pred_Tout_C__Persistence.to_numpy(float) if target==0 else rows.origin_Q_cum_MJ.to_numpy(float)


def contexts(ids):
    result=[]
    for outer,(fit,test) in enumerate(folds(ids,5),1):
        for inner,(train,meta) in enumerate(folds(fit,2),1):
            st,sv=folds(train,3)[0]
            result.append(dict(outer=outer,name=f'inner{inner}',fit_cases=train,predict_cases=meta,
                selector_train_cases=st,selector_validation_cases=sv,outer_test_cases=test))
        st,sv=folds(fit,3)[0]
        result.append(dict(outer=outer,name='refit',fit_cases=fit,predict_cases=test,
            selector_train_cases=st,selector_validation_cases=sv,outer_test_cases=test))
    ids=sorted(ids);st,sv=folds(ids,3)[0]
    result.append(dict(outer=0,name='final',fit_cases=ids,predict_cases=ids,
        selector_train_cases=st,selector_validation_cases=sv,outer_test_cases=[]))
    return result


def model_seed(seed,ctx,ti,bi):
    if ctx['name'].startswith('inner'):
        return seed*100000+ctx['outer']*1000+int(ctx['name'][-1])*100+ti*10+bi
    return seed*1000+ctx['outer']*100+ti*10+bi


class SelectEpoch(tf.keras.callbacks.Callback):
    def __init__(self,ws,proc,target):
        super().__init__();self.ws=ws;self.proc=proc;self.target=target;self.best=None;self.curve=[]
    def on_epoch_end(self,epoch,logs=None):
        pred=physical(self.model.predict(self.ws.X,batch_size=256,verbose=0),self.proc,self.target,'state_anchored')+anchor(self.ws.rows,self.target)
        truth=self.ws.rows['true_'+TARGETS[self.target]].to_numpy(float)
        error=pred-truth
        score=float(np.mean([np.mean(error[idx]**2) for idx in self.ws.rows.groupby('case_id').indices.values()]))
        self.curve.append(dict(epoch=epoch+1,validation_case_mean_MSE=score,training_scaled_MSE=float(logs['loss'])))
        if self.best is None or score<self.best[1]:self.best=(epoch+1,score)


class SaveSelected(tf.keras.callbacks.Callback):
    def __init__(self,epoch,path):
        super().__init__();self.epoch=epoch;self.path=path
    def on_epoch_end(self,epoch,logs=None):
        if epoch+1==self.epoch:self.model.save(self.path)


def fit_base(frames,ctx,ti,bi,seed,h,variant,folder):
    folder.mkdir(parents=True,exist_ok=True)
    model_path=folder/'model.keras';proc_path=folder/'preprocessor.joblib'
    chosen_path=folder/'selection.json'
    s=model_seed(seed,ctx,ti,bi)
    if variant=='state_anchored':
        proc=processor(frames,ctx['selector_train_cases'],ti,variant)
        tr=windows(frames,ctx['selector_train_cases'],proc,h)
        va=windows(frames,ctx['selector_validation_cases'],proc,h)
        original.set_seed(s+500000);model=BUILDERS[bi][1](ti)
        callback=SelectEpoch(va,proc,ti)
        model.fit(tr.X,label(tr,proc,ti,variant),epochs=30,batch_size=32,shuffle=True,verbose=0,callbacks=[callback])
        chosen=callback.best[0]
        pd.DataFrame(callback.curve).to_csv(folder/'epoch_selection.csv',index=False)
        del model,tr,va,proc;tf.keras.backend.clear_session();gc.collect()
    else:chosen=10
    proc=processor(frames,ctx['fit_cases'],ti,variant)
    tr=windows(frames,ctx['fit_cases'],proc,h)
    va=windows(frames,ctx['predict_cases'],proc,h)
    original.set_seed(s);model=BUILDERS[bi][1](ti)
    model.fit(tr.X,label(tr,proc,ti,variant),epochs=max(10,chosen),batch_size=32,shuffle=True,verbose=0,
              callbacks=[SaveSelected(chosen,model_path)])
    model.load_weights(model_path)
    pred=physical(model.predict(va.X,batch_size=256,verbose=0),proc,ti,variant)
    joblib.dump(proc,proc_path)
    chosen_path.write_text(json.dumps(dict(selected_epoch=chosen,model_seed=s,context=ctx),indent=2)+'\n')
    result=va.rows.copy();result['base_prediction']=pred
    del model,tr,va,proc;tf.keras.backend.clear_session();gc.collect()
    print(variant,seed,h,ctx['outer'],ctx['name'],TARGETS[ti],BASES[bi],'epoch',chosen,flush=True)
    return result


def run(seed,h,variant,output):
    frames=read_cases(False);ctxs=contexts(list(frames))
    records={(ti,ctx['outer'],ctx['name']):None for ti in range(2) for ctx in ctxs}
    for ctx in ctxs:
        if ctx['outer']:
            assert not set(ctx['fit_cases'])&set(ctx['predict_cases'])
            assert not set(ctx['fit_cases'])&set(ctx['outer_test_cases'])
        for ti in range(2):
            current=None
            for bi,base in enumerate(BASES):
                result=fit_base(frames,ctx,ti,bi,seed,h,variant,output/f'outer{ctx["outer"]}_{ctx["name"]}/{base}_{TARGETS[ti]}')
                if current is None:current=result.drop(columns='base_prediction').copy()
                assert current[['case_id','origin_index']].equals(result[['case_id','origin_index']])
                current[base]=result.base_prediction.to_numpy(float)
            records[ti,ctx['outer'],ctx['name']]=current
    for ti,target in enumerate(TARGETS):
        oof=[]
        for outer in range(1,6):
            tr=pd.concat([records[ti,outer,'inner1'],records[ti,outer,'inner2']],ignore_index=True)
            held=records[ti,outer,'refit'].copy()
            assert not set(tr.case_id)&set(held.case_id)
            truth=tr['true_'+(target if variant=='state_anchored' or ti==0 else 'log1p_Q_J')].to_numpy(float)
            if variant=='state_anchored':truth=truth-anchor(tr,ti)
            meta=LinearRegression().fit(tr[BASES].astype(np.float64),truth)
            pred=meta.predict(held[BASES].astype(np.float64))
            if variant=='state_anchored':pred=pred+anchor(held,ti)
            elif ti==1:pred=np.expm1(pred)/1e6
            held['prediction']=pred;held['fold']=outer;oof.append(held)
        oof=pd.concat(oof,ignore_index=True)
        oof[['case_id','origin_index','fold','true_'+target,'prediction']].to_csv(output/f'oof_{target}.csv.gz',index=False,compression='gzip')
        truth=oof['true_'+(target if variant=='state_anchored' or ti==0 else 'log1p_Q_J')].to_numpy(float)
        if variant=='state_anchored':truth=truth-anchor(oof,ti)
        meta=LinearRegression().fit(oof[BASES].astype(np.float64),truth)
        joblib.dump(meta,output/f'final_meta_{target}.joblib')
    (output/'protocol.json').write_text(json.dumps(dict(variant=variant,seed=seed,horizon_steps=h,contexts=ctxs),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['direct','state_anchored'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,123,2024,3407],default=42)
    parser.add_argument('--horizon',type=int,choices=[1,10,30],default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.variant=='state_anchored' and args.horizon!=1:parser.error('State-anchored analysis uses horizon 1.')
    if args.variant=='direct' and args.seed!=42:parser.error('The direct-horizon analysis uses seed 42.')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose an empty output directory.')
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    run(args.seed,args.horizon,args.variant,args.output)


if __name__=='__main__':main()
