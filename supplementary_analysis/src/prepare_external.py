"""Extract the positive-time third-discharge trajectory for Appendix A6."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['E01','E02'],required=True)
    parser.add_argument('--source',type=Path,required=True,help='cycle03_discharge.csv')
    parser.add_argument('--output',type=Path,required=True,help='Destination .csv or .csv.gz')
    args=parser.parse_args()
    ri,length,phi,tin={'E01':(.0175,11.0,.15,560),'E02':(.025,13.5,.25,575)}[args.case]
    source=pd.read_csv(args.source,header=None)
    source=source.loc[source[0]>0].reset_index(drop=True)
    result=pd.DataFrame(dict(case_id=args.case,time_min=source[0],f_liquid=source[1],Tin_C=source[2],
        Tout_C=source[3],Q_cum_J=source[5],Ri_m=ri,L_m=length,Phi=phi,Tin_ch_C=tin))
    result['delta_Q_J']=result.Q_cum_J.diff().fillna(result.Q_cum_J.iloc[0])
    result['log1p_Q_J']=np.log1p(result.Q_cum_J)
    result['split']='new_external_locked'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False,float_format='%.17g',compression='infer')
    print(args.case,len(result),'samples',len(result)-59,'common prediction origins')


if __name__=='__main__':main()
