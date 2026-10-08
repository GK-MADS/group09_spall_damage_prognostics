import pandas as pd, numpy as np
from sklearn.covariance import LedoitWolf
F=['ci_fm4','ci_m6a','ci_m8a','ci_na4','ci_e_ratio','ci_kurtosis_tsa','ci_rms_tsa','ci_alr']
def score(g, base_h=(0.5,12.0), feats=F, q=0.995, persist=60, regress=True):
    g=g.dropna(subset=feats+['ctx_temperature']).copy()
    X=np.log(g[feats].abs()+1e-9).values
    base=((g.hours>base_h[0])&(g.hours<base_h[1])).values
    T=np.c_[np.ones(len(g)),g.ctx_temperature] if regress else np.ones((len(g),1))
    beta=np.linalg.lstsq(T[base],X[base],rcond=None)[0]; R=X-T@beta
    Z=(R-R[base].mean(0))/R[base].std(0)
    d2=LedoitWolf().fit(Z[base]).mahalanobis(Z)
    s=pd.Series(np.sqrt(d2/len(feats)),index=g.hours.values)
    sm=s.rolling(30,min_periods=10).median()
    thr=float(np.quantile(sm[base].dropna(),q))
    above=(sm>thr)&(sm.index>base_h[1])
    run=above.astype(int).groupby((~above).cumsum()).cumsum()
    hit=run[run>=persist]
    on=float(hit.index[0]-persist/60) if len(hit) else None
    return s, sm, thr, on
if __name__=='__main__':
    m=pd.read_parquet('eda/out/all_1min.parquet')
    for bh in [(0.5,6),(0.5,12),(0.5,18)]:
        r={}
        for e,g in m.groupby('experiment'):
            s,sm,thr,on=score(g,bh); r[e]=(None if on is None else round(on,1), round(g.hours.max(),1))
        print(bh,r)
    g=m[m.experiment=='Exp-C']; s,sm,thr,on=score(g)
    print(sm.groupby((sm.index//6.16).astype(int)).median().round(2).to_dict(), thr)
