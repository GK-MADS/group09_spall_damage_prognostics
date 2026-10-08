import pandas as pd, numpy as np
S='eda/out'
frames=[];parts=[]
for s in ['train','validation','test']:
    lf=pd.read_parquet(f'data/{s}/lf_1s.parquet'); lf['split']=s
    pt=pd.read_parquet(f'data/{s}/particles.parquet'); pt['split']=s
    # dedup: Exp-A Run-2 folder is byte-identical to Run-1 -> null it out, keep the nominal hour axis
    dup=(lf.experiment=='Exp-A')&(lf.run_num==2)
    ci=[c for c in lf if c.startswith(('ci_','ctx_'))]
    lf.loc[dup,ci]=np.nan
    parts.append(pt[~((pt.experiment=='Exp-A')&(pt.run_num==2))])
    lf['minute']=(lf.test_hours*60).astype(int)
    frames.append(lf)
lf=pd.concat(frames,ignore_index=True)
ci=[c for c in lf if c.startswith('ci_')]; ctx=[c for c in lf if c.startswith('ctx_')]
# steady-state flag: nominal speed and torque
nom_t=lf.groupby('experiment').ctx_pau_torque.transform('median')
lf['steady']=(lf.ctx_pau_speed.sub(1400).abs()<20)&(lf.ctx_pau_torque.sub(nom_t).abs()<0.05*nom_t)
lf.to_parquet(S+'/all_1s.parquet')
m=lf[lf.steady].groupby(['split','experiment','run_num','minute'])[ci+ctx].median().reset_index()
m['hours']=m.minute/60
m.to_parquet(S+'/all_1min.parquet')
pd.concat(parts).to_parquet(S+'/particles_all.parquet')
print(lf.shape, lf.steady.mean().round(3), m.shape)
print(lf.groupby('experiment').steady.mean().round(3).to_dict())
