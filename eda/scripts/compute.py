import json, sys, numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr
sys.path.insert(0,'eda/scripts')
from novelty import score, F
S='eda/out'
m=pd.read_parquet(S+'/all_1min.parquet'); s1=pd.read_parquet(S+'/all_1s.parquet')
pt=pd.read_parquet(S+'/particles_all.parquet'); ph=pd.read_csv(S+'/photo_proxy.csv')
ch=pd.read_csv('data/channels.csv')
CI=[c for c in m if c.startswith('ci_')]; CTX=[c for c in m if c.startswith('ctx_')]
GEARS=['Exp-A','Exp-B','Exp-C','Exp-F','Exp-G','Exp-D','Exp-E']
split={g:s for g,s in m.groupby('experiment').split.first().items()}
BASE=(0.5,12.0)
r3=lambda x: None if x is None or (isinstance(x,float) and not np.isfinite(x)) else round(float(x),3)
# ---------- runs & inspections
runs=s1.groupby(['experiment','run_num']).agg(h0=('test_hours','min'),h1=('test_hours','max'),temp=('ctx_temperature','median'),rows=('file','size')).reset_index()
runs['dup']=(runs.experiment=='Exp-A')&(runs.run_num==2)
runs['short']=(runs.h1-runs.h0)<3
ph['tooth']=ph.tooth.astype(int)
def insp_hour(e,o):
    if o<=0: return 0.0
    return float(runs[(runs.experiment==e)&(runs.run_num==o)].h1.iloc[0])
ph['hour']=[insp_hour(e,o) for e,o in zip(ph.experiment,ph.insp_order)]
EYE={('Exp-A',16,4):'init',('Exp-A',28,4):'init',('Exp-B',11,6):'init',('Exp-B',14,6):'init',('Exp-F',9,7):'init',('Exp-F',12,7):'init',('Exp-F',13,7):'init',
 **{k:'spall' for k in [('Exp-A',16,5),('Exp-A',24,5),('Exp-A',28,5),('Exp-B',11,7),('Exp-B',14,7),('Exp-C',6,12),('Exp-C',7,12),('Exp-F',9,8),('Exp-F',12,8),('Exp-F',13,8),('Exp-F',20,8),('Exp-F',19,8),('Exp-F',18,6),('Exp-F',18,7),('Exp-F',18,8)]}}
ph['eye']=[EYE.get((e,t,o)) for e,t,o in zip(ph.experiment,ph.tooth,ph.insp_order)]
ph['cls']=np.where(ph.eye=='spall','spall',np.where(ph.eye=='init','init',np.where(ph.tex>=0.02,'ambiguous','healthy')))
# ---------- gears table
gears=[]
for g in GEARS:
    x=s1[s1.experiment==g]; r=runs[runs.experiment==g]; p=ph[ph.experiment==g]
    fin=p[p.insp_order==p.insp_order.max()] if len(p) else p
    first_sp=p[p.eye.notna()].insp_order.min() if len(p) else None
    fsh=(insp_hour(g,int(first_sp)) if len(p) and pd.notna(first_sp) else None)
    gears.append(dict(first_dmg_h=r3(fsh),gear=g[-1],split=split[g],runs=int(r.run_num.nunique()),end_h=r3(r.h1.max()),
        sensor_h=r3((x.ci_fm4.notna()).sum()/3600),steady_pct=r3(x.steady.mean()*100),
        torque=r3(x.ctx_pau_torque.median()),temp=r3(x.ctx_temperature.median()),files=int(x.file.nunique()),rows=int(len(x)),
        short_runs=int(r.short.sum()),photo_insp=int(p.inspection.nunique()),photos=int(p.n_photos.sum()) if len(p) else 0,
        spalled_final=int((fin.cls=='spall').sum()) if len(p) else None,
        first_spall_insp=(None if len(p)==0 or pd.isna(first_sp) else p[p.insp_order==first_sp].inspection.iloc[0])))
# ---------- robust z per gear (baseline window)
def robust(g):
    b=g[(g.hours>BASE[0])&(g.hours<BASE[1])]
    med=b[CI].median(); mad=(b[CI]-med).abs().median()*1.4826
    return (g[CI]-med)/mad.replace(0,np.nan)
Zs={}
for g in GEARS:
    x=m[m.experiment==g].sort_values('hours')
    z=robust(x); z['hours']=x.hours.values; z['temp']=x.ctx_temperature.values
    Zs[g]=z
# ---------- channel stats
def mono(v):
    d=np.sign(np.diff(v)); d=d[d!=0]; return abs(d.sum())/max(len(v)-1,1)
stats=[]
for c in CI+CTX:
    v=m[c].dropna()
    row=dict(col=c,unit=(ch.set_index('column').unit.get(c) if c in set(ch.column) else None),
             mean=r3(v.mean()),sd=r3(v.std()),p1=r3(v.quantile(.01)),p50=r3(v.median()),p99=r3(v.quantile(.99)),
             miss=r3(s1[c].isna().mean()*100))
    if c in CI:
        mons=[];traj=[];ends=[];rhos=[];bmeds=[];bsds=[]
        for g in GEARS:
            z=Zs[g]; zz=z[[c,'hours']].dropna()
            h30=zz.groupby((zz.hours*2).astype(int))[c].median()
            mons.append(mono(h30.rolling(4,min_periods=1).median().values))
            life=zz.hours/zz.hours.max()
            traj.append(np.interp(np.linspace(0.05,1,40),life.values,zz[c].rolling(60,min_periods=10).median().bfill().values))
            ends.append(zz[zz.hours>zz.hours.max()-1][c].median())
            x=m[m.experiment==g]; b=x[(x.hours>BASE[0])&(x.hours<BASE[1])]
            rhos.append(abs(spearmanr(b[c],b.ctx_temperature,nan_policy='omit')[0]))
            bmeds.append(b[c].median()); bsds.append(b[c].std())
        T=np.array(traj); cc=np.corrcoef(T); tr=cc[np.triu_indices(len(T),1)]
        ends=np.array(ends)
        row.update(mono=r3(np.mean(mons)),trend=r3(np.nanmean(tr)),
                   prog=r3(np.exp(-np.nanstd(ends)/np.nanmean(np.abs(ends)))) if np.nanmean(np.abs(ends))>0 else None,
                   end_z=r3(np.nanmedian(ends)),temp_rho=r3(np.nanmedian(rhos)),
                   offset=r3(np.std(bmeds)/np.mean(bsds)))
    stats.append(row)
# ---------- trajectories (10-min medians of robust z) + debris
traj={}
pt['dm_total']=pt[[f'dm4500_bin_{i}' for i in range(8)]].sum(axis=1)
hmap=s1.groupby('file').test_hours.min()
pt['hours']=pt.file.map(hmap)
for g in GEARS:
    z=Zs[g]; k=(z.hours*6).astype(int)
    a=z.groupby(k).median()
    d=pt[pt.experiment==g].dropna(subset=['hours']); dk=d.groupby((d.hours*6).astype(int)).dm_total.max()
    traj[g[-1]]=dict(h=[r3(v) for v in a.hours],**{c:[None if not np.isfinite(v) else round(float(v),2) for v in a[c]] for c in CI},
        temp=[r3(v) for v in a.temp], dm_h=[r3(v/6) for v in dk.index], dm=[int(v) for v in dk.values])
# ---------- correlation (pooled robust z, 10-min)
P=pd.concat([Zs[g][CI].groupby((Zs[g].hours*6).astype(int)).median() for g in GEARS]).dropna()
C=P.rank().corr().values
order=leaves_list(linkage(1-np.abs(C[np.triu_indices(len(CI),1)]),'average'))
corr=dict(cols=[CI[i] for i in order],m=[[round(float(C[i,j]),2) for j in order] for i in order])
# ---------- novelty prototype
nov={}; configs={'base 0.5–6 h':(0.5,6.0),'base 0.5–12 h':(0.5,12.0),'base 0.5–18 h':(0.5,18.0)}
onsets={k:{} for k in configs}
for g in GEARS:
    for k,bh in configs.items():
        s,sm,thr,on=score(m[m.experiment==g],bh); onsets[k][g[-1]]=r3(on)
        if bh==BASE:
            q=sm.groupby((sm.index*6).astype(int)).median()
            nov[g[-1]]=dict(h=[round(v/6,3) for v in q.index],s=[None if not np.isfinite(v) else round(float(v),2) for v in q.values],thr=r3(thr),onset=r3(on))
# ---------- photo grid
grid={}
for g,p in ph.groupby('experiment'):
    ins=p.drop_duplicates('insp_order').sort_values('insp_order')[['inspection','insp_order','hour']]
    grid[g[-1]]=dict(insp=ins.inspection.tolist(),hour=[r3(v) for v in ins.hour],
        eye=[[(p[(p.tooth==t)&(p.insp_order==o)].eye.iloc[0] if len(p[(p.tooth==t)&(p.insp_order==o)]) else None) for o in ins.insp_order] for t in range(1,29)],
        cells=[[r3(p[(p.tooth==t)&(p.insp_order==o)].tex.max()) if len(p[(p.tooth==t)&(p.insp_order==o)]) else None for o in ins.insp_order] for t in range(1,29)])
bal=ph.cls.value_counts().to_dict()
# ---------- life split per config
ends={g[-1]:float(runs[runs.experiment==g].h1.max()) for g in GEARS}
split_out={}
for k in configs:
    rows=[];I=[];P_=[]
    for g,on in onsets[k].items():
        if on is None: continue
        rows.append(dict(gear=g,init=round(on,2),prop=round(ends[g]-on,2))); I.append(on); P_.append(ends[g]-on)
    I=np.array(I);Pp=np.array(P_);T=I+Pp;vt=T.var()
    split_out[k]=dict(rows=rows,var=dict(init=r3(I.var()/vt),prop=r3(Pp.var()/vt),cov=r3(2*np.cov(I,Pp,bias=True)[0,1]/vt)),n=len(rows))
out=dict(gears=gears,runs=[dict(gear=r.experiment[-1],run=int(r.run_num),h0=r3(r.h0),h1=r3(r.h1),temp=r3(r.temp),dup=bool(r.dup),short=bool(r.short)) for r in runs.itertuples()],
    stats=stats,traj=traj,corr=corr,nov=nov,onsets=onsets,grid=grid,balance=bal,life=split_out,
    kpi=dict(rows=int(len(s1)),files=int(s1.file.nunique()),photos=int(ph.n_photos.sum()),tooth_insp=int(len(ph)),minutes=int(len(m))))
json.dump(out,open(S+'/data.json','w'),separators=(',',':'))
print(json.dumps(gears,indent=0)[:1500]); print(bal); print(onsets); print({k:v['var'] for k,v in split_out.items()})
st=pd.DataFrame(stats); print(st[st.col.str.startswith('ci_')].sort_values('mono',ascending=False)[['col','mono','trend','prog','end_z','temp_rho','offset']].to_string())
import os; print(os.path.getsize(S+'/data.json')/1e6,'MB')
