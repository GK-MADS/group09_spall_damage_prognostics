import re, glob, numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
ORDER={'Break-In':-2,'Pre-Run':-1,'Test-Start':-1}
def parse(p):
    n=p.split('/')[-1]
    m=re.match(r'(Exp-\w)_(Break-In|Test-Start|Pre-Run|Run-\d+)_(.*)\.jpg',n); e,insp,rest=m.groups()
    t=re.findall(r'Tooth_?(\d+)',rest)
    if not t: return None
    tooth=int(t[0]) if 'All_Teeth' not in rest else int(re.findall(r'Tooth_?(\d+)',rest)[0])
    return dict(experiment=e,inspection=insp,insp_order=ORDER.get(insp, int(insp.split('-')[1]) if insp.startswith('Run') else 0),tooth=tooth,file=n,closeup='WIN' in rest)
def score(p):
    im=np.asarray(Image.open(p).convert('L').resize((320,180)),dtype=float)
    face=im[:, 48:272]
    ref=np.median(face[80:170])
    zone=face[85:176]            # flank below the tip band
    dark=(zone<0.55*ref)
    return dark.mean()
def run(p):
    d=parse(p)
    if d: d['dark_frac']=score(p)
    return d
if __name__=='__main__':
    files=sorted(glob.glob('data/train/photos/*.jpg'))
    with ProcessPoolExecutor(8) as ex: rows=[r for r in ex.map(run,files,chunksize=20) if r]
    df=pd.DataFrame(rows)
    # one value per tooth/inspection: median over repeat shots
    g=df.groupby(['experiment','inspection','insp_order','tooth'],as_index=False).agg(dark_frac=('dark_frac','median'),n_photos=('file','size'))
    g.to_csv('eda/out/photo_proxy.csv',index=False)
    print(g.shape); print(g.dark_frac.describe())
    fin=g.sort_values('insp_order').groupby(['experiment','tooth']).tail(1)
    for e,x in fin.groupby('experiment'): print(e, x.sort_values('dark_frac',ascending=False).head(8)[['tooth','inspection','dark_frac']].round(3).values.tolist())
