import sys; sys.path.insert(0,'eda/scripts')
from photo_proxy import parse
import glob, numpy as np, pandas as pd
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor
def score(p):
    im=Image.open(p).convert('L').resize((320,180)).filter(ImageFilter.GaussianBlur(1.2))
    a=np.asarray(im,dtype=float)
    gy,gx=np.gradient(a); g=np.hypot(gx,gy)
    z=g[92:160,64:256]; ref=np.median(a[92:160,64:256])
    return (z>0.09*ref).mean()
def run(p):
    d=parse(p)
    if d: d['tex']=score(p)
    return d
if __name__=='__main__':
    files=sorted(glob.glob('data/train/photos/*.jpg'))
    with ProcessPoolExecutor(8) as ex: rows=[r for r in ex.map(run,files,chunksize=20) if r]
    df=pd.DataFrame(rows)
    g=df.groupby(['experiment','inspection','insp_order','tooth'],as_index=False).agg(tex=('tex','median'),n_photos=('file','size'))
    old=pd.read_csv(sys.argv[1]+'/photo_proxy.csv'); g=g.merge(old[['experiment','inspection','tooth','dark_frac']])
    g.to_csv(sys.argv[1]+'/photo_proxy.csv',index=False)
    print(g.tex.describe().round(3))
    fin=g.sort_values('insp_order').groupby(['experiment','tooth']).tail(1)
    for e,x in fin.groupby('experiment'): print(e, x.sort_values('tex',ascending=False).head(9)[['tooth','tex']].round(3).values.tolist())
