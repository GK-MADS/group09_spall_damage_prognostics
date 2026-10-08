INTRO = '''# Gear Spall Data Atlas: interactive figures (Plotly)

Builds every figure and table in the Atlas report from the flattened parquet tables in the team bucket.

**Run order:**
1. **Load data** reuses your `lf`, `particles` and `channels` if you've already loaded them, and adds the validation and test splits.
2. **Build tables** removes the duplicate Exp-A Run-2, flags steady-state seconds and makes the 1-minute table.
3. **Photo labels** reads `photo_proxy.csv` if it's in the bucket; otherwise it scores the tooth photos (slower, about 1 GB).
4. **Compute** builds `D`, the dict every figure cell reads.
5. **Plot setup**, then any figure cell in any order.

Colab already has everything this needs (pandas, gcsfs, scikit-learn, scipy, Pillow, Plotly).'''

LOAD = r'''# 1 · Load data: reuse lf / particles / channels if already loaded; add the validation and test splits
import pandas as pd, numpy as np
try:
    from google.colab import auth
    auth.authenticate_user()          # pick your @umich.edu account in the popup
except ImportError:
    pass                              # not in Colab: DATA_ROOT can be a local path

DATA_ROOT = "gs://siads696-phm2026/data"
SPLITS = ["train", "validation", "test"]

def _read(path, reader=pd.read_parquet):
    try:
        return reader(path)
    except (FileNotFoundError, OSError) as e:
        print(f"  skipped {path} ({type(e).__name__})"); return None

LF = {"train": lf} if "lf" in globals() else {}
PT = {"train": particles} if "particles" in globals() else {}
for s in SPLITS:
    if s not in LF: LF[s] = _read(f"{DATA_ROOT}/{s}/lf_1s.parquet")
    if s not in PT: PT[s] = _read(f"{DATA_ROOT}/{s}/particles.parquet")
LF = {s: v for s, v in LF.items() if v is not None}; PT = {s: v for s, v in PT.items() if v is not None}
if "channels" not in globals():
    channels = pd.read_csv(f"{DATA_ROOT}/channels.csv")
for s, v in LF.items():
    print(f"{s:<11} {len(v):>9,} one-second rows · gears {sorted(v.experiment.unique())}")'''

BUILD = r'''# 2 · Build tables: drop the duplicate run, flag steady-state seconds, 1-minute medians
frames, parts = [], []
for s, x in LF.items():
    x = x.copy(); x["split"] = s
    # Exp-A Run-2 files are byte copies of Run-1: blank their values but keep the nominal hour axis for photo alignment
    dup = (x.experiment == "Exp-A") & (x.run_num == 2)
    x.loc[dup, [c for c in x if c.startswith(("ci_", "ctx_"))]] = np.nan
    x["minute"] = (x.test_hours * 60).astype(int)
    frames.append(x)
    p = PT[s].copy(); p["split"] = s
    parts.append(p[~((p.experiment == "Exp-A") & (p.run_num == 2))])
S1 = pd.concat(frames, ignore_index=True)
PART = pd.concat(parts, ignore_index=True)
CI  = [c for c in S1 if c.startswith("ci_")]
CTX = [c for c in S1 if c.startswith("ctx_")]

# steady state: within 20 rpm of 1,400 rpm and within 5% of the gear's nominal (median) torque
nom_t = S1.groupby("experiment").ctx_pau_torque.transform("median")
S1["steady"] = (S1.ctx_pau_speed.sub(1400).abs() < 20) & (S1.ctx_pau_torque.sub(nom_t).abs() < 0.05 * nom_t)
M1 = S1[S1.steady].groupby(["split", "experiment", "run_num", "minute"])[CI + CTX].median().reset_index()
M1["hours"] = M1.minute / 60
print(f"{len(S1):,} one-second rows · {S1.steady.mean():.1%} steady · {len(M1):,} one-minute rows")'''

PHOTOS = r'''# 3 · Photo labels: texture score per tooth-inspection + eye-checked initiation / spall labels
import re, io, fsspec
from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor

PHOTO_PROXY = "gs://siads696-phm2026/eda/photo_proxy.csv"   # copy of eda/out/photo_proxy.csv (fast path)
PHOTO_DIR   = f"{DATA_ROOT}/train/photos"                     # fallback: score the photos here (~1 GB)
ORDER = {"Break-In": -2, "Pre-Run": -1, "Test-Start": -1}

def parse_photo(name):
    m = re.match(r"(Exp-\w)_(Break-In|Test-Start|Pre-Run|Run-\d+)_(.*)\.jpg", name)
    if not m: return None
    e, insp, rest = m.groups(); t = re.findall(r"Tooth_?(\d+)", rest)
    if not t: return None
    return dict(experiment=e, inspection=insp, tooth=int(t[0]), file=name,
                insp_order=ORDER.get(insp, int(insp.split("-")[1]) if insp.startswith("Run") else 0))

def texture_score(img):
    """Share of high-gradient pixels on the flank below the contact band: large spalls score high."""
    a = np.asarray(img.convert("L").resize((320, 180)).filter(ImageFilter.GaussianBlur(1.2)), dtype=float)
    gy, gx = np.gradient(a); g = np.hypot(gx, gy)
    return float((g[92:160, 64:256] > 0.09 * np.median(a[92:160, 64:256])).mean())

PH = _read(PHOTO_PROXY, pd.read_csv)
if PH is None:
    fs, root = fsspec.core.url_to_fs(PHOTO_DIR)
    paths = fs.glob(root.rstrip("/") + "/*.jpg")
    if not paths:
        raise FileNotFoundError(f"No photo_proxy.csv at {PHOTO_PROXY} and no .jpg files under {PHOTO_DIR}")
    print(f"scoring {len(paths)} photos from {PHOTO_DIR} ...")
    def _one(p):
        d = parse_photo(p.split("/")[-1])
        if d:
            with fs.open(p, "rb") as f: d["tex"] = texture_score(Image.open(io.BytesIO(f.read())))
        return d
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(_one, paths) if r]
    PH = (pd.DataFrame(rows).groupby(["experiment", "inspection", "insp_order", "tooth"], as_index=False)
            .agg(tex=("tex", "median"), n_photos=("file", "size")))
PH = PH[["experiment", "inspection", "insp_order", "tooth", "tex", "n_photos"]].copy()
PH["tooth"] = PH.tooth.astype(int)

# Checked by eye on each gear's last two inspections; everything else relies on the texture score
EYE = {("Exp-A",16,4):"init", ("Exp-A",28,4):"init", ("Exp-B",11,6):"init", ("Exp-B",14,6):"init",
       ("Exp-F",9,7):"init", ("Exp-F",12,7):"init", ("Exp-F",13,7):"init",
       **{k: "spall" for k in [("Exp-A",16,5), ("Exp-A",24,5), ("Exp-A",28,5), ("Exp-B",11,7), ("Exp-B",14,7),
                               ("Exp-C",6,12), ("Exp-C",7,12), ("Exp-F",9,8), ("Exp-F",12,8), ("Exp-F",13,8),
                               ("Exp-F",20,8), ("Exp-F",19,8), ("Exp-F",18,6), ("Exp-F",18,7), ("Exp-F",18,8)]}}
PH["eye"] = [EYE.get(k) for k in zip(PH.experiment, PH.tooth, PH.insp_order)]
PH["cls"] = np.select([PH.eye == "spall", PH.eye == "init", PH.tex >= 0.02], ["spall", "init", "ambiguous"], "healthy")
print(f"{len(PH):,} tooth-inspections from {int(PH.n_photos.sum()):,} photos:", PH.cls.value_counts().to_dict())'''

COMPUTE = r'''# 4 · Compute D: gear table, channel stats, indicator metrics, trajectories, correlation, novelty prototype
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

ORDER_G = ["Exp-A", "Exp-B", "Exp-C", "Exp-F", "Exp-G", "Exp-D", "Exp-E"]
EXPS = [g for g in ORDER_G if g in set(M1.experiment)]
BASE = (0.5, 12.0)                               # healthy baseline window (hours)
r3 = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)
nums = lambda a, d=2: [None if v is None or not np.isfinite(v) else round(float(v), d) for v in a]

# runs and photo-inspection hours (a photo set is taken after each run; Break-In / Test-Start sit at 0 h)
runs = S1.groupby(["experiment", "run_num"]).agg(h0=("test_hours", "min"), h1=("test_hours", "max"),
                                                 temp=("ctx_temperature", "median")).reset_index()
runs["dup"] = (runs.experiment == "Exp-A") & (runs.run_num == 2)
runs["short"] = (runs.h1 - runs.h0) < 3
H1 = runs.set_index(["experiment", "run_num"]).h1
insp_hour = lambda e, o: 0.0 if o <= 0 else float(H1[(e, o)])
PH["hour"] = [insp_hour(e, o) for e, o in zip(PH.experiment, PH.insp_order)]

# gear table
split = M1.groupby("experiment").split.first()
gears = []
for g in EXPS:
    x, r, p = S1[S1.experiment == g], runs[runs.experiment == g], PH[PH.experiment == g]
    fo = p[p.eye.notna()].insp_order.min() if len(p) else np.nan
    gears.append(dict(gear=g[-1], split=split[g], runs=int(r.run_num.nunique()), end_h=r3(r.h1.max()),
        sensor_h=r3(x.ci_fm4.notna().sum() / 3600), steady_pct=r3(x.steady.mean() * 100),
        torque=r3(x.ctx_pau_torque.median()), temp=r3(x.ctx_temperature.median()), files=int(x.file.nunique()),
        rows=int(len(x)), short_runs=int(r.short.sum()), photo_insp=int(p.inspection.nunique()),
        photos=int(p.n_photos.sum()) if len(p) else 0,
        first_spall_insp=None if pd.isna(fo) else p[p.insp_order == fo].inspection.iloc[0],
        first_dmg_h=None if pd.isna(fo) else r3(insp_hour(g, int(fo)))))

# robust z per gear against its own baseline: (x - median) / (1.4826 * MAD)
Zs = {}
for g in EXPS:
    x = M1[M1.experiment == g].sort_values("hours"); b = x[(x.hours > BASE[0]) & (x.hours < BASE[1])]
    med = b[CI].median(); mad = (b[CI] - med).abs().median() * 1.4826
    z = (x[CI] - med) / mad.replace(0, np.nan); z["hours"] = x.hours.values; z["temp"] = x.ctx_temperature.values
    Zs[g] = z

# channel stats + Coble & Hines metrics
def mono(v):
    d = np.sign(np.diff(v)); d = d[d != 0]; return abs(d.sum()) / max(len(v) - 1, 1)
units = channels.set_index("column").unit
stats = []
for c in CI + CTX:
    v = M1[c].dropna()
    row = dict(col=c, unit=units.get(c), mean=r3(v.mean()), sd=r3(v.std()), p1=r3(v.quantile(.01)),
               p50=r3(v.median()), p99=r3(v.quantile(.99)), miss=r3(S1[c].isna().mean() * 100))
    if c in CI:
        mons, traj, ends, rhos, bmeds, bsds = [], [], [], [], [], []
        for g in EXPS:
            zz = Zs[g][[c, "hours"]].dropna()
            mons.append(mono(zz.groupby((zz.hours * 2).astype(int))[c].median().rolling(4, min_periods=1).median().values))
            traj.append(np.interp(np.linspace(0.05, 1, 40), (zz.hours / zz.hours.max()).values,
                                  zz[c].rolling(60, min_periods=10).median().bfill().values))
            ends.append(zz[zz.hours > zz.hours.max() - 1][c].median())
            x = M1[M1.experiment == g]; b = x[(x.hours > BASE[0]) & (x.hours < BASE[1])]
            rhos.append(abs(spearmanr(b[c], b.ctx_temperature, nan_policy="omit")[0]))
            bmeds.append(b[c].median()); bsds.append(b[c].std())
        cc = np.corrcoef(np.array(traj)); ends = np.array(ends)
        row.update(mono=r3(np.mean(mons)), trend=r3(np.nanmean(cc[np.triu_indices(len(traj), 1)])),
                   prog=r3(np.exp(-np.nanstd(ends) / np.nanmean(np.abs(ends)))), end_z=r3(np.nanmedian(ends)),
                   temp_rho=r3(np.nanmedian(rhos)), offset=r3(np.std(bmeds) / np.mean(bsds)))
    stats.append(row)

# 10-minute trajectories + DM4500 cumulative debris
PART["dm_total"] = PART[[f"dm4500_bin_{i}" for i in range(8)]].sum(axis=1)
PART["hours"] = PART.file.map(S1.groupby("file").test_hours.min())
traj = {}
for g in EXPS:
    a = Zs[g].groupby((Zs[g].hours * 6).astype(int)).median()
    d = PART[PART.experiment == g].dropna(subset=["hours"]); dk = d.groupby((d.hours * 6).astype(int)).dm_total.max()
    traj[g[-1]] = dict(h=nums(a.hours, 3), temp=nums(a.temp, 3), dm_h=nums(dk.index / 6, 3), dm=[int(v) for v in dk.values],
                       **{c: nums(a[c]) for c in CI})

# Spearman correlation of pooled 10-minute robust z, ordered by hierarchical clustering
P = pd.concat([Zs[g][CI].groupby((Zs[g].hours * 6).astype(int)).median() for g in EXPS]).dropna()
Cm = P.rank().corr().values
order = leaves_list(linkage(1 - np.abs(Cm[np.triu_indices(len(CI), 1)]), "average"))
corr = dict(cols=[CI[i] for i in order], m=[[round(float(Cm[i, j]), 2) for j in order] for i in order])

# Part B prototype: temperature-regressed, Ledoit-Wolf Mahalanobis novelty score + persistence onset rule
FEATS = ["ci_fm4", "ci_m6a", "ci_m8a", "ci_na4", "ci_e_ratio", "ci_kurtosis_tsa", "ci_rms_tsa", "ci_alr"]
def novelty(g, base_h=BASE, q=0.995, persist=60):
    g = g.dropna(subset=FEATS + ["ctx_temperature"])
    X = np.log(g[FEATS].abs() + 1e-9).values
    base = ((g.hours > base_h[0]) & (g.hours < base_h[1])).values
    T = np.c_[np.ones(len(g)), g.ctx_temperature]
    R = X - T @ np.linalg.lstsq(T[base], X[base], rcond=None)[0]
    Z = (R - R[base].mean(0)) / R[base].std(0)
    s = pd.Series(np.sqrt(LedoitWolf().fit(Z[base]).mahalanobis(Z) / len(FEATS)), index=g.hours.values)
    sm = s.rolling(30, min_periods=10).median()
    thr = float(np.quantile(sm[base].dropna(), q))
    above = (sm > thr) & (sm.index > base_h[1])
    run = above.astype(int).groupby((~above).cumsum()).cumsum(); hit = run[run >= persist]
    return sm, thr, (float(hit.index[0] - persist / 60) if len(hit) else None)

configs = {"base 0.5–6 h": (0.5, 6.0), "base 0.5–12 h": (0.5, 12.0), "base 0.5–18 h": (0.5, 18.0)}
nov, onsets = {}, {k: {} for k in configs}
for g in EXPS:
    for k, bh in configs.items():
        sm, thr, on = novelty(M1[M1.experiment == g], bh); onsets[k][g[-1]] = r3(on)
        if bh == BASE:
            q = sm.groupby((sm.index * 6).astype(int)).median()
            nov[g[-1]] = dict(h=nums(q.index / 6, 3), s=nums(q.values), thr=r3(thr), onset=r3(on))

# photo grid (tooth x inspection) and class balance
grid = {}
for g, p in PH.groupby("experiment"):
    ins = p.drop_duplicates("insp_order").sort_values("insp_order")
    key = p.set_index(["tooth", "insp_order"])
    cell = lambda t, o, col: (key.loc[(t, o), col] if (t, o) in key.index else None)
    grid[g[-1]] = dict(insp=ins.inspection.tolist(), hour=nums(ins.hour, 3),
                       eye=[[e if isinstance(e := cell(t, o, "eye"), str) else None for o in ins.insp_order] for t in range(1, 29)],
                       cells=[[r3(cell(t, o, "tex")) for o in ins.insp_order] for t in range(1, 29)])

# life split: hours to onset vs onset to end, with variance shares
END_H = runs.groupby("experiment").h1.max()
life = {}
for k in configs:
    rows = [dict(gear=g, init=round(on, 2), prop=round(END_H["Exp-" + g] - on, 2)) for g, on in onsets[k].items() if on is not None]
    I = np.array([r["init"] for r in rows]); Pp = np.array([r["prop"] for r in rows]); vt = (I + Pp).var()
    life[k] = dict(rows=rows, n=len(rows), var=dict(init=r3(I.var() / vt), prop=r3(Pp.var() / vt),
                                                    cov=r3(2 * np.cov(I, Pp, bias=True)[0, 1] / vt)))

D = dict(gears=gears, stats=stats, traj=traj, corr=corr, nov=nov, onsets=onsets, grid=grid, life=life,
         balance=PH.cls.value_counts().to_dict(),
         runs=[dict(gear=r.experiment[-1], run=int(r.run_num), h0=r3(r.h0), h1=r3(r.h1), temp=r3(r.temp),
                    dup=bool(r.dup), short=bool(r.short)) for r in runs.itertuples()],
         kpi=dict(rows=len(S1), files=int(S1.file.nunique()), photos=int(PH.n_photos.sum()),
                  tooth_insp=len(PH), minutes=len(M1)))
print("gears:", [g["gear"] for g in gears], "· onsets (0.5–12 h):", onsets["base 0.5–12 h"])'''

SETUP = r'''# 5 · Plot setup: shared Plotly styling (reads D from the Compute cell)
import json, math, plotly.graph_objects as go
from plotly.subplots import make_subplots
# D = json.load(open("data.json"))   # alternative: skip cells 1-4 and load the precomputed eda/out/data.json

GI    = {g["gear"]: g for g in D["gears"]}
GEARS = [g for g in ["A", "B", "C", "F", "G", "D", "E"] if g in GI]
SPLIT = {g: GI[g]["split"] for g in GEARS}
FIRST = {g: GI[g]["first_dmg_h"] for g in GEARS}      # None for gears without photos
END   = {g: GI[g]["end_h"] for g in GEARS}
RUNS  = pd.DataFrame(D["runs"])
NROW  = math.ceil(len(GEARS) / 2)                      # small-multiple grid rows

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"                 # categorical slots 1-3 (validated)
Q = ["#e4e6e1", "#b7d3f6", "#3987e5", "#104281"]              # none / auto-flag / initiation / spall
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

def style(fig, title, height=None):
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0, font=dict(size=15, color=INK)),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK2),
                      margin=dict(l=60, r=20, t=90, b=40), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"))
    if height: fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

def slog(v):
    v = np.asarray(v, dtype=float); return np.sign(v) * np.log10(1 + np.abs(v))

def gapped(x, y, gap=0.5):
    """Insert NaN where time jumps > gap hours so lines don't bridge missing data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    br = np.where(np.diff(x) > gap)[0] + 1
    return np.insert(x, br, np.nan), np.insert(y, br, np.nan)

DIAMOND = dict(symbol="diamond", size=11, color=INK, line=dict(color="white", width=2))
print(D["kpi"])'''

RUBRIC = r'''# Rubric examples: no damage -> initiation -> spall, plus a shallow spall the texture score misses
RUBRIC = [("No visible damage", "Exp-B tooth 11 · after Run-5", "Exp-B_Run-5_All_Teeth_Tooth_11.jpg"),
          ("Initiation: small flake below the contact band", "Exp-B tooth 11 · after Run-6", "Exp-B_Run-6_All_Teeth_Tooth_11.jpg"),
          ("Spall: large crater across the flank", "Exp-B tooth 11 · after Run-7", "Exp-B_Run-7_All_Teeth_Tooth_11.jpg"),
          ("Shallow spall the texture score misses", "Exp-F tooth 19 · after Run-8", "Exp-F_Run-8_Tooth_19.jpg")]
imgs = []
for title, sub, name in RUBRIC:
    try:
        with fsspec.open(f"{PHOTO_DIR}/{name}", "rb") as f:
            im = Image.open(io.BytesIO(f.read())); w, h = im.size
            imgs.append((title, sub, np.asarray(im.crop((int(w * .12), int(h * .15), int(w * .88), int(h * .95))).reduce(4))))
    except (FileNotFoundError, OSError):
        print(f"  {name} not found under {PHOTO_DIR}")
if imgs:
    fig = make_subplots(rows=1, cols=len(imgs), subplot_titles=[f"{t}<br><sub>{s}</sub>" for t, s, _ in imgs],
                        horizontal_spacing=0.02)
    for i, (_, _, a) in enumerate(imgs, 1):
        fig.add_trace(go.Image(z=a, hoverinfo="skip"), 1, i)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    style(fig, "Tooth photo rubric examples", 340).update_annotations(font_size=11).show()'''
