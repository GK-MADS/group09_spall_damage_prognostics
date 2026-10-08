"""Build eda/spall_figures.ipynb: one Colab cell per figure/table in the Gear Spall Data Atlas.

Usage (repo root):  python eda/scripts/build_notebook.py
"""

import nbformat as nbf

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(
    MD(
        """# Gear Spall Data Atlas: figures

One cell per figure or table in the Atlas report. Run **Setup** first; every other cell is independent.

Needs two small files from the repo: `eda/out/data.json` (≈450 KB, required) and `eda/out/thumbs.json` (≈90 KB, only for the photo-rubric cell).
Put them next to the notebook, point `DATA_DIR` at a Drive folder, or let the Setup cell prompt you to upload them.
Regenerate `data.json` from the parquet tables with the scripts in `eda/scripts/` (see `eda/README.md`)."""
    )
)

cells.append(
    CODE(
        r'''# Setup: load data.json and set a shared plotting style
import json, os, base64, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

DATA_DIR = "."  # e.g. "/content/drive/MyDrive/siads696/eda/out"

def _load(name, required=True):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        try:
            from google.colab import files  # prompts for an upload in Colab
            print(f"Upload {name}"); files.upload(); path = name
        except ImportError:
            if required:
                raise FileNotFoundError(f"{path} not found; set DATA_DIR")
            return None
    with open(path) as f:
        return json.load(f)

D = _load("data.json")

# Palette (validated for colour-vision deficiency; see the dataviz palette): categorical slots, one-hue ordinal ramp, diverging pair
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
Q0, Q1, Q2, Q3 = "#e4e6e1", "#b7d3f6", "#3987e5", "#104281"   # none, auto-flagged, initiation, spall
NEG, MID, POS = "#2a78d6", "#f0efec", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "figure.dpi": 110,
})

GEARS = ["A", "B", "C", "F", "G", "D", "E"]
GT = pd.DataFrame(D["gears"]).set_index("gear")
SPLIT, FIRST, END = GT.split.to_dict(), GT.first_dmg_h.to_dict(), GT.end_h.to_dict()
RUNS = pd.DataFrame(D["runs"])
XMAX = 100

def slog(v):
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.log10(1 + np.abs(v))

def break_gaps(x, y, gap=0.5):
    """Insert NaNs where consecutive x are further apart than `gap` hours so lines don't bridge missing runs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cut = np.where(np.diff(x) > gap)[0] + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)

def first_damage(ax, g, y):
    if FIRST.get(g) is not None and not pd.isna(FIRST[g]):
        ax.plot(FIRST[g], y, marker="D", ms=7, color=INK, mec=SURFACE, mew=1.5, zorder=5, clip_on=False)

def run_ticks(ax, g):
    tr = ax.get_xaxis_transform()
    for h0 in RUNS[(RUNS.gear == g) & (RUNS.run > 1)].h0:
        ax.plot([h0, h0], [0, 0.05], color=AXIS, lw=1, transform=tr, clip_on=False)

print({k: D["kpi"][k] for k in D["kpi"]})'''
    )
)

cells.append(MD("## Table 1: one row per gear"))
cells.append(
    CODE(
        r'''# Table 1: one row per gear
t1 = GT.reindex(GEARS).reset_index()
t1["first visible damage"] = [f"{i} ({h:.1f} h)" if isinstance(i, str) else "no photos"
                              for i, h in zip(t1.first_spall_insp, t1.first_dmg_h)]
t1 = t1.rename(columns={"gear": "Gear", "split": "Split", "runs": "Runs", "end_h": "End of record (h)",
                        "sensor_h": "Sensor hours", "steady_pct": "Steady %", "torque": "Torque (ft·lbf)",
                        "temp": "Oil temp (°F)", "rows": "1-s rows", "short_runs": "Short runs",
                        "photo_insp": "Photo inspections", "photos": "Photos"})
t1["Gear"] = "Exp-" + t1["Gear"]
cols = ["Gear", "Split", "Runs", "End of record (h)", "Sensor hours", "Steady %", "Torque (ft·lbf)", "Oil temp (°F)",
        "1-s rows", "Short runs", "Photo inspections", "Photos", "first visible damage"]
t1[cols].style.format({"End of record (h)": "{:.1f}", "Sensor hours": "{:.1f}", "Steady %": "{:.0f}",
                       "Torque (ft·lbf)": "{:.0f}", "Oil temp (°F)": "{:.0f}", "1-s rows": "{:,}"}).hide(axis="index")'''
    )
)

cells.append(MD("## Figure 1: run timeline on the cumulative operating-hour axis"))
cells.append(
    CODE(
        r'''# Figure 1: run timeline per gear (short runs pale, duplicate run hatched, photo inspections ticked)
fig, ax = plt.subplots(figsize=(11, 3.6))
for i, g in enumerate(GEARS):
    y = len(GEARS) - 1 - i
    for r in RUNS[RUNS.gear == g].itertuples():
        kw = dict(facecolor="none", edgecolor=ORANGE, hatch="////", lw=0) if r.dup else \
             dict(color=BLUE, alpha=0.4 if r.short else 1)
        ax.barh(y, r.h1 - r.h0 - 0.2, left=r.h0 + 0.1, height=0.55, **kw)
    if g in D["grid"]:
        for h in set(D["grid"][g]["hour"]):
            ax.plot([h, h], [y + 0.3, y + 0.42], color=INK2, lw=1.5)
    first_damage(ax, g, y)
ax.set_yticks(range(len(GEARS)), [f"Exp-{g} · {SPLIT[g][:5]}" for g in GEARS[::-1]])
ax.set_xlim(0, XMAX); ax.set_xlabel("cumulative operating hours"); ax.grid(axis="y", visible=False)
ax.set_title("Run timeline (≈6.15 h runs; photos taken between runs)")
ax.legend(handles=[Patch(color=BLUE, label="run"), Patch(color=BLUE, alpha=.4, label="short run (<3 h)"),
                   Patch(facecolor="none", edgecolor=ORANGE, hatch="////", label="duplicate sensor data (Exp-A Run-2)"),
                   Line2D([], [], color=INK2, lw=1.5, label="photo inspection"),
                   Line2D([], [], marker="D", ls="", color=INK, label="first visible damage")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=9)
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Table 2: channel summary statistics (steady-state 1-minute medians, all gears)"))
cells.append(
    CODE(
        r'''# Table 2: channel summary statistics
t2 = pd.DataFrame(D["stats"])[["col", "unit", "mean", "sd", "p1", "p50", "p99", "miss"]]
t2 = t2.rename(columns={"col": "Channel", "unit": "Unit", "mean": "Mean", "sd": "SD", "p1": "P1",
                        "p50": "Median", "p99": "P99", "miss": "Missing %"}).fillna({"Unit": "–"})
print(f"n = {D['kpi']['minutes']:,} steady-state minutes; missing % over all 1-s rows")
t2.style.format({c: "{:,.3g}" for c in ["Mean", "SD", "P1", "Median", "P99"]} | {"Missing %": "{:.2f}"}).hide(axis="index")'''
    )
)

cells.append(MD("## Photo rubric examples (needs `thumbs.json`)"))
cells.append(
    CODE(
        r'''# Photo rubric examples: none -> initiation -> spall, plus a shallow spall the texture score misses
from PIL import Image
T = _load("thumbs.json")
fig, axes = plt.subplots(1, len(T), figsize=(3.2 * len(T), 2.9))
for ax, t in zip(axes, T):
    ax.imshow(Image.open(io.BytesIO(base64.b64decode(t["src"].split(",", 1)[1]))))
    ax.set_title(t["c"], fontsize=9.5, loc="left", wrap=True); ax.set_xlabel(t["t"], fontsize=8, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Figure 2: tooth × inspection damage map and class balance"))
cells.append(
    CODE(
        r'''# Figure 2: tooth x inspection damage map for training gears + class balance bar
CLASS = {"none": 0, "flag": 1, "init": 2, "spall": 3}
cmap = ListedColormap([Q0, Q1, Q2, Q3])
fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 4, height_ratios=[12, 1], hspace=0.45, wspace=0.25)
for k, g in enumerate(["A", "B", "C", "F"]):
    G = D["grid"][g]; n = len(G["insp"])
    M = np.full((28, n), np.nan)
    for t in range(28):
        for j in range(n):
            v, e = G["cells"][t][j], G["eye"][t][j]
            if v is None: continue
            M[t, j] = CLASS["spall"] if e == "spall" else CLASS["init"] if e == "init" else CLASS["flag"] if v >= 0.02 else CLASS["none"]
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-.5, vmax=3.5, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, 28), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    labels = [s.replace("Run-", "R").replace("Break-In", "Brk").replace("Test-Start", "Start").replace("Pre-Run", "Pre") for s in G["insp"]]
    ax.set_xticks(range(n), labels, rotation=60, fontsize=8)
    ax.set_yticks(range(0, 28, 3), [str(t + 1) for t in range(0, 28, 3)], fontsize=8)
    ax.set_title(f"Exp-{g} · {n} inspections", fontsize=10)
    fj = next((j for j in range(n) if any(isinstance(G["eye"][t][j], str) for t in range(28))), None)
    if fj is not None:
        ax.plot(fj, -1.4, marker="D", ms=7, color=INK, mec=SURFACE, clip_on=False)
    if k == 0: ax.set_ylabel("tooth")
# class balance
b = D["balance"]; order = [("healthy", Q0, "no visible damage"), ("ambiguous", Q1, "auto-flagged, not reviewed"),
                           ("init", Q2, "initiation"), ("spall", Q3, "spall")]
tot = sum(b.values()); ax = fig.add_subplot(gs[1, :]); left = 0
for key, c, lab in order:
    v = b.get(key, 0); ax.barh(0, v / tot, left=left, color=c, height=0.8, edgecolor=SURFACE, lw=2,
                               label=f"{lab}: {v} ({100 * v / tot:.1f}%)"); left += v / tot
ax.set_xlim(0, 1); ax.axis("off")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=9)
ax.set_title(f"Class balance, all {tot:,} tooth-inspections", fontsize=10)
fig.suptitle("Tooth damage map (eye-checked on last two inspections; pale blue = texture score only)  ◆ first visible damage",
             x=0.05, ha="left", fontsize=11, fontweight="semibold")
plt.show()'''
    )
)

cells.append(MD("## Figure 3: condition-indicator trajectories, one panel per gear\nChange `CI` to any column in `D['traj']['A']` (e.g. `ci_fm4`, `ci_na4`, `ci_fleet_wtmade`) or `'dm'` for the DM4500 debris counter."))
cells.append(
    CODE(
        r'''# Figure 3: CI trajectories (robust z vs each gear's first 12 h, signed-log y), shared axes
CI = "ci_m8a"   # or "dm" for the DM4500 cumulative debris count

fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, sharey=True)
for ax, g in zip(axes.flat, GEARS):
    T = D["traj"][g]
    if CI == "dm":
        x, y = break_gaps(T["dm_h"], T["dm"])
    else:
        x, y = break_gaps(T["h"], [np.nan if v is None else v for v in T[CI]]); y = slog(y)
        ax.axhline(0, color=AXIS, lw=1)
    ax.plot(x, y, color=BLUE, lw=1.6)
    run_ticks(ax, g)
    ax.set_title(f"Exp-{g}  ·  {SPLIT[g]}", fontsize=10)
axes.flat[-1].axis("off")
ax0 = axes.flat[0]; ax0.set_xlim(0, XMAX)
if CI != "dm":
    lo, hi = ax0.get_ylim(); ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
    ax0.set_yticks(ticks, [("−" if t < 0 else "") + ("0" if t == 0 else f"10$^{{{int(abs(t))}}}$" if abs(t) > 1 else "10") for t in ticks])
for ax, g in zip(axes.flat, GEARS):
    first_damage(ax, g, ax.get_ylim()[1])
for ax in axes[-1]: ax.set_xlabel("cumulative operating hours")
fig.supylabel("DM4500 particles (cumulative)" if CI == "dm" else f"{CI[3:].upper()} robust z (signed log)", color=INK2)
fig.suptitle(f"{'DM4500 debris' if CI == 'dm' else CI[3:].upper()} per gear · ◆ first visible damage · ticks = run boundaries",
             x=0.05, ha="left", fontweight="semibold")
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Table 3: indicator quality (Coble & Hines metrics across all 7 gears)"))
cells.append(
    CODE(
        r'''# Table 3: indicator quality, sorted by trendability
q = pd.DataFrame(D["stats"]); q = q[q.col.str.startswith("ci_")]
q = q[["col", "trend", "mono", "prog", "end_z", "temp_rho", "offset"]].sort_values("trend", ascending=False)
q["col"] = q.col.str[3:].str.upper()
q.columns = ["Indicator", "Trendability", "Monotonicity", "Prognosability", "End z (median)", "Temp |ρ|", "Gear offset (SD)"]
(q.style.hide(axis="index")
  .bar(subset=["Trendability", "Monotonicity", "Prognosability"], color=Q1, vmin=0, vmax=1)
  .format({c: "{:.2f}" for c in q.columns[1:]} | {"End z (median)": "{:.1f}"}))'''
    )
)

cells.append(MD("## Figure 4: indicator redundancy (clustered Spearman correlation)"))
cells.append(
    CODE(
        r'''# Figure 4: clustered Spearman correlation of per-gear robust z (10-min medians)
C = D["corr"]; labels = [c[3:].upper() for c in C["cols"]]
div = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(np.array(C["m"]), cmap=div, vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.set_xticks(np.arange(-.5, len(labels)), minor=True); ax.set_yticks(np.arange(-.5, len(labels)), minor=True)
ax.grid(which="minor", color=SURFACE, lw=1.5); ax.grid(which="major", visible=False); ax.tick_params(which="minor", length=0)
fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
ax.set_title("Indicator redundancy: pick one or two CIs per block")
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Figure 5: novelty score per gear (Part B prototype, baseline 0.5–12 h)"))
cells.append(
    CODE(
        r'''# Figure 5: Mahalanobis novelty score per gear with threshold, onset and first visible damage
fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, sharey=True)
for ax, g in zip(axes.flat, GEARS):
    N = D["nov"][g]
    x, y = break_gaps(N["h"], [np.nan if v is None else v for v in N["s"]])
    ax.plot(x, y, color=BLUE, lw=1.6)
    ax.plot([0, END[g]], [N["thr"], N["thr"]], color=ORANGE, lw=1.5)
    if N["onset"] is not None:
        ax.axvline(N["onset"], color=INK, lw=1.3)
        ax.text(N["onset"] + 1, 60, f"onset {N['onset']:.1f} h", fontsize=8.5, color=INK2)
    run_ticks(ax, g); ax.set_title(f"Exp-{g}  ·  {SPLIT[g]}", fontsize=10)
axes.flat[-1].axis("off")
ax0 = axes.flat[0]; ax0.set_yscale("log"); ax0.set_ylim(0.4, 200); ax0.set_xlim(0, XMAX)
ax0.set_yticks([0.5, 1, 3, 10, 30, 100], ["0.5", "1", "3", "10", "30", "100"])
for ax, g in zip(axes.flat, GEARS): first_damage(ax, g, 200)
for ax in axes[-1]: ax.set_xlabel("cumulative operating hours")
fig.supylabel("novelty score √(D²/p), 30-min median", color=INK2)
fig.legend(handles=[Line2D([], [], color=BLUE, label="novelty score"),
                    Line2D([], [], color=ORANGE, label="alarm threshold (99.5th pct of baseline)"),
                    Line2D([], [], color=INK, lw=1.3, label="detected onset"),
                    Line2D([], [], marker="D", ls="", color=INK, label="first visible photo damage")],
           loc="lower right", bbox_to_anchor=(0.95, 0.08), fontsize=9)
fig.suptitle("Novelty score per gear (prototype; Exp-A gap at 6–12 h = removed duplicate run)", x=0.05, ha="left", fontweight="semibold")
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Figure 6: onset stability across baseline windows"))
cells.append(
    CODE(
        r'''# Figure 6: onset per gear under three baseline windows vs first visible photo damage
CFG = list(D["onsets"]); COLS = [BLUE, ORANGE, AQUA]
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, g in enumerate(GEARS):
    y = len(GEARS) - 1 - i
    ax.plot([0, END[g]], [y, y], color=AXIS, lw=2, solid_capstyle="round")
    ax.plot([END[g]] * 2, [y - .25, y + .25], color=INK2, lw=2)
    first_damage(ax, g, y)
    for j, k in enumerate(CFG):
        v = D["onsets"][k][g]
        if v is not None:
            ax.plot(v, y + (j - 1) * 0.18, "o", ms=8, color=COLS[j], mec=SURFACE, mew=1.5, zorder=4)
ax.set_yticks(range(len(GEARS)), [f"Exp-{g}" for g in GEARS[::-1]]); ax.grid(axis="y", visible=False)
ax.set_xlim(0, XMAX); ax.set_xlabel("cumulative operating hours")
ax.set_title("Onset moves 6–30 h with the baseline window")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=8, label=f"onset, {k}") for k, c in zip(CFG, COLS)] +
          [Line2D([], [], marker="D", ls="", color=INK, label="first visible photo damage"),
           Line2D([], [], marker="|", ls="", color=INK2, ms=12, mew=2, label="end of record")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=9)
plt.tight_layout(); plt.show()'''
    )
)

cells.append(MD("## Figure 7: life split into initiation and propagation, with variance shares"))
cells.append(
    CODE(
        r'''# Figure 7: hours to onset vs onset-to-end, sorted by total life; variance-share table
photo = pd.DataFrame([dict(gear=g, init=FIRST[g], prop=END[g] - FIRST[g]) for g in GEARS if not pd.isna(FIRST[g])])
sensor = pd.DataFrame(D["life"][CFG[1]]["rows"])

def life_bars(ax, df, title):
    df = df.assign(total=df.init + df.prop).sort_values("total", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    ax.barh(y, df.init, color=BLUE, height=0.55, edgecolor=SURFACE, lw=1)
    ax.barh(y, df.prop, left=df.init, color=ORANGE, height=0.55, edgecolor=SURFACE, lw=1)
    for yi, t in zip(y, df.total): ax.text(t + 1, yi, f"{t:.0f} h", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, "Exp-" + df.gear); ax.set_xlim(0, XMAX); ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=10); ax.set_xlabel("hours")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1]))
life_bars(a1, photo, "Training gears · onset = first visible photo damage")
life_bars(a2, sensor, f"All gears · onset = sensor prototype ({CFG[1]})")
fig.legend(handles=[Patch(color=BLUE, label="hours to onset (initiation)"),
                    Patch(color=ORANGE, label="onset to end (propagation)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9)
plt.tight_layout(); plt.show()

def shares(df):
    I, P = df.init.values, df.prop.values; vt = np.var(I + P)
    return dict(gears=len(df), init=np.var(I) / vt, prop=np.var(P) / vt, cov=2 * np.cov(I, P, bias=True)[0, 1] / vt)

rows = [("Photo, first visible damage", shares(photo))] + \
       [(f"Sensor prototype, {k}", dict(gears=D["life"][k]["n"], **D["life"][k]["var"])) for k in CFG]
vt = pd.DataFrame([dict(onset=l, **s) for l, s in rows])
vt["largest share"] = vt[["init", "prop", "cov"]].idxmax(axis=1).map({"init": "initiation", "prop": "propagation", "cov": "covariance"})
vt.columns = ["Onset definition", "Gears", "var(I) share", "var(P) share", "2·cov share", "Largest share"]
vt.style.hide(axis="index").format({c: "{:.2f}" for c in ["var(I) share", "var(P) share", "2·cov share"]})'''
    )
)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                "colab": {"provenance": []}})
nbf.write(nb, "eda/spall_figures.ipynb")
print("wrote eda/spall_figures.ipynb with", len(cells), "cells")
