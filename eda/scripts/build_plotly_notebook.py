"""Build eda/spall_figures_plotly.ipynb: interactive Plotly version of every figure/table in the Gear Spall Data Atlas.

Usage (repo root):  python eda/scripts/build_plotly_notebook.py
"""

import sys

import nbformat as nbf

sys.path.insert(0, "eda/scripts")
from colab_cells import BUILD, COMPUTE, INTRO, LOAD, PHOTOS, RUBRIC, SETUP  # noqa: E402

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    MD(INTRO),
    MD("## 1 · Load data"), CODE(LOAD),
    MD("## 2 · Build 1-second and 1-minute tables"), CODE(BUILD),
    MD("## 3 · Photo labels"), CODE(PHOTOS),
    MD("## 4 · Compute figure data"), CODE(COMPUTE),
    MD("## 5 · Plot setup"), CODE(SETUP),
    MD("## Table 1: gear inventory"),
    CODE(
        r'''# Table 1: one row per gear
t1 = pd.DataFrame(D["gears"])[["gear","split","runs","end_h","sensor_h","steady_pct","torque","temp",
                               "rows","short_runs","photo_insp","photos","first_spall_insp","first_dmg_h"]]
t1["gear"] = "Exp-" + t1.gear
t1.columns = ["Gear","Split","Runs","End of record (h)","Sensor hours","Steady %","Torque (ft·lbf)",
              "Oil temp (°F)","1-s rows","Short runs","Photo inspections","Photos","First visible damage","First damage (h)"]
t1.fillna({"First visible damage": "no photos"}).round(1)'''
    ),
    MD("## Figure 1: run timeline"),
    CODE(
        r'''# Figure 1: runs on the cumulative operating-hour axis, photo inspections and first visible damage
r = RUNS.assign(dur=RUNS.h1 - RUNS.h0 - 0.2, start=RUNS.h0 + 0.1, y="Exp-" + RUNS.gear)
fig = go.Figure(go.Bar(
    y=r.y, x=r.dur, base=r.start, orientation="h", width=0.55, name="run (≈6.15 h)",
    marker=dict(color=[C1 if not d else "white" for d in r.dup],
                opacity=[0.4 if s else 1.0 for s in r.short],
                line=dict(color=[C2 if d else C1 for d in r.dup], width=[1.5 if d else 0 for d in r.dup]),
                pattern=dict(shape=["/" if d else "" for d in r.dup], fgcolor=C2)),
    customdata=np.c_[r.run, r.h0, r.h1, r.temp, np.where(r.dup, "yes (byte copy of Run-1)", "no")],
    hovertemplate="Run-%{customdata[0]}: %{customdata[1]:.1f}–%{customdata[2]:.1f} h<br>"
                  "median oil temp %{customdata[3]:.0f} °F<br>duplicate: %{customdata[4]}<extra>%{y}</extra>"))
for g, G in D["grid"].items():
    hrs = sorted(set(G["hour"]))
    fig.add_scatter(x=hrs, y=["Exp-"+g]*len(hrs), mode="markers", name="photo inspection",
                    marker=dict(symbol="line-ns-open", size=16, color=INK2, line=dict(width=1.5)),
                    showlegend=(g == "A"), hovertemplate="inspection at %{x:.1f} h<extra></extra>")
fd = [g for g in GEARS if FIRST[g] is not None]
fig.add_scatter(x=[FIRST[g] for g in fd], y=["Exp-"+g for g in fd], mode="markers", marker=DIAMOND,
                name="first visible damage", hovertemplate="first visible damage %{x:.1f} h<extra></extra>")
fig.update_yaxes(categoryorder="array", categoryarray=["Exp-"+g for g in GEARS][::-1])
fig.update_xaxes(title="cumulative operating hours", range=[0, 102])
style(fig, "Figure 1 · Run timeline (pale = short run <3 h, hatched = duplicate sensor data)", 400).show()'''
    ),
    MD("## Table 2: channel summary statistics"),
    CODE(
        r'''# Table 2: pooled steady-state 1-minute medians, all 7 gears (missing % over all 1-s rows)
t2 = pd.DataFrame(D["stats"])[["col","unit","mean","sd","p1","p50","p99","miss"]]
t2.columns = ["Channel","Unit","Mean","SD","P1","Median","P99","Missing %"]
t2.fillna({"Unit": "–"}).style.format(precision=3, thousands=",").hide(axis="index")'''
    ),
    MD("## Rubric example photos (read from the bucket)"), CODE(RUBRIC),
    MD("## Figure 2: tooth × inspection damage map and class balance"),
    CODE(
        r'''# Figure 2: one heatmap per training gear; rows = teeth, columns = inspections
LAB = ["no visible damage", "auto-flagged, not reviewed", "initiation (eye-checked)", "spall (eye-checked)"]
cs  = [[0, Q[0]], [.25, Q[0]], [.25, Q[1]], [.5, Q[1]], [.5, Q[2]], [.75, Q[2]], [.75, Q[3]], [1, Q[3]]]
tg  = [g for g in ["A", "B", "C", "F"] if g in D["grid"]]
fig = make_subplots(rows=1, cols=len(tg), subplot_titles=[f"Exp-{g} · {len(D['grid'][g]['insp'])} inspections" for g in tg],
                    horizontal_spacing=0.04)
for i, g in enumerate(tg, 1):
    G = D["grid"][g]
    cells = np.array([[np.nan if v is None else v for v in row] for row in G["cells"]], dtype=float)
    eye = np.array([[e if isinstance(e, str) else "" for e in row] for row in G["eye"]], dtype=object)
    z = np.where(eye == "spall", 3, np.where(eye == "init", 2, np.where(np.nan_to_num(cells) >= 0.02, 1, 0))).astype(float)
    z[np.isnan(cells)] = np.nan
    x = [s.replace("Run-", "R").replace("Break-In", "Brk").replace("Test-Start", "Start").replace("Pre-Run", "Pre")
         for s in G["insp"]]
    cd = np.empty(z.shape + (2,), dtype=object)
    cd[..., 0] = np.where(np.isnan(cells), None, np.round(cells, 3))
    cd[..., 1] = [[LAB[int(k)] if not np.isnan(k) else "" for k in row] for row in z]
    fig.add_trace(go.Heatmap(z=z, x=x, y=list(range(1, 29)), zmin=-.5, zmax=3.5, colorscale=cs, showscale=False,
                             xgap=1.5, ygap=1.5, customdata=cd,
                             hovertemplate="tooth %{y} · %{x}<br>%{customdata[1]}<br>texture score %{customdata[0]}<extra></extra>"),
                  1, i)
    j = next((j for j in range(eye.shape[1]) if any(eye[:, j] != "")), None)
    if j is not None:
        fig.add_trace(go.Scatter(x=[x[j]], y=[-0.3], mode="markers", marker=DIAMOND, showlegend=False, cliponaxis=False,
                                 hovertemplate="first visible damage<extra></extra>"), 1, i)
for k, lab in enumerate(LAB):   # legend swatches
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab,
                             marker=dict(symbol="square", size=12, color=Q[k])), 1, 1)
fig.update_yaxes(autorange="reversed", showgrid=False, tickvals=list(range(1, 29, 3)))
fig.update_yaxes(title_text="tooth", col=1)
fig.update_xaxes(tickangle=-60, tickfont_size=10, showgrid=False)
style(fig, "Figure 2 · Tooth × inspection damage map (diamond = first visible damage)", 720)
fig.update_layout(legend=dict(y=-0.14, yanchor="top"), margin=dict(b=90)).show()

# Class balance across all tooth-inspections
b = D["balance"]; tot = sum(b.values())
bal = go.Figure()
for k, lab, c in [("healthy", LAB[0], Q[0]), ("ambiguous", LAB[1], Q[1]), ("init", LAB[2], Q[2]), ("spall", LAB[3], Q[3])]:
    v = b.get(k, 0)
    bal.add_bar(y=["tooth-inspections"], x=[v], orientation="h", name=f"{lab}: {v} ({100*v/tot:.1f}%)",
                marker=dict(color=c, line=dict(color="white", width=2)),
                hovertemplate=f"{lab}<br>{v} ({100*v/tot:.1f}%)<extra></extra>")
bal.update_layout(barmode="stack", yaxis_visible=False, xaxis_title="count")
style(bal, f"Class balance, all {tot:,} tooth-inspections", 240).show()'''
    ),
    MD("## Figure 3: condition-indicator trajectories\nChange `CI` (or call `plot_traj(\"ci_fm4\")`) to switch indicators; use `\"dm\"` for the DM4500 oil-debris counter."),
    CODE(
        r'''# Figure 3: robust z vs each gear's own 0.5-12 h baseline, signed-log y, shared axes.
# Change CI to any key in D["traj"]["A"] (e.g. "ci_fm4", "ci_m6a", "ci_na4"), or "dm" for DM4500 debris count.
CI = "ci_m8a"

def plot_traj(ci=CI):
    fig = make_subplots(rows=NROW, cols=2, shared_xaxes="all", shared_yaxes="all", vertical_spacing=0.06,
                        subplot_titles=[f"Exp-{g} · {SPLIT[g]}" for g in GEARS])
    ys = {}
    for g in GEARS:
        T = D["traj"][g]
        if ci == "dm":
            ys[g] = (np.asarray(T["dm_h"], float), np.asarray(T["dm"], float), np.asarray(T["dm"], float))
        else:
            raw = np.asarray([np.nan if v is None else v for v in T[ci]], float)
            ys[g] = (np.asarray(T["h"], float), raw, slog(raw))
    top = np.nanmax([np.nanmax(v[2]) for v in ys.values()])
    for i, g in enumerate(GEARS):
        row, col = i // 2 + 1, i % 2 + 1
        x, raw, y = ys[g]
        xg, yg = gapped(x, y); _, rg = gapped(x, raw)
        fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", line=dict(color=C1, width=2), customdata=rg,
                                 showlegend=False, hovertemplate="%{x:.1f} h<br>value %{customdata:.2f}<extra>Exp-" + g + "</extra>"),
                      row, col)
        for h0 in RUNS[(RUNS.gear == g) & (RUNS.run > 1)].h0:   # run boundaries
            fig.add_shape(type="line", x0=h0, x1=h0, y0=0, y1=0.06, yref="y domain" if i == 0 else f"y{i+1} domain",
                          xref="x" if i == 0 else f"x{i+1}", line=dict(color=AXIS, width=1))
        if FIRST[g] is not None:
            fig.add_trace(go.Scatter(x=[FIRST[g]], y=[top], mode="markers", marker=DIAMOND, showlegend=False,
                                     hovertemplate="first visible damage %{x:.1f} h<extra></extra>"), row, col)
    if ci != "dm":
        lo = int(np.floor(np.nanmin([np.nanmin(v[2]) for v in ys.values()]))); tv = np.arange(lo, int(np.ceil(top)) + 1)
        fig.update_yaxes(tickvals=tv, ticktext=[("−" if t < 0 else "") + ("0" if t == 0 else "10" if abs(t) == 1 else f"10<sup>{abs(t)}</sup>") for t in tv])
    fig.update_xaxes(range=[0, 100]); fig.update_xaxes(title_text="cumulative hours", row=NROW)
    ylab = "DM4500 cumulative count" if ci == "dm" else "robust z (signed log)"
    name = "DM4500 debris" if ci == "dm" else ci.replace("ci_", "").upper()
    style(fig, f"Figure 3 · {name} trajectory per gear (diamond = first visible damage; ticks = run boundaries)", 60 + 225 * NROW)
    fig.update_yaxes(title_text=ylab, col=1).show()

plot_traj(CI)'''
    ),
    MD("## Table 3: indicator quality (Coble & Hines metrics)"),
    CODE(
        r'''# Table 3: trendability, monotonicity, prognosability, end-of-life z, temperature sensitivity, between-gear offset
t3 = (pd.DataFrame([s for s in D["stats"] if s["col"].startswith("ci_")])
        [["col","trend","mono","prog","end_z","temp_rho","offset"]]
        .assign(col=lambda d: d.col.str.replace("ci_", "").str.upper())
        .rename(columns={"col":"Indicator","trend":"Trendability","mono":"Monotonicity","prog":"Prognosability",
                         "end_z":"End z (median)","temp_rho":"Temp |ρ|","offset":"Gear offset (SD)"})
        .sort_values("Trendability", ascending=False).reset_index(drop=True))
t3.style.format(precision=2).bar(subset=["Trendability","Monotonicity","Prognosability"], color="#b7d3f6", vmin=0, vmax=1).hide(axis="index")'''
    ),
    MD("## Figure 4: indicator redundancy"),
    CODE(
        r'''# Figure 4: Spearman correlation of per-gear robust z (10-min), clustered order
labs = [c.replace("ci_", "").upper() for c in D["corr"]["cols"]]
fig = go.Figure(go.Heatmap(z=D["corr"]["m"], x=labs, y=labs, zmin=-1, zmax=1, xgap=1, ygap=1,
                           colorscale=[[0, C1], [.5, "#f0efec"], [1, "#e34948"]],
                           colorbar=dict(title="ρ", thickness=12),
                           hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False); fig.update_xaxes(tickangle=-60, showgrid=False)
style(fig, "Figure 4 · Indicator redundancy (Spearman ρ): pick one or two CIs per block", 780).show()'''
    ),
    MD("## Figure 5: novelty score per gear (Part B prototype)"),
    CODE(
        r'''# Figure 5: Mahalanobis novelty score (baseline 0.5-12 h), threshold, onset, first visible damage
fig = make_subplots(rows=NROW, cols=2, shared_xaxes="all", shared_yaxes="all", vertical_spacing=0.06,
                    subplot_titles=[f"Exp-{g} · {SPLIT[g]}" for g in GEARS])
for i, g in enumerate(GEARS):
    row, col = i // 2 + 1, i % 2 + 1; N = D["nov"][g]; first = i == 0
    x, s = gapped(N["h"], [np.nan if v is None else v for v in N["s"]])
    fig.add_trace(go.Scatter(x=x, y=s, mode="lines", line=dict(color=C1, width=2), name="novelty score",
                             showlegend=first, hovertemplate="%{x:.1f} h · score %{y:.2f}<extra>Exp-" + g + "</extra>"), row, col)
    fig.add_trace(go.Scatter(x=[0, END[g]], y=[N["thr"]]*2, mode="lines", line=dict(color=C2, width=1.5),
                             name="alarm threshold (99.5th pct of baseline)", showlegend=first,
                             hovertemplate="threshold %{y:.2f}<extra></extra>"), row, col)
    if N["onset"] is not None:
        fig.add_trace(go.Scatter(x=[N["onset"]]*2, y=[0.4, 150], mode="lines", line=dict(color=INK, width=1.5),
                                 name="detected onset", showlegend=first, hovertemplate=f"onset {N['onset']:.1f} h<extra></extra>"), row, col)
    if FIRST[g] is not None:
        fig.add_trace(go.Scatter(x=[FIRST[g]], y=[120], mode="markers", marker=DIAMOND, name="first visible damage",
                                 showlegend=(g == "A"), hovertemplate="first visible damage %{x:.1f} h<extra></extra>"), row, col)
fig.update_yaxes(type="log", range=[np.log10(0.4), np.log10(150)], tickvals=[0.5, 1, 2, 5, 10, 20, 50, 100],
                 ticktext=["0.5", "1", "2", "5", "10", "20", "50", "100"])
fig.update_xaxes(range=[0, 100]); fig.update_xaxes(title_text="cumulative hours", row=NROW)
style(fig, "Figure 5 · Novelty score per gear (√(D²/p), 30-min median, log y)", 60 + 225 * NROW).show()'''
    ),
    MD("## Figure 6: onset stability across baseline windows"),
    CODE(
        r'''# Figure 6: onset per gear under three baseline windows vs first visible photo damage
CFG = list(D["onsets"]); cols = [C1, C2, C3]
ypos = {g: len(GEARS) - 1 - i for i, g in enumerate(GEARS)}
fig = go.Figure()
for g in GEARS:
    fig.add_shape(type="line", x0=0, x1=END[g], y0=ypos[g], y1=ypos[g], line=dict(color=AXIS, width=2), layer="below")
    fig.add_shape(type="line", x0=END[g], x1=END[g], y0=ypos[g]-.25, y1=ypos[g]+.25, line=dict(color=INK2, width=2))
for j, k in enumerate(CFG):
    gs = [g for g in GEARS if D["onsets"][k][g] is not None]
    fig.add_scatter(x=[D["onsets"][k][g] for g in gs], y=[ypos[g] + (j-1)*0.15 for g in gs], mode="markers",
                    name=f"onset, {k}", marker=dict(size=10, color=cols[j], line=dict(color="white", width=2)),
                    customdata=gs, hovertemplate="Exp-%{customdata}: onset %{x:.1f} h<extra>" + k + "</extra>")
fd = [g for g in GEARS if FIRST[g] is not None]
fig.add_scatter(x=[FIRST[g] for g in fd], y=[ypos[g] for g in fd], mode="markers", marker=DIAMOND,
                name="first visible photo damage", hovertemplate="first visible damage %{x:.1f} h<extra></extra>")
fig.update_yaxes(tickvals=list(ypos.values()), ticktext=["Exp-"+g for g in ypos], range=[-.6, len(GEARS)-.4], showgrid=False)
fig.update_xaxes(title="cumulative hours (tick = end of record)", range=[0, 102])
style(fig, "Figure 6 · Onset stability across baseline windows", 440).show()'''
    ),
    MD("## Figure 7: life split into initiation and propagation, with variance shares"),
    CODE(
        r'''# Figure 7: hours to onset (initiation) + onset to end (propagation), sorted by total life
CFG = list(D["onsets"])
photo = [dict(gear=g, init=FIRST[g], prop=END[g]-FIRST[g]) for g in GEARS if FIRST[g] is not None]
sensor = D["life"][CFG[1]]["rows"]                     # baseline 0.5-12 h
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                    subplot_titles=["Training gears · onset = first visible photo damage",
                                    f"All gears · onset = sensor prototype ({CFG[1]})"])
for c, rows in enumerate([photo, sensor], 1):
    rows = sorted(rows, key=lambda r: r["init"] + r["prop"])
    y = ["Exp-" + r["gear"] for r in rows]
    fig.add_bar(y=y, x=[r["init"] for r in rows], orientation="h", name="hours to onset (initiation)",
                marker=dict(color=C1, line=dict(color="white", width=2)), width=0.6, showlegend=(c == 1),
                hovertemplate="%{y}: %{x:.1f} h to onset<extra></extra>", row=1, col=c)
    fig.add_bar(y=y, x=[r["prop"] for r in rows], orientation="h", name="hours from onset to end (propagation)",
                marker=dict(color=C2, line=dict(color="white", width=2)), width=0.6, showlegend=(c == 1),
                text=[f"{r['init']+r['prop']:.0f} h" for r in rows], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.1f} h onset to end<extra></extra>", row=1, col=c)
fig.update_layout(barmode="stack"); fig.update_xaxes(range=[0, 110], title_text="hours"); fig.update_yaxes(showgrid=False)
style(fig, "Figure 7 · Life split into initiation and propagation (longest life at top)", 460)
fig.update_layout(legend=dict(y=-0.2, yanchor="top")).show()

# Share of var(total life): var(I) + var(P) + 2·cov(I,P) = 1
def shares(rows):
    I = np.array([r["init"] for r in rows]); P = np.array([r["prop"] for r in rows]); vt = (I + P).var()
    return dict(n=len(rows), init=I.var()/vt, prop=P.var()/vt, cov=2*np.cov(I, P, bias=True)[0, 1]/vt)
vt = pd.DataFrame([dict(onset="Photo, first visible damage", **shares(photo))] +
                  [dict(onset=f"Sensor prototype, {k}", n=D["life"][k]["n"], **D["life"][k]["var"]) for k in CFG])
vt["largest"] = vt[["init","prop","cov"]].idxmax(axis=1).map({"init":"initiation","prop":"propagation","cov":"covariance"})
vt.rename(columns={"onset":"Onset definition","n":"Gears","init":"var(I) share","prop":"var(P) share",
                   "cov":"2·cov share","largest":"Largest share"}).round(2)'''
    ),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                "colab": {"provenance": []}})
nbf.write(nb, "eda/spall_figures_plotly.ipynb")
print("wrote eda/spall_figures_plotly.ipynb with", len(cells), "cells")
