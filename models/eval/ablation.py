"""Step 1.1: remove or alter one component at a time and measure the change in T1/T2 pooled MSE.

Usage (repo root):  .venv/bin/python -m models.eval.ablation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spall_hi import FEATS, TOOTH

from .adapters import HI, IForest, RidgeApproach
from .harness import OUT, evaluate_runs, load_dataset, record, run_folds, save_fig, with_debris_feature


def _drop(cols, c):
    return [x for x in cols if x != c]


def cases(ds, ds_nocorr, ds_pooled, ds_unsteady, ds_debris):
    """(approach, ablation, factory, dataset) for every ablation in the spec."""
    full = {"hi_tooth": lambda **k: HI("tooth", **k), "hi_maha": lambda **k: HI("maha", **k), "iforest": lambda **k: IForest(**k)}
    out = []
    for name, mk in full.items():
        feat_kw = "tooth" if name == "hi_tooth" else "feats"
        base_cols = TOOTH if name == "hi_tooth" else FEATS
        out.append((name, "full", lambda mk=mk: mk(), ds))
        alt_cols = FEATS if name == "hi_tooth" else TOOTH
        out.append((name, "all CIs" if name == "hi_tooth" else "tooth CIs only", lambda mk=mk, kw=feat_kw, c=alt_cols: mk(**{kw: c}), ds))
        for c in TOOTH:
            out.append((name, f"drop {c[3:]}", lambda mk=mk, kw=feat_kw, c=_drop(base_cols, c): mk(**{kw: c}), ds))
        out.append((name, "+ debris", lambda mk=mk, kw=feat_kw, c=base_cols + ["debris_z"]: mk(**{kw: c}), ds_debris))
        out.append((name, "no condition correction", lambda mk=mk: mk(), ds_nocorr))
        out.append((name, "pooled standardisation", lambda mk=mk: mk(), ds_pooled))
        out.append((name, "no steady filter", lambda mk=mk: mk(), ds_unsteady))
        for mono in ["none", "cummax"]:
            out.append((name, f"monotone {mono}", lambda mk=mk, m=mono: mk(monotone=m), ds))
    out.append(("ridge_tooth", "full", lambda: RidgeApproach(TOOTH), ds))
    out.append(("ridge_tooth", "no sign split", lambda: RidgeApproach(TOOTH, sign_split=False), ds))
    out.append(("ridge_tooth", "no log compression", lambda: RidgeApproach(TOOTH, log=False), ds))
    out.append(("ridge_tooth", "unconstrained sign", lambda: RidgeApproach(TOOTH, positive=False), ds))
    return out


def main(ds=None) -> pd.DataFrame:
    ds = ds or load_dataset()
    variants = dict(
        ds_nocorr=load_dataset(minutes=ds.minutes, correct=False),
        ds_pooled=load_dataset(minutes=ds.minutes, per_gear=False),
        ds_unsteady=load_dataset(steady=False),
        ds_debris=with_debris_feature(ds),
    )
    rows = []
    for name, abl, make, d in cases(ds, **variants):
        fr = run_folds(make, d)
        rows.append(dict(approach=name, ablation=abl, t1_mse_pooled=evaluate_runs(fr.runs, "photo")[0]["mse_pooled"],
                         t2_mse_pooled=evaluate_runs(fr.runs, "mean_ex")[0]["mse_pooled"]))
        print(rows[-1])
    t = pd.DataFrame(rows)
    full = t[t.ablation == "full"].set_index("approach")
    t["d_t1"] = t.t1_mse_pooled - t.approach.map(full.t1_mse_pooled)
    t["d_t2"] = t.t2_mse_pooled - t.approach.map(full.t2_mse_pooled)
    t.round(5).to_csv(OUT / "ablation.csv", index=False)

    p = t[t.ablation != "full"].pivot(index="approach", columns="ablation", values="d_t1")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    lim = np.nanmax(np.abs(p.to_numpy()))
    im = ax.imshow(p.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(p.shape[1]), p.columns, rotation=45, ha="right")
    ax.set_yticks(range(p.shape[0]), p.index)
    for (i, j), v in np.ndenumerate(p.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ T1 pooled MSE (+ = worse without it)")
    ax.set_title("Ablation: change in T1 error when one component is removed or altered")
    save_fig(fig, "fig_ablation_heatmap.png")
    plt.close(fig)

    nf = t[t.ablation != "full"]
    best, worst = nf.loc[nf.d_t1.idxmin()], nf.loc[nf.d_t1.idxmax()]
    record(ablation_largest_t1_gain=f"{best.approach}: {best.ablation} ({best.d_t1:+.4f})",
           ablation_largest_t1_loss=f"{worst.approach}: {worst.ablation} ({worst.d_t1:+.4f})")
    return t


if __name__ == "__main__":
    main()
