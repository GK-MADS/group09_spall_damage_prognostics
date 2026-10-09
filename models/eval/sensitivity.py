"""Step 1.2: one-at-a-time sweeps around the chosen settings, one line per held-out gear.

Usage (repo root):  .venv/bin/python -m models.eval.sensitivity
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spall_hi import FEATS, TOOTH

from .adapters import HI, IForest, LabelTuned, RidgeApproach
from .harness import OUT, SEED, evaluate_runs, load_dataset, record, run_folds, save_fig

ALPHAS = np.logspace(-3, 4, 13)


def _tick(v) -> str:
    try:
        return f"{float(f'{float(v):.2g}'):g}"
    except ValueError:
        return str(v)


def _rows(approach, param, value, fr, coef_norm=np.nan):
    s1, g1 = evaluate_runs(fr.runs, "photo")
    s2, _ = evaluate_runs(fr.runs, "mean_ex")
    return [dict(approach=approach, param=param, value=value, gear=g, gear_err=e, t1_mse_pooled=s1["mse_pooled"],
                 t2_mse_pooled=s2["mse_pooled"], coef_norm=coef_norm) for g, e in g1.items()]


def main(ds=None) -> pd.DataFrame:
    ds = ds or load_dataset()
    rows = []
    for v in [1.5, 3.0, 4.5, 6.0]:
        d = load_dataset(minutes=ds.minutes, base_h=v)
        rows += _rows("hi_tooth", "base_h", v, run_folds(lambda: HI("tooth"), d))
    for v in [10, 30, 60, 120, 240]:
        rows += _rows("hi_tooth", "tail", v, run_folds(lambda v=v: HI("tooth", tail=v), ds))
    for v in [0.0, 0.25, 0.5, 1.0]:
        d = load_dataset(minutes=ds.minutes, break_in_h=v)
        rows += _rows("hi_tooth", "break_in_h", v, run_folds(lambda: HI("tooth"), d))
    for feats, nm in [(TOOTH, "ridge_tooth"), (FEATS, "ridge_all")]:
        for a in ALPHAS:
            fr = run_folds(lambda a=a, f=feats: RidgeApproach(f, alpha=a), ds)
            norm = float(np.mean([np.linalg.norm(m.m.coef_) for m in fr.models.values()]))
            rows += _rows(nm, "alpha", float(a), fr, norm)
    for k, vals in dict(n_estimators=[50, 100, 200, 500], max_samples=[64, 256, 1024, "auto"], max_features=[0.25, 0.5, 1.0]).items():
        for v in vals:
            rows += _rows("iforest", k, str(v), run_folds(lambda k=k, v=v: IForest(**{k: v}), ds))
    for s in range(5):
        rows += _rows("iforest", "seed", SEED + s, run_folds(lambda s=s: IForest(seed=SEED + s), ds))
    t = pd.DataFrame(rows)
    t.round(6).to_csv(OUT / "sensitivity.csv", index=False)

    fr = run_folds(lambda: LabelTuned(ds.minutes, nested=True), ds)
    win = pd.DataFrame([dict(held_out=g, **{k: m.selected[k] for k in ["tooth", "form", "base_h", "tail", "monotone"]})
                        for g, m in fr.models.items()])
    win.to_csv(OUT / "sensitivity_tuned_winners.csv", index=False)

    panels = [("hi_tooth", "base_h"), ("hi_tooth", "tail"), ("hi_tooth", "break_in_h"), ("ridge_tooth", "alpha"),
              ("ridge_all", "alpha"), ("iforest", "n_estimators"), ("iforest", "max_samples"), ("iforest", "max_features")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey=True)
    for ax, (ap, pa) in zip(axes.flat, panels):
        d = t[(t.approach == ap) & (t.param == pa)]
        for g, dg in d.groupby("gear"):
            ax.plot([_tick(v) for v in dg.value], dg.gear_err, marker="o", label=g)
        ax.set_title(f"{ap}: {pa}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].set_ylabel("gear error (T1)")
    axes[1, 0].set_ylabel("gear error (T1)")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Hyperparameter sensitivity (one line per held-out gear)")
    fig.tight_layout()
    save_fig(fig, "fig_sensitivity.png")
    plt.close(fig)

    ra = t[(t.approach == "ridge_tooth") & (t.param == "alpha")].drop_duplicates("value")
    flat = ra[np.isclose(ra.t1_mse_pooled, ra.t1_mse_pooled.iloc[0])].value
    seeds = t[(t.approach == "iforest") & (t.param == "seed")].drop_duplicates("value").t1_mse_pooled
    record(ridge_alpha_flat_range=f"{flat.min():g} to {flat.max():g}", iforest_seed_sd_t1=float(seeds.std()),
           tuned_distinct_winners=int(win[["tooth", "form", "base_h", "tail", "monotone"]].drop_duplicates().shape[0]))
    return t


if __name__ == "__main__":
    main()
