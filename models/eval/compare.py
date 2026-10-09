"""Step 2: every approach through the same folds, metrics, detection and gear-level tests.

Usage (repo root):  .venv/bin/python -m models.eval.compare
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adapters import registry
from .harness import (
    OUT,
    bootstrap_range,
    detection,
    evaluate_runs,
    lead_times,
    load_dataset,
    p_floor,
    pooled_residuals,
    record,
    run_folds,
    save_fig,
    sign_flip_p,
)


def rank_approaches(table: pd.DataFrame) -> list[str]:
    """Winner rule fixed in the spec: most gears beating hours, then lowest T1 mse_pooled, then fewest tuned settings.

    In-sample rows (settings chosen on the evaluation labels) are reported but cannot win.
    """
    in_sample = table["in_sample"].astype(bool) if "in_sample" in table else pd.Series(False, index=table.index)
    t = table[(table.kind != "baseline") & ~in_sample]
    return t.sort_values(["k_beats_hours", "t1_mse_pooled", "n_tuned"], ascending=[False, True, True]).index.tolist()


def label_noise_mse(runs: pd.DataFrame) -> float:
    """T1 mse_pooled on the final two run ends per gear only (the inspections that were checked by eye)."""
    last2 = runs.groupby("experiment").run_num.transform(lambda r: r >= r.max() - 1)
    return evaluate_runs(runs[last2])[0]["mse_pooled"]


def main(ds=None) -> pd.DataFrame:
    ds = ds or load_dataset()
    results = {name: run_folds(make, ds) for name, make in registry(ds).items()}
    rows, gear_rows, lead_rows, run_rows = [], [], [], []
    hours_err = evaluate_runs(results["hours"].runs)[1]
    for name, fr in results.items():
        a = next(iter(fr.models.values()))
        s1, g1 = evaluate_runs(fr.runs, "photo")
        s2, _ = evaluate_runs(fr.runs, "mean_ex")
        diffs = (hours_err - g1).reindex(g1.index)
        lo, hi = bootstrap_range(g1.to_numpy())
        lt = lead_times(fr.runs)
        rows.append(dict(
            approach=name, kind=a.kind, n_tuned=a.n_tuned, in_sample=getattr(a, "in_sample", False),
            t1_mse_pooled=s1["mse_pooled"], t1_mse_per_gear=s1["mse_per_gear"], t1_spearman=s1["spearman"],
            t1_gear_err_mean=g1.mean(), t1_gear_err_lo=lo, t1_gear_err_hi=hi,
            t2_mse_pooled=s2["mse_pooled"], t2_spearman=s2["spearman"],
            **detection(fr.runs), lead_time_median_h=lt.median(),
            k_beats_hours=int((diffs > 0).sum()), p_vs_hours=sign_flip_p(diffs), p_floor=p_floor(len(diffs)),
            t1_mse_checked_only=label_noise_mse(fr.runs), fit_s=fr.fit_s, score_s=fr.score_s,
        ))
        gear_rows += [dict(approach=name, experiment=g, gear_err=v) for g, v in g1.items()]
        lead_rows += [dict(approach=name, experiment=g, lead_h=v) for g, v in lt.items()]
        run_rows.append(fr.runs.assign(approach=name, kind=a.kind))
    table = pd.DataFrame(rows).set_index("approach")
    table["rank_checked_only"] = table.t1_mse_checked_only.rank()
    table["rank_all"] = table.t1_mse_pooled.rank()
    order = rank_approaches(table)

    OUT.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(OUT / "comparison.csv")
    gears = pd.DataFrame(gear_rows)
    gears.to_csv(OUT / "gear_errors.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(OUT / "lead_time.csv", index=False)
    runs = pd.concat(run_rows, ignore_index=True)
    runs.to_csv(OUT / "runs.csv", index=False)

    _fig_dots(gears, order + ["hours", "debris"])
    _fig_trajectories(runs, order[:3])
    _fig_lead(pd.DataFrame(lead_rows), order + ["hours", "debris"])
    w = table.loc[order[0]]
    record(
        winner=order[0], winner_k_of_4=int(w.k_beats_hours), winner_mse_pooled_t1=float(w.t1_mse_pooled),
        winner_p_vs_hours=float(w.p_vs_hours), p_floor_4=p_floor(4), p_floor_2=p_floor(2),
        selection_bias_gap_t1=float(table.loc["tuned_nested", "t1_mse_pooled"] - table.loc["tuned_nonnested", "t1_mse_pooled"]),
        ranking_changes_on_checked_only=bool((table.rank_checked_only != table.rank_all).any()),
    )
    print(table.round(4).to_string())
    print("\nwinner order:", order)
    return table


def _fig_dots(gears, order):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (g, d) in enumerate(gears.groupby("experiment")):
        d = d.set_index("approach").reindex(order)
        ax.scatter(np.arange(len(order)) + (i - 1.5) * 0.12, d.gear_err, label=g, s=22)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_ylabel("gear error (T1, pooled map)")
    ax.legend(title="held-out gear", ncol=4, fontsize=8)
    ax.set_title("Per-gear error by approach (lower is better)")
    save_fig(fig, "fig_per_gear_dots.png")
    plt.close(fig)


def _fig_trajectories(runs, top):
    gears = sorted(runs.experiment.unique())
    fig, axes = plt.subplots(1, len(gears), figsize=(14, 3.2), sharey=True)
    for ax, g in zip(axes, gears):
        for i, name in enumerate(top):
            r = runs[runs.approach == name]
            fitted = pooled_residuals(r.score, r.photo) + r.photo / max(r.photo.max(), 1e-9)
            k = (r.experiment == g).to_numpy()
            ax.step(r.end_h[k], fitted[k], where="post", label=name, ls=["-", "--", ":"][i % 3], lw=2.2 - 0.5 * i)
        t = runs[(runs.approach == top[0]) & (runs.experiment == g)]
        ax.step(t.end_h, t.photo / runs.photo.max(), where="post", color="k", lw=1, label="photo truth")
        ax.set_title(g)
        ax.set_xlabel("hours")
    axes[0].set_ylabel("damage (scaled)")
    axes[-1].legend(fontsize=7)
    save_fig(fig, "fig_trajectories.png")
    plt.close(fig)


def _fig_lead(lead, order):
    p = lead.pivot(index="approach", columns="experiment", values="lead_h").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    p.plot.bar(ax=ax, width=0.8)
    w = 0.8 / p.shape[1]
    for j, g in enumerate(p.columns):
        x = np.arange(len(p)) - 0.4 + w * (j + 0.5)
        v = p[g].to_numpy()
        ax.scatter(x[v == 0], v[v == 0], marker="o", s=14, color=f"C{j}", zorder=3)
        ax.scatter(x[np.isnan(v)], np.zeros(np.isnan(v).sum()), marker="x", s=24, color=f"C{j}", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lead time (h, + = early)")
    ax.set_title("Hours between first alarm and first spall photo (dot = alarm at the spall inspection, x = never)")
    save_fig(fig, "fig_lead_time.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
