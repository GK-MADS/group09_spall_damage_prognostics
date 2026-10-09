"""Step 1.3: performance against the amount of training data (gears and healthy hours, not rows).

Usage (repo root):  .venv/bin/python -m models.eval.curves
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd
from spall_hi import challenge_score, photo_damage
from spall_supervised import LABELLED

from .adapters import HI, IForest, RidgeApproach
from .harness import OUT, TRAIN, evaluate_runs, load_dataset, pooled_residuals, record, run_folds, save_fig

APPROACHES = {
    "hi_tooth": lambda: HI("tooth"),
    "hi_tooth_maha": lambda: HI("tooth_maha"),
    "hi_maha": lambda: HI("maha"),
    "iforest": lambda: IForest(),
}
# hi_tooth = clip(T, 0).mean() fits nothing but its threshold, so its gear curve is flat by construction;
# the quoted number comes from an approach that actually learns from the healthy gears.
HEADLINE = "iforest"


def gain_2_to_3(g: pd.DataFrame, approach: str) -> float:
    """Drop in mean gear error from 2 to 3 training gears (positive = still improving)."""
    m = g[g.approach == approach].groupby("k").err.mean()
    return float(m[2] - m[3])


def gear_curve(ds) -> pd.DataFrame:
    """Healthy model fit on every 1-, 2- and 3-gear subset of the non-held-out gears; one pooled map per k."""
    rows = []
    for name, make in APPROACHES.items():
        preds = []
        for held in TRAIN:
            others = [g for g in TRAIN if g != held]
            for k in (1, 2, 3):
                for combo in itertools.combinations(others, k):
                    a = make().fit(
                        ds.z[ds.z.experiment.isin(combo)], ds.ends[ds.ends.experiment.isin(combo)], ds.lab.iloc[:0]
                    )
                    r = a.score(ds.z[ds.z.experiment == held], ds.ends[ds.ends.experiment == held])
                    preds.append(r.assign(k=k, held=held, combo="+".join(c[-1] for c in combo)))
        p = pd.concat(preds, ignore_index=True)
        p["photo"] = photo_damage(p)
        for k, pk in p.groupby("k"):
            pk = pk.assign(sq=pooled_residuals(pk.score, pk.photo) ** 2)
            rows += [
                dict(approach=name, k=k, held=h, combo=c, err=e)
                for (h, c), e in pk.groupby(["held", "combo"]).sq.mean().items()
            ]
    return pd.DataFrame(rows)


def baseline_hours_curve(ds) -> pd.DataFrame:
    rows = []
    for base_h in [1.0, 2.0, 3.0, 4.5, 6.0]:
        d = load_dataset(minutes=ds.minutes, base_h=base_h)
        for name, make in APPROACHES.items():
            rows.append(
                dict(
                    approach=name, base_h=base_h, t1_mse_pooled=evaluate_runs(run_folds(make, d).runs)[0]["mse_pooled"]
                )
            )
    return pd.DataFrame(rows)


def ridge_curve(ds) -> pd.DataFrame:
    """Train on the first k inspections of one labelled gear, test on the other labelled gear."""
    rows = []
    for tr, te in [LABELLED, LABELLED[::-1]]:
        zte, ete = ds.z[ds.z.experiment == te], ds.ends[ds.ends.experiment == te]
        yte = ete.merge(ds.lab, on=["experiment", "run_num"])
        for k in sorted(ds.lab[ds.lab.experiment == tr].run_num):
            z = ds.z[(ds.z.experiment == tr) & (ds.z.run_num <= k)]
            if z.empty:
                continue  # no usable minutes in the first k runs
            e = ds.ends[(ds.ends.experiment == tr) & (ds.ends.run_num <= k)]
            lab = ds.lab[(ds.lab.experiment == tr) & (ds.lab.run_num <= k)]
            r = RidgeApproach().fit(z, e, lab).score(zte, ete).merge(yte[["run_num", "mean_ex"]], on="run_num")
            s = challenge_score(r.score, r.mean_ex)
            rows.append(dict(train_gear=tr, test_gear=te, k=k, t2_mse=s["mse_pooled"], spearman=s["spearman"]))
    return pd.DataFrame(rows)


def main(ds=None):
    ds = ds or load_dataset()
    g, b, r = gear_curve(ds), baseline_hours_curve(ds), ridge_curve(ds)
    g.round(6).to_csv(OUT / "curves_gears.csv", index=False)
    b.round(6).to_csv(OUT / "curves_baseline_hours.csv", index=False)
    r.round(6).to_csv(OUT / "curves_ridge.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for name, d in g.groupby("approach"):
        m = d.groupby("k").err.agg(["mean", "min", "max"])
        axes[0].errorbar(
            m.index, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]], marker="o", capsize=3, label=name
        )
    axes[0].set(
        xlabel="training gears for the healthy model", ylabel="gear error (T1)", xticks=[1, 2, 3], title="Gears"
    )
    for name, d in b.groupby("approach"):
        axes[1].plot(d.base_h, d.t1_mse_pooled, marker="o", label=name)
    axes[1].set(xlabel="healthy-baseline hours", ylabel="T1 pooled MSE", title="Healthy hours")
    for (tr, te), d in r.groupby(["train_gear", "test_gear"]):
        axes[2].plot(d.k, d.t2_mse, marker="o", label=f"train {tr[-1]} → test {te[-1]}")
    axes[2].set(xlabel="labelled inspections used", ylabel="T2 MSE on other gear", title="Ridge labels")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Learning curves (bars = min–max over gear combinations)")
    fig.tight_layout()
    save_fig(fig, "fig_learning_curves.png")
    plt.close(fig)

    record(
        curve_headline=HEADLINE,
        curve_gain_2_to_3_gears=gain_2_to_3(g, HEADLINE),
        curve_gain_2_to_3_gears_hi_tooth_maha=gain_2_to_3(g, "hi_tooth_maha"),
    )


if __name__ == "__main__":
    main()
