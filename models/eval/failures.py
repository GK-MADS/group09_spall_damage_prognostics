"""Step 1.4: rank failures, pick examples by a fixed rule, and explain them (exact, SHAP, LIME).

Usage (repo root):  .venv/bin/python -m models.eval.failures   (needs out/runs.csv from models.eval.compare)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from spall_hi import TAIL_MIN

from .adapters import HI, IForest, LabelTuned, RidgeApproach, registry
from .harness import OUT, SEED, folds, load_dataset, pooled_residuals, record, run_folds, save_fig

# Hypotheses from the design, each tied to a category; select_examples confirms or rejects them against the data.
HYPOTHESES = [
    dict(
        approach="hi_maha",
        experiment="Exp-A",
        target="photo",
        category="systematic",
        source="design",
        why="all-CI distance jumps when the rig restarts for run 3, driven by energy/global CIs, while the photo shows "
        "no damage",
    ),
    dict(
        approach="hi_tooth_maha",
        experiment="Exp-F",
        target="photo",
        category="systematic",
        source="revised after ranking",
        why="photo truth sums damage over teeth (6 spalled teeth on Exp-F's last inspection vs 2 on Exp-C) while the "
        "vibration index tracks the worst tooth, so the most-damaged gear is under-scored by every approach",
    ),
    dict(
        approach="ridge_all",
        experiment="Exp-B",
        target="mean_ex",
        category="systematic",
        source="design",
        why="trained on Exp-A's labels alone, 24 sign-split inputs fit Exp-A-specific CI shifts and under-estimate "
        "Exp-B's final wear (overfitting with one labelled gear)",
    ),
    dict(
        approach="tuned_nested",
        experiment="Exp-C",
        target="photo",
        category="edge case",
        source="design",
        why="Exp-C runs twice as long as any other gear and damages only at run 12 (two spalled teeth, two camera "
        "setups), so its final score is over-mapped relative to its photo label",
    ),
    dict(
        approach="ridge_tooth",
        experiment="Exp-A",
        target="mean_ex",
        category="edge case",
        source="design",
        why="trained on Exp-B's gradual wear and tested on Exp-A's late spalls: a damage regime it never saw",
    ),
    dict(
        approach="iforest",
        experiment="Exp-B",
        target="photo",
        category="systematic",
        source="revised after ranking",
        why="the Isolation Forest over-scores Exp-B's last run under every seed; only the size of the error depends "
        "on the seed (systematic with seed-dependent magnitude)",
    ),
    dict(
        approach="iforest",
        experiment="Exp-C",
        run_num=12,
        target="photo",
        category="random",
        source="revised after ranking",
        why="the Isolation Forest's error on Exp-C's spall run changes sign with the random seed: its mean across "
        "seeds is smaller than its spread, so no fix to the model's assumptions would remove it",
    ),
]


def candidates(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (name, kind), r in runs.groupby(["approach", "kind"]):
        for target in ["photo", "mean_ex"]:
            x = r[r[target].notna()].copy()
            if x.empty:
                continue
            res = pooled_residuals(x.score, x[target])
            x = x.assign(target=target, truth=x[target] / max(x[target].max(), 1e-9), residual=res)
            x["fitted"] = x.truth + x.residual
            x["rank"] = x.residual.abs().rank(ascending=False, method="first").astype(int)
            out.append(
                x[
                    [
                        "approach",
                        "kind",
                        "target",
                        "experiment",
                        "run_num",
                        "end_h",
                        "score",
                        "truth",
                        "fitted",
                        "residual",
                        "rank",
                    ]
                ]
            )
    return pd.concat(out, ignore_index=True)


def select_examples(cands: pd.DataFrame, max_rank: int = 5) -> pd.DataFrame:
    rows = []
    for h in HYPOTHESES:
        c = cands[
            (cands.approach == h["approach"]) & (cands.experiment == h["experiment"]) & (cands.target == h["target"])
        ]
        if "run_num" in h:  # pinned run end (random errors are not consistently the worst)
            c = c[c.run_num == h["run_num"]]
        top = c.sort_values("rank").iloc[0]
        rows.append(
            {
                **h,
                **top[["run_num", "end_h", "score", "truth", "fitted", "residual", "rank"]].to_dict(),
                "confirmed": bool(top["rank"] <= max_rank),
            }
        )
    return pd.DataFrame(rows)


def exact_attribution(a, x: np.ndarray) -> pd.Series | None:
    """Closed-form per-CI contributions for HI tooth (sums to the score) and maha / tooth_maha (sum to score**2)."""
    form = a.col if isinstance(a, HI) else a.st["form"] if isinstance(a, LabelTuned) else None
    if form == "tooth":
        idx = [a.features.index(c) for c in a.hm.tooth]
        return pd.Series(np.clip(x[idx], 0, None) / len(idx), index=a.hm.tooth)
    if form in ("maha", "tooth_maha"):
        cols = a.hm.feats if form == "maha" else a.hm.tooth
        lw = a.hm.lw if form == "maha" else a.hm.lw_t
        d = x[[a.features.index(c) for c in cols]] - lw.location_
        contrib = d * (lw.precision_ @ d) / len(cols)
        if form == "tooth_maha" and x[[a.features.index(c) for c in cols]].mean() <= 0:
            contrib = contrib * 0
        return pd.Series(contrib, index=cols)
    return None


def shap_attribution(a, x: np.ndarray) -> pd.Series | None:
    try:
        import shap
    except ImportError as e:
        raise ImportError("failures.py needs shap: uv pip install --python .venv/bin/python shap") from e
    if isinstance(a, RidgeApproach):
        ex = shap.LinearExplainer(a.m, a.background)
        sv = pd.Series(ex.shap_values(a.design(x[None]))[0], index=a.design_names)
        return sv.groupby(lambda n: n.rstrip("+-")).sum().reindex(a.features)
    if isinstance(a, IForest):
        sv = shap.TreeExplainer(a.m).shap_values(x[None])[0]
        # shap explains IsolationForest's path-length output (higher = more normal): flip to anomaly direction
        return pd.Series(-sv, index=a.features)
    return None


def lime_attribution(a, x: np.ndarray, background: np.ndarray) -> pd.Series:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise ImportError("failures.py needs lime: uv pip install --python .venv/bin/python lime") from e
    ex = LimeTabularExplainer(
        background,
        feature_names=a.features,
        mode="regression",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=SEED,
    )
    e = ex.explain_instance(x, a.predict_rows, num_features=len(a.features), num_samples=2000)
    # LIME weights are slopes per standardised unit; contribution = slope x the instance's standardised deviation,
    # which is the quantity SHAP and the exact decompositions report
    w = pd.Series(dict(e.as_list())).reindex(a.features).fillna(0.0)
    return w * (x - ex.scaler.mean_) / ex.scaler.scale_


def agreement(ref: pd.Series | None, lime: pd.Series) -> float:
    """Spearman correlation of two attributions over the features both cover (NaN if fewer than 2)."""
    if ref is None:
        return np.nan
    common = ref.dropna().index.intersection(lime.index)
    return float(spearmanr(ref[common], lime[common])[0]) if len(common) >= 2 else np.nan


def _instance(a, ds, gear, run_num):
    """Median feature vector over the run's last TAIL_MIN steady minutes (the run-end state the score summarises)."""
    z = ds.z if not isinstance(a, LabelTuned) else _tuned_z(a, ds)
    r = z[(z.experiment == gear) & (z.run_num == run_num)].sort_values("hours").tail(TAIL_MIN)
    return r[a.features].median().to_numpy(), z


def _tuned_z(a, ds):
    from spall_hi import condition_correct

    return condition_correct(ds.minutes, a.st["base_h"])


def is_random(vals) -> bool:
    """Random error: the residual's mean across seeds is smaller than its spread (its sign is not stable)."""
    v = np.asarray(vals, float)
    return bool(abs(v.mean()) < v.std())


def seed_residual_sd(ds, make_seeded, gear: str, run_num: int, seeds=range(5), target: str = "photo") -> float:
    """SD across random seeds of one run end's pooled-map residual (0 for a deterministic approach)."""
    return float(np.std(seed_residuals(ds, make_seeded, gear, run_num, seeds, target)))


def seed_residuals(ds, make_seeded, gear: str, run_num: int, seeds=range(5), target: str = "photo") -> np.ndarray:
    vals = []
    for seed in seeds:
        r = run_folds(lambda seed=seed: make_seeded(seed), ds).runs
        r = r[r[target].notna()]
        res = pooled_residuals(r.score, r[target])
        vals.append(res[((r.experiment == gear) & (r.run_num == run_num)).to_numpy()][0])
    return np.array(vals)


def explain(ex_row, ds, n):
    reg = registry(ds)
    train, held = next((t, h) for t, h in folds() if h == ex_row.experiment)
    a = reg[ex_row.approach]().fit(
        ds.z[ds.z.experiment.isin(train)],
        ds.ends[ds.ends.experiment.isin(train)],
        ds.lab[ds.lab.experiment.isin(train)],
    )
    x, z = _instance(a, ds, held, int(ex_row.run_num))
    if np.isnan(x).any():
        return dict(explained=False, note="no minutes in this run (carried forward from the previous run)")
    background = z.loc[z.experiment.isin(train), a.features].to_numpy()
    ref = exact_attribution(a, x)
    ref = shap_attribution(a, x) if ref is None else ref
    lime = lime_attribution(a, x, background)
    agree = agreement(ref, lime)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 2]))
    att = pd.DataFrame({"exact/SHAP": ref, "LIME": lime}) if ref is not None else pd.DataFrame({"LIME": lime})
    att = att.reindex(a.features)
    (att / att.abs().sum()).plot.barh(ax=ax1)  # methods report different units; compare shares
    ax1.set_xlabel("share of total |attribution|")
    ax1.set_title(f"Attribution, {ex_row.approach} {held} run {int(ex_row.run_num)}", fontsize=9)
    zh = z[z.experiment == held].sort_values("hours")
    sc = pd.Series(a.predict_rows(zh[a.features].to_numpy()), index=zh.hours.to_numpy())
    sc[zh.hours.diff().gt(0.5).to_numpy()] = np.nan  # break the line across gaps (e.g. nulled Exp-A Run-2)
    ax2.plot(sc.index, sc.to_numpy(), lw=0.6, label="minute score")
    ax2.axhline(a.threshold, color="r", ls=":", label="alarm threshold")
    for h in ds.ends.loc[ds.ends.experiment == held, "end_h"]:
        ax2.axvline(h, color="grey", lw=0.5)
    mh = ds.minutes[ds.minutes.experiment == held].sort_values("hours").copy()
    mh.loc[mh.hours.diff().gt(0.5), ["ctx_temperature", "ctx_pau_torque"]] = np.nan
    tw = ax2.twinx()
    tw.plot(mh.hours, mh.ctx_temperature, color="tab:orange", lw=0.4, alpha=0.6, label="temperature")
    tw.plot(mh.hours, mh.ctx_pau_torque, color="tab:green", lw=0.4, alpha=0.6, label="torque")
    tw.set_ylabel("temperature / torque")
    tw.legend(fontsize=7, loc="upper right")
    ax2.set(xlabel="hours (grey = run ends / restarts)", ylabel="score", title=f"{held} over its life")
    ax2.legend(fontsize=7, loc="upper left")
    save_fig(fig, f"fig_failure_{n}.png")
    plt.close(fig)
    top = lambda s: ", ".join(s.abs().sort_values(ascending=False).index[:3]) if s is not None else ""  # noqa: E731
    return dict(explained=True, top_exact_or_shap=top(ref), top_lime=top(lime), lime_agreement=agree)


def main(ds=None):
    ds = ds or load_dataset()
    runs = pd.read_csv(OUT / "runs.csv")
    cands = candidates(runs)
    cands.round(5).to_csv(OUT / "failure_candidates.csv", index=False)
    ex = select_examples(cands)
    extra = [explain(row, ds, i + 1) for i, row in enumerate(ex.itertuples())]
    ex = pd.concat([ex, pd.DataFrame(extra)], axis=1)
    seeds = [
        seed_residuals(ds, lambda s: IForest(seed=s), e.experiment, int(e.run_num), range(SEED, SEED + 10), e.target)
        if e.approach == "iforest"
        else np.array([e.residual])  # deterministic: every seed gives the same residual
        for e in ex.itertuples()
    ]
    ex["seed_mean"] = [v.mean() for v in seeds]
    ex["seed_residual_sd"] = [v.std() for v in seeds]
    ex["seed_sign_flips"] = [int(min((v > 0).sum(), (v < 0).sum())) for v in seeds]
    ex["random_like"] = [is_random(v) for v in seeds]
    # A random example must have an unstable sign and an error at least as large as the approach's 5th-largest;
    # every other category must be stable in sign and rank in the approach's top 5.
    fifth = cands.groupby(["approach", "target"]).residual.apply(lambda r: r.abs().nlargest(5).min())
    big = [np.abs(v).max() >= fifth[(e.approach, e.target)] for v, e in zip(seeds, ex.itertuples())]
    ex["confirmed"] = [
        (rl and b) if c == "random" else (conf and not rl)
        for rl, b, c, conf in zip(ex.random_like, big, ex.category, ex.confirmed)
    ]
    ex.round(5).to_csv(OUT / "failure_examples.csv", index=False)
    conf = ex[ex.confirmed]
    record(
        n_failure_examples=int(len(conf)),
        n_failure_categories=int(conf.category.nunique()),
        lime_agreement_median=float(ex.get("lime_agreement", pd.Series(dtype=float)).median()),
    )
    print(
        ex[
            [
                "approach",
                "experiment",
                "run_num",
                "category",
                "rank",
                "confirmed",
                "top_exact_or_shap",
                "top_lime",
                "lime_agreement",
            ]
        ].to_string()
    )
    return ex


if __name__ == "__main__":
    main()
