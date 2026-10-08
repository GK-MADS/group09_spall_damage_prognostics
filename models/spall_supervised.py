"""Label-driven damage model using the per-tooth spall scores for Exp-A and Exp-B.

Labels (data/labels/spall_scores.csv): one score per tooth per run end, 1.0 = healthy, ~3-4 = spall.
Run-level targets: mean_ex = mean(score - 1) over the 28 teeth (primary: 0 on healthy runs, rises with
both Exp-B's gradual wear and Exp-A's late spalls), max, and top3 = mean excess of the 3 worst teeth.

There are 11 labelled run ends on 2 gears (Exp-A Run-2 has no sensor data), so the script does two things:

1. Supervised comparison. Non-negative ridge on sign-split, log-compressed per-gear CI deviations, trained
   on every steady minute with labels interpolated between inspections. Checked A<->B leave-one-gear-out.
   On all CIs it learns Exp-B's restart shifts as "wear"; on tooth CIs it only ties the unsupervised HI.
2. Label-tuned unsupervised HI (submitted). The labels choose the settings of spall_hi's healthy-state
   score (tooth CIs, score form, baseline window, run-end window, monotone step). Each setting is scored
   leave-one-gear-out on the A/B labels and on Exp-C/Exp-F against the by-eye photo proxy; the setting
   with the best rank sum is refit on all training baselines and scored on validation and test.

Usage
  python models/spall_supervised.py            # tables + models/out/labelled_*.csv
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from spall_hi import (
    FEATS,
    SPLITS,
    TOOTH,
    HealthyModel,
    challenge_score,
    condition_correct,
    load_minutes,
    photo_damage,
    run_end_scores,
    run_ends,
)

LABELLED = ["Exp-A", "Exp-B"]
AGGS = {
    "mean_ex": lambda s: (s - 1).mean(),
    "max": lambda s: s.max(),
    "top3": lambda s: (s - 1).nlargest(3).mean(),
}
GRID = dict(
    tooth=[["ci_fm4"], ["ci_fm4", "ci_na4"], ["ci_fm4", "ci_m6a", "ci_m8a"], TOOTH],
    form=["tooth", "tooth_maha"],
    base_h=[3.0, 6.0],
    tail=[30, 60, 120],
    monotone=["iso", "cummax"],
)


# ---------------------------------------------------------------- labels
def load_labels(path: Path) -> pd.DataFrame:
    """Run-level targets, one column per aggregation of the 28 tooth scores."""
    d = pd.read_csv(path)
    return d.groupby(["experiment", "run_num"]).score.agg(**AGGS).reset_index()


def minute_targets(z: pd.DataFrame, ends: pd.DataFrame, lab: pd.DataFrame, target: str) -> pd.Series:
    """Label at each minute: linear in hours between the previous inspection and this run's end."""
    r = ends.merge(lab, on=["experiment", "run_num"]).sort_values(["experiment", "run_num"])
    out = pd.Series(np.nan, index=z.index)
    for e, g in r.groupby("experiment"):
        h = np.r_[0.0, g.end_h.to_numpy()]
        y = np.r_[g[target].iloc[0], g[target].to_numpy()]  # before run 1 = the run-1 inspection
        k = z.experiment == e
        out[k] = np.interp(z.loc[k, "hours"], h, y)
    return out


def scores(pred: pd.Series, r: pd.DataFrame) -> dict:
    return {k: challenge_score(pred, r[k], r.experiment.to_numpy())["mse_pooled"] for k in AGGS}


# ---------------------------------------------------------------- 1. supervised comparison
class RidgeDamage:
    def __init__(self, feats: list[str], alpha: float = 1.0, target: str = "mean_ex"):
        self.feats, self.alpha, self.target = feats, alpha, target

    def design(self, z):
        Z = z[self.feats].to_numpy()
        return np.log1p(np.c_[np.clip(Z, 0, None), np.clip(-Z, 0, None)])

    def fit(self, z, ends, lab):
        y = minute_targets(z, ends, lab, self.target)
        k = y.notna().to_numpy()
        self.m = Ridge(alpha=self.alpha, positive=True).fit(self.design(z)[k], y[k])
        return self

    def score(self, z):
        return z[["experiment", "run_num", "hours"]].assign(hi=self.m.predict(self.design(z)))


def supervised_table(z, ends, lab) -> pd.DataFrame:
    zab, eab = z[z.experiment.isin(LABELLED)], ends[ends.experiment.isin(LABELLED)]
    rows = []
    for name, feats in [("ridge all CIs", FEATS), ("ridge tooth CIs", TOOTH)]:
        for alpha in [0.1, 10, 1000]:
            s = pd.concat(
                RidgeDamage(feats, alpha).fit(zab[zab.experiment != h], eab, lab).score(zab[zab.experiment == h])
                for h in LABELLED
            )
            r = run_end_scores(s, eab, ["hi"]).merge(lab)
            m = RidgeDamage(feats, alpha).fit(zab, eab, lab)
            rows.append(
                dict(model=f"{name}, alpha={alpha}", **scores(r.hi, r), CF_photo=cf_score(m.score(z), ends, "hi"))
            )
    return pd.DataFrame(rows).set_index("model")


def cf_score(s: pd.DataFrame, ends: pd.DataFrame, col: str, **kw) -> float:
    """Unlabelled training gears (Exp-C, Exp-F) against the by-eye photo proxy."""
    r = run_end_scores(s[~s.experiment.isin(LABELLED)], ends[~ends.experiment.isin(LABELLED)], [col], **kw)
    return challenge_score(r[col], photo_damage(r))["mse_pooled"]


def write_submissions(score_split, col: str, data: Path, path_fmt: str) -> pd.DataFrame:
    """Template times are run ends; each gets the HI of the nearest run end of that gear."""
    subs = []
    for split, tmpl in [("validation", "example_validation_submission.csv"), ("test", "example_submission.csv")]:
        r = score_split(split)
        r["run_id"] = r.experiment.str[-1]
        sub = pd.read_csv(data / split / tmpl)
        idx = [
            r[r.run_id == g].pipe(lambda x: x.index[(x.end_h - t).abs().argmin()]) for g, t in zip(sub.run_id, sub.time)
        ]
        sub["metric"] = r.loc[idx, col].round(4).to_numpy()
        sub.to_csv(path_fmt.format(split=split), index=False)
        subs.append(sub)
    return pd.concat(subs)


# ---------------------------------------------------------------- 2. label-tuned unsupervised HI
def logo_hi(z, tooth) -> pd.DataFrame:
    """Healthy-state scores for every training gear, each from models fit without that gear."""
    tr = SPLITS["train"]
    return pd.concat(
        HealthyModel(tooth).fit(z[z.experiment.isin(tr) & (z.experiment != h)]).score(z[z.experiment == h]) for h in tr
    )


def tune(minutes, ends, lab) -> pd.DataFrame:
    rows = []
    for base_h in GRID["base_h"]:
        z = condition_correct(minutes, base_h)
        for tooth in GRID["tooth"]:
            s = logo_hi(z, tooth)
            for form, tail, mono in itertools.product(GRID["form"], GRID["tail"], GRID["monotone"]):
                r = run_end_scores(s, ends, [form], mono, tail)
                ab = r.merge(lab)
                cf = r[~r.experiment.isin(LABELLED)].assign(photo=lambda x: photo_damage(x))
                rows.append(
                    dict(
                        tooth="/".join(c[3:] for c in tooth),
                        form=form,
                        base_h=base_h,
                        tail=tail,
                        monotone=mono,
                        **scores(ab[form], ab),
                        CF_photo=challenge_score(cf[form], cf.photo)["mse_pooled"],
                    )
                )
    t = pd.DataFrame(rows)
    t["AB"] = t[list(AGGS)].mean(1)
    t["rank"] = t.AB.rank() + t.CF_photo.rank()
    return t.sort_values(["rank", "AB"], kind="stable")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels/spall_scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("models/out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    lab = load_labels(a.labels)
    minutes = {s: load_minutes(a.data, s) for s in SPLITS}
    ends = {s: run_ends(a.data, s) for s in SPLITS}
    ztr = condition_correct(minutes["train"])

    sup = supervised_table(ztr, ends["train"], lab)
    default = run_end_scores(logo_hi(ztr, TOOTH), ends["train"], ["tooth_maha"]).merge(lab)
    sup.loc["unsupervised tooth_maha (previous default)"] = {
        **scores(default.tooth_maha, default),
        "CF_photo": cf_score(logo_hi(ztr, TOOTH), ends["train"], "tooth_maha"),
    }
    sup.loc["elapsed hours"] = {
        **scores(default.end_h, default),
        "CF_photo": cf_score(ztr.assign(h=ztr.hours), ends["train"], "h"),
    }
    print("A<->B leave-one-gear-out vs labels, pooled MSE after best monotone map (lower is better).")
    print("CF_photo: trained on A+B, scored on Exp-C/Exp-F against the by-eye photo proxy.")
    print(sup.round(4).to_string())

    t = tune(minutes["train"], ends["train"], lab)
    print("\nLabel-tuned unsupervised HI, top 10 of", len(t), "settings (AB = mean over targets):")
    print(t.head(10).round(4).to_string(index=False))
    best = t.iloc[0].to_dict()
    tooth = ["ci_" + c for c in best["tooth"].split("/")]
    print("\nselected:", {k: best[k] for k in ["tooth", "form", "base_h", "tail", "monotone"]})

    # final: healthy models on all four training gears; label-unit calibration for reading only
    z = {s: condition_correct(minutes[s], best["base_h"]) for s in SPLITS}
    hm = HealthyModel(tooth).fit(z["train"])

    def run(s):
        return run_end_scores(hm.score(z[s]), ends[s], [best["form"]], best["monotone"], int(best["tail"]))

    rtr = run("train").merge(lab, how="left")
    k = rtr.mean_ex.notna()
    cal = IsotonicRegression(out_of_bounds="clip").fit(rtr.loc[k, best["form"]], rtr.loc[k, "mean_ex"])
    detail = write_submissions(run, best["form"], a.data, str(a.out / "labelled_{split}_submission.csv"))
    detail["est_mean_excess"] = cal.predict(detail.metric).round(3)

    # alternate: tooth-CI ridge fit on the A/B labels (best on A<->B, worse on C/F; see table above)
    zd = {s: condition_correct(minutes[s]) for s in SPLITS}
    rm = RidgeDamage(TOOTH, alpha=10).fit(zd["train"][zd["train"].experiment.isin(LABELLED)], ends["train"], lab)
    alt = write_submissions(
        lambda s: run_end_scores(rm.score(zd[s]), ends[s], ["hi"]),
        "hi",
        a.data,
        str(a.out / "labelled_ridge_{split}_submission.csv"),
    )
    detail["ridge_alt"] = alt.metric.to_numpy()
    print(
        "\nsubmissions (metric = submitted HI; est_mean_excess = isotonic map onto the A/B label scale, for reading):"
    )
    print(detail.to_string(index=False))

    detail.to_csv(a.out / "labelled_predictions_detail.csv", index=False)
    sup.round(5).to_csv(a.out / "labelled_supervised_eval.csv")
    t.round(5).to_csv(a.out / "labelled_tuning.csv", index=False)
    json.dump(
        {k: (v if isinstance(v, str) else float(v)) for k, v in best.items()},
        open(a.out / "labelled_config.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
