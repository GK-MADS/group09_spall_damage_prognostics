"""One adapter per approach, all with fit(z, ends, lab) / score(z, ends) -> run-end table.

`z`, `ends`, `lab` passed to fit only ever contain the fold's training gears (enforced by harness.run_folds).
Minute-scored adapters set `threshold` to the 95th percentile of their minute scores in the training healthy window.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from spall_hi import (
    FEATS,
    TAIL_MIN,
    TOOTH,
    HealthyModel,
    challenge_score,
    condition_correct,
    photo_damage,
    run_end_scores,
)
from spall_supervised import AGGS, GRID, LABELLED, minute_targets

from .harness import SEED, Dataset

TUNED_CONFIG = Path(__file__).resolve().parents[1] / "out" / "labelled_config.json"


def _run_end(s: pd.DataFrame, ends: pd.DataFrame, col: str, monotone: str, tail: int) -> pd.DataFrame:
    r = run_end_scores(s, ends, [col], monotone, tail)
    return r[["experiment", "run_num", "end_h"]].assign(score=r[col].to_numpy())


def _rows(X: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=cols).assign(experiment="x", run_num=0, hours=0.0)


class HI:
    kind, n_tuned = "unsupervised", 0

    def __init__(self, col, tooth=TOOTH, feats=FEATS, tail=TAIL_MIN, monotone="iso"):
        self.col, self.tooth, self.feats, self.tail, self.monotone = col, tooth, feats, tail, monotone
        self.name = f"hi_{col}"
        self.features = list(dict.fromkeys(feats + tooth))

    def fit(self, z, ends, lab):
        self.hm = HealthyModel(self.tooth, self.feats).fit(z)
        self.threshold = float(np.percentile(self.hm.score(z[z.base])[self.col], 95))
        return self

    def predict_rows(self, X):
        return self.hm.score(_rows(X, self.features))[self.col].to_numpy()

    def score(self, z, ends):
        return _run_end(self.hm.score(z), ends, self.col, self.monotone, self.tail)


class IForest:
    kind, n_tuned, name = "unsupervised", 0, "iforest"

    def __init__(self, feats=FEATS, tail=TAIL_MIN, monotone="iso", seed=SEED, **params):
        self.features, self.tail, self.monotone, self.seed = list(feats), tail, monotone, seed
        self.params = dict(n_estimators=200, max_samples="auto", max_features=1.0) | params

    def fit(self, z, ends, lab):
        B = z.loc[z.base, self.features].to_numpy()
        self.m = IsolationForest(random_state=self.seed, **self.params).fit(B)
        self.threshold = float(np.percentile(-self.m.score_samples(B), 95))
        return self

    def predict_rows(self, X):
        return -self.m.score_samples(np.asarray(X, float))

    def score(self, z, ends):
        s = z[["experiment", "run_num", "hours"]].assign(score=self.predict_rows(z[self.features].to_numpy()))
        return _run_end(s, ends, "score", self.monotone, self.tail)


class RidgeApproach:
    """Non-negative ridge on sign-split, log-compressed CI deviations (RidgeDamage), with ablation switches."""

    kind, n_tuned = "supervised", 1

    def __init__(
        self, feats=TOOTH, alpha=10.0, target="mean_ex", sign_split=True, log=True, positive=True, tail=TAIL_MIN
    ):
        self.features, self.alpha, self.target = list(feats), alpha, target
        self.sign_split, self.log, self.positive, self.tail = sign_split, log, positive, tail
        self.name = "ridge_tooth" if feats == TOOTH else "ridge_all"
        self.design_names = [f"{c}+" for c in feats] + [f"{c}-" for c in feats] if sign_split else list(feats)

    def design(self, Z):
        X = np.c_[np.clip(Z, 0, None), np.clip(-Z, 0, None)] if self.sign_split else np.asarray(Z, float)
        return np.sign(X) * np.log1p(np.abs(X)) if self.log else X

    def fit(self, z, ends, lab):
        zl = z[z.experiment.isin(LABELLED) & z.experiment.isin(lab.experiment.unique())]
        if zl.empty:
            raise ValueError("no labelled gear in this fold's training data (need Exp-A or Exp-B)")
        y = minute_targets(zl, ends, lab, self.target)
        k = y.notna().to_numpy()
        self.background = self.design(zl[self.features].to_numpy())[k]
        self.m = Ridge(alpha=self.alpha, positive=self.positive).fit(self.background, y[k])
        self.threshold = float(np.percentile(self.predict_rows(zl.loc[zl.base, self.features].to_numpy()), 95))
        return self

    def predict_rows(self, X):
        return self.m.predict(self.design(np.asarray(X, float)))

    def score(self, z, ends):
        s = z[["experiment", "run_num", "hours"]].assign(score=self.predict_rows(z[self.features].to_numpy()))
        return _run_end(s, ends, "score", "iso", self.tail)


class LabelTuned:
    """Healthy-state tooth HI with settings chosen on the labels.

    nested=False: the setting chosen by spall_supervised.tune on all labels (as submitted; in-sample).
    nested=True: the same 96-setting search re-run inside each fold on the fold's gears only.
    """

    kind, n_tuned = "semi-supervised", 96

    def __init__(self, minutes: pd.DataFrame, nested: bool = True, fixed: dict | None = None):
        self.minutes, self.nested = minutes, nested
        self.name = "tuned_nested" if nested else "tuned_nonnested"
        self.in_sample = not nested  # setting chosen on all labels, including the held-out gear's
        self.fixed = fixed or json.loads(TUNED_CONFIG.read_text())

    @staticmethod
    def _setting(d):
        return dict(
            tooth=["ci_" + c for c in d["tooth"].split("/")],
            form=d["form"],
            base_h=float(d["base_h"]),
            tail=int(d["tail"]),
            monotone=d["monotone"],
        )

    def _select(self, gears, ends, lab):
        rows = []
        for base_h in GRID["base_h"]:
            zc = condition_correct(self.minutes[self.minutes.experiment.isin(gears)], base_h)
            for tooth in GRID["tooth"]:
                s = pd.concat(
                    HealthyModel(tooth).fit(zc[zc.experiment != h]).score(zc[zc.experiment == h]) for h in gears
                )
                for form in GRID["form"]:
                    for tail in GRID["tail"]:
                        for mono in GRID["monotone"]:
                            r = run_end_scores(s, ends, [form], mono, tail)
                            ab = r.merge(lab, on=["experiment", "run_num"])
                            cf = r[~r.experiment.isin(LABELLED)]
                            rows.append(
                                dict(
                                    tooth="/".join(c[3:] for c in tooth),
                                    form=form,
                                    base_h=base_h,
                                    tail=tail,
                                    monotone=mono,
                                    AB=np.mean(
                                        [
                                            challenge_score(ab[form], ab[k], ab.experiment.to_numpy())["mse_pooled"]
                                            for k in AGGS
                                        ]
                                    ),
                                    CF=challenge_score(cf[form], photo_damage(cf))["mse_pooled"],
                                )
                            )
        t = pd.DataFrame(rows)
        t["rank"] = t.AB.rank() + t.CF.rank()
        return t.sort_values(["rank", "AB"], kind="stable").iloc[0].to_dict()

    def fit(self, z, ends, lab):
        gears = sorted(ends.experiment.unique())
        self.selected = self._select(gears, ends, lab) if self.nested else dict(self.fixed)
        st = self._setting(self.selected)
        self.features, self.st = st["tooth"], st
        zc = condition_correct(self.minutes[self.minutes.experiment.isin(gears)], st["base_h"])
        self.hm = HealthyModel(st["tooth"]).fit(zc)
        self.threshold = float(np.percentile(self.hm.score(zc[zc.base])[st["form"]], 95))
        return self

    def predict_rows(self, X):
        # HealthyModel.score needs every FEATS column; pad the non-tooth ones with 0 (tooth forms ignore them)
        df = _rows(np.zeros((len(X), len(FEATS))), FEATS)
        df[self.features] = np.asarray(X, float)
        return self.hm.score(df)[self.st["form"]].to_numpy()

    def score(self, z, ends):
        held = self.minutes[self.minutes.experiment.isin(ends.experiment.unique())]
        zc = condition_correct(held, self.st["base_h"])
        return _run_end(self.hm.score(zc), ends, self.st["form"], self.st["monotone"], self.st["tail"])


class Hours:
    kind, n_tuned, name = "baseline", 0, "hours"

    def fit(self, z, ends, lab):
        self.threshold = float(
            ends.loc[ends.run_num == ends.groupby("experiment").run_num.transform("min"), "end_h"].max()
        )
        return self

    def score(self, z, ends):
        return ends[["experiment", "run_num", "end_h"]].assign(score=ends.end_h.to_numpy())


class Debris:
    """Large-particle rate relative to the gear's first run, carried forward and made non-decreasing."""

    kind, n_tuned, name = "baseline", 0, "debris"

    def __init__(self, debris: pd.DataFrame):
        self.debris = debris

    def _rel(self, ends):
        r = ends.merge(self.debris, on=["experiment", "run_num"], how="left").sort_values(["experiment", "run_num"])
        rel = r.debris_rate / r.groupby("experiment").debris_rate.transform("first")
        r["score"] = rel.groupby(r.experiment).ffill().fillna(1.0).groupby(r.experiment).cummax()
        return r

    def fit(self, z, ends, lab):
        r = self._rel(ends)
        first = r.run_num == r.groupby("experiment").run_num.transform("min")
        self.threshold = float(r.loc[first, "score"].max())
        return self

    def score(self, z, ends):
        return self._rel(ends)[["experiment", "run_num", "end_h", "score"]]


def registry(ds: Dataset) -> dict:
    reg = {f"hi_{c}": (lambda c=c: HI(c)) for c in ["maha", "spe", "zpos", "zmax", "tooth", "tooth_maha"]}
    reg |= {
        "iforest": lambda: IForest(),
        "ridge_tooth": lambda: RidgeApproach(TOOTH),
        "ridge_all": lambda: RidgeApproach(FEATS),
        "tuned_nonnested": lambda: LabelTuned(ds.minutes, nested=False),
        "tuned_nested": lambda: LabelTuned(ds.minutes, nested=True),
        "hours": lambda: Hours(),
        "debris": lambda: Debris(ds.debris),
    }
    return reg
