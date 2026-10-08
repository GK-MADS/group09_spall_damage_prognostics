"""Unsupervised gear-damage health index (HI) for the PHM 2026 data challenge.

No damage labels are used to fit anything. Every model is fit on *healthy* data only
(the first hours of each training gear), and damage is scored as distance from that
healthy state. The photo labels in eda/scripts/compute.py (EYE) are used only to
compare candidate HIs with the challenge's scoring rule.

Pipeline
  1. 1-s rows -> steady-state filter -> 1-min medians            (load_minutes)
  2. log|CI| -> regress out temperature/torque per gear          (condition_correct)
  3. standardise per gear on its own healthy baseline window     (condition_correct)
  4. fit healthy-state models on pooled training baselines       (HealthyModel)
  5. HI per minute -> value at each run end, made monotone       (run_end_scores)

Candidates: all-CI novelty (maha, spe, zpos, zmax) and a directional tooth-damage HI on
FM4/M6A/M8A/NA4 (tooth, tooth_maha). The all-CI scores are dominated by the level shifts
that follow every rig restart; the tooth HIs stay flat on healthy runs and rise at spall.

Usage
  python models/spall_hi.py                 # evaluate candidates, write submissions to models/out/
  python models/spall_hi.py --data /path    # folder holding <split>/lf_1s.parquet + particles.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

SPLITS = {"train": ["Exp-A", "Exp-B", "Exp-C", "Exp-F"], "validation": ["Exp-G"], "test": ["Exp-D", "Exp-E"]}
# Gear-mesh CIs with a damage interpretation. Excluded: *star variants (baselined on another rig
# run), fleet_wtmade (trained on other experiments), raw rms/ptp/v_peak (load-dominated).
FEATS = [
    "ci_fm4",
    "ci_m6a",
    "ci_m8a",
    "ci_na4",
    "ci_e_ratio",
    "ci_kurtosis_tsa",
    "ci_rms_tsa",
    "ci_alr",
    "ci_fm0",
    "ci_crest_tsa",
    "ci_kurtosis_e_op",
    "ci_rms_e_op",
]
# Localised-tooth-damage CIs (sidebands / residual impulsiveness around mesh). In EDA these stay near 0 on
# healthy runs, while the energy and global CIs above shift by tens of sigma after each rig restart.
TOOTH = ["ci_fm4", "ci_m6a", "ci_m8a", "ci_na4"]
COND = ["ctx_temperature", "ctx_pau_torque"]
BREAK_IN_H = 0.5  # skip the first half hour of every gear (run-in transients)
BASE_H = 6.0  # healthy window = [BREAK_IN_H, BASE_H]: inside run 1 of every gear
TAIL_MIN = 60  # run-end value = median HI over the last TAIL_MIN steady minutes of the run
DEBRIS_BINS = [f"dm4500_bin_{i}" for i in range(2, 8)]  # larger ferrous particles

# Photo-based damage per (gear, inspection-after-run) from the by-eye labels in eda/scripts/compute.py:
# initiation = 1, spall = 2, summed over teeth. Evaluation only; never used for fitting.
EYE = {
    ("Exp-A", 16, 4): 1,
    ("Exp-A", 28, 4): 1,
    ("Exp-B", 11, 6): 1,
    ("Exp-B", 14, 6): 1,
    ("Exp-F", 9, 7): 1,
    ("Exp-F", 12, 7): 1,
    ("Exp-F", 13, 7): 1,
}
EYE.update(
    {
        k: 2
        for k in [
            ("Exp-A", 16, 5),
            ("Exp-A", 24, 5),
            ("Exp-A", 28, 5),
            ("Exp-B", 11, 7),
            ("Exp-B", 14, 7),
            ("Exp-C", 6, 12),
            ("Exp-C", 7, 12),
            ("Exp-F", 9, 8),
            ("Exp-F", 12, 8),
            ("Exp-F", 13, 8),
            ("Exp-F", 20, 8),
            ("Exp-F", 19, 8),
            ("Exp-F", 18, 6),
            ("Exp-F", 18, 7),
            ("Exp-F", 18, 8),
        ]
    }
)


# ---------------------------------------------------------------- data
def load_minutes(data: Path, split: str) -> pd.DataFrame:
    lf = pd.read_parquet(data / split / "lf_1s.parquet")
    ci = [c for c in lf if c.startswith(("ci_", "ctx_"))]
    # Exp-A Run-2 is a byte copy of Run-1 at the source: keep the hour axis, drop the values
    lf.loc[(lf.experiment == "Exp-A") & (lf.run_num == 2), ci] = np.nan
    nom_t = lf.groupby("experiment").ctx_pau_torque.transform("median")
    steady = (lf.ctx_pau_speed.sub(1400).abs() < 20) & (lf.ctx_pau_torque.sub(nom_t).abs() < 0.05 * nom_t)
    lf = lf[steady].assign(minute=(lf.test_hours * 60).astype(int))
    m = lf.groupby(["experiment", "run_num", "minute"])[FEATS + COND].median().reset_index()
    m["hours"] = (m.minute + 0.5) / 60
    return m.dropna(subset=FEATS + COND)


def load_debris(data: Path, split: str) -> pd.DataFrame:
    """Large-particle count rate per run (particles/hour), from the cumulative DM4500 counters."""
    p = pd.read_parquet(data / split / "particles.parquet").sort_values(["experiment", "timestamp"])
    p = p[~((p.experiment == "Exp-A") & (p.run_num == 2))]
    p["dm"] = p[DEBRIS_BINS].sum(axis=1)
    g = p.groupby(["experiment", "run_num"]).agg(
        c0=("dm", "first"), c1=("dm", "last"), t0=("timestamp", "first"), t1=("timestamp", "last")
    )
    hrs = (g.t1 - g.t0).dt.total_seconds() / 3600
    return ((g.c1 - g.c0) / hrs.clip(lower=0.25)).rename("debris_rate").reset_index()


def run_ends(data: Path, split: str) -> pd.DataFrame:
    lf = pd.read_parquet(data / split / "lf_1s.parquet", columns=["experiment", "run_num", "test_hours"])
    return lf.groupby(["experiment", "run_num"]).test_hours.max().rename("end_h").reset_index()


# ---------------------------------------------------------------- features
def condition_correct(m: pd.DataFrame, base_h: float = BASE_H) -> pd.DataFrame:
    """log|CI| with temperature/torque regressed out and scaled on each gear's own healthy window.

    Gears run at different loads (Exp-A at 120 ft-lbf, the rest at 100) and every gear has its own
    healthy vibration signature, so each gear is referenced to itself; the pooled models then only
    see departures from that gear's normal.
    """
    out = []
    for _, g in m.groupby("experiment"):
        X = np.log(g[FEATS].abs().to_numpy() + 1e-9)
        base = ((g.hours > BREAK_IN_H) & (g.hours < base_h)).to_numpy()
        C = g[COND].to_numpy()
        C = (C - C[base].mean(0)) / (C[base].std(0) + 1e-9)
        T = np.c_[np.ones(len(g)), C]
        beta = np.linalg.lstsq(T[base], X[base], rcond=None)[0]
        R = X - T @ beta
        med = np.median(R[base], 0)
        mad = np.median(np.abs(R[base] - med), 0) * 1.4826
        Z = (R - med) / np.where(mad > 0, mad, 1)
        z = pd.DataFrame(Z, columns=FEATS, index=g.index)
        z[["experiment", "run_num", "hours"]] = g[["experiment", "run_num", "hours"]]
        z["base"] = base
        out.append(z)
    return pd.concat(out)


class HealthyModel:
    """Healthy-state models fit on pooled baseline minutes of the training gears only."""

    def __init__(self, tooth: list[str] = TOOTH):
        self.tooth = tooth

    def fit(self, z: pd.DataFrame):
        B = z.loc[z.base, FEATS].to_numpy()
        self.lw = LedoitWolf().fit(B)
        self.lw_t = LedoitWolf().fit(z.loc[z.base, self.tooth].to_numpy())
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(B)
        R = B - self.pca.inverse_transform(self.pca.transform(B))
        self.spe_scale = np.median((R**2).sum(1))
        return self

    def score(self, z: pd.DataFrame) -> pd.DataFrame:
        X = z[FEATS].to_numpy()
        R = X - self.pca.inverse_transform(self.pca.transform(X))
        s = z[["experiment", "run_num", "hours"]].copy()
        s["maha"] = np.sqrt(self.lw.mahalanobis(X) / len(FEATS))
        s["spe"] = np.sqrt((R**2).sum(1) / self.spe_scale)
        # directional: damage raises impulsiveness/energy CIs, so only upward departures count
        s["zpos"] = np.sqrt((np.clip(X, 0, None) ** 2).mean(1))
        s["zmax"] = np.abs(X).max(1)
        T = z[self.tooth].to_numpy()
        s["tooth"] = np.clip(T, 0, None).mean(1)
        s["tooth_maha"] = np.sqrt(self.lw_t.mahalanobis(T) / len(self.tooth)) * (T.mean(1) > 0)
        return s


# ---------------------------------------------------------------- run-end scores
def _iso(v: pd.Series) -> pd.Series:
    return pd.Series(IsotonicRegression().fit_transform(np.arange(len(v)), v.to_numpy()), index=v.index)


def run_end_scores(
    s: pd.DataFrame, ends: pd.DataFrame, cols: list[str], monotone: str = "iso", tail: int = TAIL_MIN
) -> pd.DataFrame:
    """One value per run end: median of the last `tail` minutes, carried forward, then made monotone.

    Spalls do not heal, so the reported damage never goes down. "iso" fits a non-decreasing step
    function per gear (a one-run spike is averaged with its neighbours); "cummax" locks spikes in.
    Runs with no usable minutes (Exp-A Run-2) inherit the previous value.
    """
    tail = (
        s.sort_values("hours")
        .groupby(["experiment", "run_num"])
        .tail(tail)
        .groupby(["experiment", "run_num"])[cols]
        .median()
        .reset_index()
    )
    r = ends.merge(tail, on=["experiment", "run_num"], how="left").sort_values(["experiment", "run_num"])
    r[cols] = r.groupby("experiment")[cols].ffill().fillna(0)
    if monotone == "cummax":
        r[cols] = r.groupby("experiment")[cols].cummax()
    elif monotone == "iso":
        r[cols] = r.groupby("experiment")[cols].transform(_iso)
    return r


# ---------------------------------------------------------------- evaluation
def photo_damage(ends: pd.DataFrame) -> pd.Series:
    """Photo damage after each training run (sum of by-eye tooth scores); evaluation only."""
    d = pd.Series(EYE).rename_axis(["experiment", "tooth", "run_num"]).groupby(level=[0, 2]).sum()
    idx = pd.MultiIndex.from_frame(ends[["experiment", "run_num"]])
    return d.reindex(idx, fill_value=0).to_numpy(dtype=float)


def challenge_score(pred, truth, groups=None) -> dict:
    """Approximate the organisers' metric: MSE after the best monotone (isotonic) map of pred onto truth.

    pooled: one map shared by all gears, so the HI scale has to agree across gears.
    per_gear: a separate map per gear (only trajectory shape matters). Truth is scaled to [0, 1].
    """
    pred, truth = np.asarray(pred, float), np.asarray(truth, float) / max(np.max(truth), 1e-9)
    pooled = IsotonicRegression(out_of_bounds="clip").fit(pred, truth).predict(pred)
    out = {"mse_pooled": float(np.mean((pooled - truth) ** 2)), "spearman": float(spearmanr(pred, truth)[0])}
    if groups is not None:
        err = np.zeros_like(truth)
        for g in np.unique(groups):
            k = groups == g
            err[k] = IsotonicRegression(out_of_bounds="clip").fit(pred[k], truth[k]).predict(pred[k]) - truth[k]
        out["mse_per_gear"] = float(np.mean(err**2))
    return out


def evaluate(z_tr, ends_tr, debris_tr, cols, monotone="iso") -> pd.DataFrame:
    """Leave-one-gear-out: healthy models are refit without the held-out gear, which is then scored."""
    parts = []
    for held in SPLITS["train"]:
        hm = HealthyModel().fit(z_tr[z_tr.experiment != held])
        parts.append(hm.score(z_tr[z_tr.experiment == held]))
    s = pd.concat(parts)
    r = run_end_scores(s, ends_tr, cols, monotone).merge(debris_tr, on=["experiment", "run_num"], how="left")
    # debris relative to each gear's run-1 rate, made monotone like the vibration HIs
    r["debris"] = r.debris_rate / r.groupby("experiment").debris_rate.transform("first")
    r["debris"] = r.groupby("experiment").debris.ffill()
    r["debris"] = r.groupby("experiment").debris.cummax()
    truth = photo_damage(r)
    rows = []
    for c in cols + ["debris", "hours_proxy"]:
        pred = r.end_h.to_numpy() if c == "hours_proxy" else r[c].to_numpy()
        rows.append({"hi": c, **challenge_score(pred, truth, r.experiment.to_numpy())})
    return pd.DataFrame(rows).set_index("hi").round(4), r.assign(photo=truth)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("models/out"))
    ap.add_argument("--monotone", default="iso", choices=["none", "cummax", "iso"])
    ap.add_argument("--hi", default=None, help="HI column to submit (default: best pooled MSE in LOGO)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["maha", "spe", "zpos", "zmax", "tooth", "tooth_maha"]

    z = {s: condition_correct(load_minutes(a.data, s)) for s in SPLITS}
    ends = {s: run_ends(a.data, s) for s in SPLITS}

    debris = load_debris(a.data, "train")
    for mode in ["none", "cummax", "iso"]:
        t, _ = evaluate(z["train"], ends["train"], debris, cols, mode)
        print(f"\n[{mode}] leave-one-gear-out vs photo labels (lower mse / higher spearman is better):")
        print(t.to_string())
    table, r_tr = evaluate(z["train"], ends["train"], debris, cols, a.monotone)
    print(r_tr[["experiment", "run_num", "end_h", "photo"] + cols].round(2).to_string(index=False))
    best = a.hi or table.loc[cols, "mse_pooled"].idxmin()
    print(f"\nsubmitting HI = {best}")

    hm = HealthyModel().fit(z["train"])
    for split, tmpl in [("validation", "example_validation_submission.csv"), ("test", "example_submission.csv")]:
        r = run_end_scores(hm.score(z[split]), ends[split], cols, a.monotone)
        sub = pd.read_csv(a.data / split / tmpl)
        r["run_id"] = r.experiment.str[-1]
        # match each template time to the nearest run end of that gear
        sub["metric"] = [
            r.loc[(r.run_id == g), [best, "end_h"]].pipe(lambda x: x[best].iloc[(x.end_h - t).abs().argmin()])
            for g, t in zip(sub.run_id, sub.time)
        ]
        sub["metric"] = sub.metric.round(4)
        sub.to_csv(a.out / f"{split}_submission.csv", index=False)
        print(f"\n{split}:\n{sub.to_string(index=False)}")
    table.to_csv(a.out / "logo_eval.csv")
    json.dump(
        {
            "hi": best,
            "monotone": a.monotone,
            "feats": FEATS,
            "tooth": TOOTH,
            "base_h": [BREAK_IN_H, BASE_H],
            "tail_min": TAIL_MIN,
        },
        open(a.out / "config.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
