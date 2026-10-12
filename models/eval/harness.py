"""Common evaluation protocol: one set of folds, metrics, gear-level uncertainty and detection for every approach.

The independent unit is the gear (4 training gears, all with per-tooth condition labels).
T1 (primary) = mean tooth damage from data/labels/condition_labels.csv (mean over the 28 teeth of score - 1);
T2 = the earlier by-eye photo proxy (photo_damage). A spall event is a run end where any tooth reaches 2 (onset).
Per-gear numbers are the primary evidence; bootstrap ranges are descriptive and sign-flip p-values are reported next to their floor 1/2**n_gears.
"""

import itertools
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from spall_hi import (  # noqa: E402
    BASE_H,
    BREAK_IN_H,
    SPLITS,
    challenge_score,
    condition_correct,
    load_debris,
    load_minutes,
    photo_damage,
    run_ends,
)
from spall_supervised import load_labels  # noqa: E402

SEED = 696
TRAIN = SPLITS["train"]
OUT = Path(__file__).resolve().parent / "out"
RECREATE = "python scripts/phm_data.py small && python scripts/flatten_hdf5.py"
T1, T2 = "mean_ex", "photo"  # evaluation targets: condition labels (primary), by-eye photo proxy (secondary)
SPALL_ONSET = 2.0  # tooth score at spall onset (2 = onset, 3 = spalling, 4 = severe spalling)


@dataclass
class Dataset:
    minutes: pd.DataFrame  # raw steady minutes (load_minutes), all training gears
    z: pd.DataFrame  # condition-corrected minutes
    ends: pd.DataFrame  # experiment, run_num, end_h
    lab: pd.DataFrame  # run-level condition-label targets for every training gear (load_labels)
    debris: pd.DataFrame  # experiment, run_num, debris_rate


def load_dataset(
    data=Path("data"),
    labels=Path("data/labels/condition_labels.csv"),
    *,
    base_h=BASE_H,
    break_in_h=BREAK_IN_H,
    steady=True,
    correct=True,
    per_gear=True,
    minutes=None,
) -> Dataset:
    data = Path(data)
    if not (data / "train" / "lf_1s.parquet").exists():
        raise FileNotFoundError(f"{data / 'train' / 'lf_1s.parquet'} not found. Recreate it with: {RECREATE}")
    m = load_minutes(data, "train", steady=steady) if minutes is None else minutes
    z = condition_correct(m, base_h, break_in_h=break_in_h, correct=correct, per_gear=per_gear)
    return Dataset(m, z, run_ends(data, "train"), load_labels(Path(labels)), load_debris(data, "train"))


def with_debris_feature(ds: Dataset) -> Dataset:
    """Add debris_z: log of each run's large-particle rate over the gear's first-run rate (0 in run 1)."""
    d = ds.debris.sort_values(["experiment", "run_num"]).copy()
    d["debris_z"] = np.log(
        d.debris_rate.clip(lower=1e-3) / d.groupby("experiment").debris_rate.transform("first").clip(lower=1e-3)
    )
    z = ds.z.merge(d[["experiment", "run_num", "debris_z"]], on=["experiment", "run_num"], how="left")
    z.index = ds.z.index
    return replace(ds, z=z.assign(debris_z=z.debris_z.fillna(0.0)))


def folds(gears=TRAIN) -> list[tuple[list[str], str]]:
    return [([g for g in gears if g != h], h) for h in gears]


def spall_flags(runs: pd.DataFrame) -> np.ndarray:
    """Run ends where any tooth has reached spall onset (needs the label column `max`)."""
    return (runs["max"] >= SPALL_ONSET).to_numpy()


@dataclass
class FoldResult:
    runs: pd.DataFrame
    models: dict = field(default_factory=dict)  # held-out gear -> fitted approach
    fit_s: float = 0.0
    score_s: float = 0.0


def run_folds(make: Callable, ds: Dataset) -> FoldResult:
    """Fit a fresh approach on the other training gears and score the held-out gear, for every gear."""
    parts, models, fit_s, score_s = [], {}, 0.0, 0.0
    for train, held in folds():
        a = make()
        t0 = time.perf_counter()
        a.fit(
            ds.z[ds.z.experiment.isin(train)],
            ds.ends[ds.ends.experiment.isin(train)],
            ds.lab[ds.lab.experiment.isin(train)],
        )
        t1 = time.perf_counter()
        r = a.score(ds.z[ds.z.experiment == held], ds.ends[ds.ends.experiment == held])
        score_s += time.perf_counter() - t1
        fit_s += t1 - t0
        r = r.assign(causal=r["causal"] if "causal" in r else r.score)  # hours/debris are causal already
        parts.append(r[["experiment", "run_num", "end_h", "score", "causal"]].assign(threshold=a.threshold))
        models[held] = a
    runs = pd.concat(parts, ignore_index=True).sort_values(["experiment", "run_num"], ignore_index=True)
    runs["photo"] = photo_damage(runs)
    runs = runs.merge(ds.lab[["experiment", "run_num", "mean_ex", "max"]], on=["experiment", "run_num"], how="left")
    runs["spall"] = spall_flags(runs)
    return FoldResult(runs, models, fit_s, score_s)


def pooled_residuals(score, truth) -> np.ndarray:
    """Residual of each run end under one isotonic map shared by all gears (truth scaled to [0, 1])."""
    score, truth = np.asarray(score, float), np.asarray(truth, float)
    truth = truth / max(truth.max(), 1e-9)
    return IsotonicRegression(out_of_bounds="clip").fit(score, truth).predict(score) - truth


def evaluate_runs(runs: pd.DataFrame, target: str = T1) -> tuple[dict, pd.Series]:
    """challenge_score on the run ends that carry `target`, plus gear error (pooled-map residuals per gear)."""
    x = runs[runs[target].notna()]
    s = challenge_score(x.score.to_numpy(float), x[target].to_numpy(float), x.experiment.to_numpy())
    res = pooled_residuals(x.score, x[target])
    return s, pd.Series(res**2, index=x.experiment.to_numpy()).groupby(level=0).mean()


def sign_flip_p(diffs) -> float:
    """Exact one-sided paired sign-flip p for mean(diffs) > 0 (diffs = baseline error - approach error)."""
    d = np.asarray(diffs, float)
    obs = d.mean()
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(np.mean((flips * np.abs(d)).mean(1) >= obs - 1e-12))


def p_floor(n: int) -> float:
    return 1 / 2**n


def bootstrap_range(values, n: int = 2000) -> tuple[float, float]:
    """Descriptive 2.5-97.5% range of the mean over gears resampled with replacement."""
    v = np.asarray(values, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (n, len(v)))
    lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def _alarm(runs: pd.DataFrame) -> pd.Series:
    return runs["causal"] if "causal" in runs else runs.score


def detection(runs: pd.DataFrame) -> dict:
    """Spall run ends caught/missed; false alarms are flags on run ends before any tooth reached spall onset.

    Alarms use the causal series (no information from later runs) when present.
    """
    flag = _alarm(runs) > runs.threshold
    caught = int((flag & runs.spall).sum())
    missed = int((~flag & runs.spall).sum())
    false = int((flag & ~runs.spall).sum())
    denom = 2 * caught + missed + false
    return dict(caught=caught, missed=missed, false_alarms=false, f1=2 * caught / denom if denom else np.nan)


def lead_times(runs: pd.DataFrame) -> pd.Series:
    """Hours from the first flagged run end to the first spall run end, per gear with a spall (NaN = never flagged)."""
    out = {}
    for g, r in runs.groupby("experiment"):
        if not r.spall.any():
            continue
        flagged = r.loc[_alarm(r) > r.threshold, "end_h"]
        out[g] = r.loc[r.spall, "end_h"].min() - flagged.min() if len(flagged) else np.nan
    return pd.Series(out, dtype=float)


def record(**numbers):
    """Merge numbers quoted in the report into out/report_numbers.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "report_numbers.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in numbers.items()})
    p.write_text(json.dumps(cur, indent=1, sort_keys=True))


def save_fig(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    return p
