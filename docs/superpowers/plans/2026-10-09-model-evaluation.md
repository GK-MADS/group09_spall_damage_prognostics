# Model Evaluation and Failure Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `models/eval/`, a shared evaluation harness plus analysis scripts that produce every figure, table and
number for the failure-analysis section of the PDF report, and a notebook that presents them.

**Architecture:** One harness (folds, metrics, gear-level uncertainty, detection) drives thin adapters that wrap the
existing models in `models/spall_hi.py` and `models/spall_supervised.py`, plus a new Isolation Forest. Each Step-1/Step-2
analysis is its own script writing CSV/PNG/JSON to `models/eval/out/`; `run_all.py` runs them in order; the notebook
only reads `out/`.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, scipy, shap, lime, matplotlib, nbformat, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-model-evaluation-design.md`

## Global Constraints

- `SEED = 696` everywhere randomness appears (bootstrap, Isolation Forest, LIME).
- Training gears only: Exp-A, Exp-B, Exp-C, Exp-F. Validation/test gears are never used for evaluation.
- Folds: leave-one-gear-out over the 4 training gears. T1 (photo damage) scored on all 4; T2 (`mean_ex` tooth labels) only on the Exp-A and Exp-B folds.
- Gear error = squared residual under one pooled isotonic map per approach, averaged within gear. `mse_per_gear` is reported, never used to compare.
- Every p-value is printed next to its floor: 1/2**n_gears (0.0625 for 4 gears, 0.25 for 2).
- Gear-level bootstrap: 2,000 draws, described as a descriptive range, never as an inferential interval. No run-end bootstrap.
- Detection: minute-scored approaches flag a run end above the 95th percentile of their minute scores inside the training-fold healthy-baseline window; run-level baselines (hours, debris) use the maximum run-1 value of the training-fold gears.
- Winner rule: (1) most gears beating elapsed hours on T1 gear error; (2) lowest T1 `mse_pooled`; (3) fewest `n_tuned`.
- Existing behaviour of `spall_hi.py` / `spall_supervised.py` and the files in `models/out/` must not change; new parameters keep current defaults.
- Figures: PNG at 300 dpi. Tables: CSV. Every number quoted in prose: key in `models/eval/out/report_numbers.json`.
- Code style: ruff (E, F, I), 120-char lines, double quotes, 4-space indent; comments as sparse as in `models/spall_hi.py`.
- Run everything from the repo root with `.venv/bin/python -m ...`; tests with `.venv/bin/python -m pytest`.

## Review Focus

- A run end with no usable minutes (Exp-A Run-2 is nulled in `load_minutes`): every adapter must still return a row for it, carrying the previous value forward. Test in Task 3.
- A gear where the approach never raises an alarm: lead time is NaN (reported as missed), not 0 and not an exception. Test in Task 2.
- A fold where the supervised model has only one labelled gear (Exp-A held out → train on Exp-B only), and a call with no labelled gear: the first works, the second raises a clear `ValueError`. Test in Task 3.
- A constant score (e.g. a baseline that never moves): metrics return NaN Spearman and finite MSE, and the comparison still completes. Test in Task 2.
- Parquet tables not downloaded: `load_dataset` raises `FileNotFoundError` naming the recreate command. Test in Task 2.

---

## File structure

| File | Responsibility |
|---|---|
| `models/spall_hi.py` (modify) | expose `feats` on `HealthyModel`; `steady` on `load_minutes`; `break_in_h`, `correct`, `per_gear` on `condition_correct` |
| `models/eval/__init__.py` | puts `models/` on `sys.path` so `spall_hi` / `spall_supervised` import |
| `models/eval/harness.py` | `Dataset`, `load_dataset`, `folds`, `run_folds`, truth columns, `evaluate_runs`, `gear_errors`, `sign_flip_p`, `bootstrap_range`, `detection`, `lead_times`, `record`, `save_fig` |
| `models/eval/adapters.py` | `HI`, `IForest`, `RidgeApproach`, `LabelTuned`, `Hours`, `Debris`, `registry` |
| `models/eval/compare.py` | Step 2 master table, figures, winner, `runs.csv` |
| `models/eval/ablation.py` | Step 1.1 |
| `models/eval/sensitivity.py` | Step 1.2 |
| `models/eval/curves.py` | Step 1.3 |
| `models/eval/failures.py` | Step 1.4: candidates, example selection, SHAP / LIME / exact attributions, figures |
| `models/eval/run_all.py` | runs all analyses on one loaded dataset |
| `models/eval/cells.py`, `models/eval/build_notebook.py` | notebook sources and builder |
| `models/eval/spall_eval.ipynb` | generated notebook |
| `tests/test_eval_harness.py` | fast synthetic tests + one slow real-data test |

---

### Task 0: Dependencies and package skeleton

**Files:**
- Create: `models/eval/__init__.py`
- Modify: `README.md` (install line)
- Modify: `.gitignore` (nothing to add; `models/eval/out/` is committed like `models/out/`)

**Interfaces:**
- Produces: importable package `models.eval`; after `import models.eval`, `import spall_hi` works.

- [ ] **Step 1: Install the four new packages**

Run: `uv pip install --python .venv/bin/python shap lime matplotlib nbformat`
Expected: installs without error. Verify: `.venv/bin/python -c "import shap, lime, matplotlib, nbformat; print('ok')"` prints `ok`.

- [ ] **Step 2: Create the package init**

```python
"""Evaluation and failure analysis for the spall models (see docs/superpowers/specs/2026-10-09-model-evaluation-design.md)."""

import sys
from pathlib import Path

_MODELS = str(Path(__file__).resolve().parents[1])
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)
```

- [ ] **Step 3: Verify the import path**

Run: `.venv/bin/python -c "import models.eval, spall_hi, spall_supervised; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Update the README install line**

In `README.md`, section "Unsupervised damage model", replace

```
uv pip install --python .venv/bin/python scikit-learn scipy pytest
```

with

```
uv pip install --python .venv/bin/python scikit-learn scipy pytest shap lime matplotlib nbformat
```

- [ ] **Step 5: Commit**

```bash
git add models/eval/__init__.py README.md
git commit -m "Add models/eval package skeleton and evaluation dependencies"
```

---

### Task 1: Expose ablation parameters in `spall_hi.py` without changing defaults

**Files:**
- Modify: `models/spall_hi.py` (`load_minutes` ~L97, `condition_correct` ~L128, `HealthyModel` ~L154)
- Test: `tests/test_eval_harness.py` (new file)

**Interfaces:**
- Produces:
  - `load_minutes(data: Path, split: str, steady: bool = True) -> pd.DataFrame`
  - `condition_correct(m, base_h=BASE_H, break_in_h=BREAK_IN_H, correct=True, per_gear=True) -> pd.DataFrame`
  - `HealthyModel(tooth=TOOTH, feats=FEATS)`; `.feats`, `.tooth`, `.lw`, `.lw_t` attributes

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_harness.py`:

```python
import numpy as np
import pandas as pd
import pytest

import models.eval  # noqa: F401  (puts models/ on sys.path)
from spall_hi import BASE_H, BREAK_IN_H, COND, FEATS, TOOTH, HealthyModel, condition_correct

GEARS = ["Exp-A", "Exp-B", "Exp-C", "Exp-F"]


def synth_minutes(gears=GEARS, runs=4, per_run=120, seed=0, drop=()):
    """Raw-minute frame shaped like load_minutes output. Tooth CIs rise in the last run."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in gears:
        for r in range(1, runs + 1):
            if (g, r) in drop:
                continue
            h = (r - 1) * 6 + np.linspace(0.6, 5.9, per_run)
            d = {c: np.exp(rng.normal(0, 0.1, per_run)) for c in FEATS}
            if r == runs:
                for c in TOOTH:
                    d[c] = d[c] * 3
            d["ctx_temperature"] = 60 + rng.normal(0, 1, per_run)
            d["ctx_pau_torque"] = 100 + rng.normal(0, 1, per_run)
            rows.append(pd.DataFrame(dict(experiment=g, run_num=r, minute=(h * 60).astype(int), hours=h, **d)))
    return pd.concat(rows, ignore_index=True)


def synth_ends(gears=GEARS, runs=4):
    return pd.DataFrame(
        [dict(experiment=g, run_num=r, end_h=r * 6.0) for g in gears for r in range(1, runs + 1)]
    )


def synth_labels(runs=4):
    return pd.DataFrame(
        [
            dict(experiment=g, run_num=r, mean_ex=[0, 0, 0.1, 0.4][r - 1], max=1.0, top3=0.0)
            for g in ["Exp-A", "Exp-B"]
            for r in range(1, runs + 1)
        ]
    )


def _condition_correct_original(m, base_h=BASE_H):
    """Frozen copy of condition_correct before the refactor (regression reference)."""
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


def test_condition_correct_defaults_unchanged():
    m = synth_minutes()
    pd.testing.assert_frame_equal(condition_correct(m), _condition_correct_original(m))
    pd.testing.assert_frame_equal(condition_correct(m, 3.0), _condition_correct_original(m, 3.0))


def test_condition_correct_ablation_switches_change_output():
    m = synth_minutes()
    base = condition_correct(m)
    for kw in [dict(correct=False), dict(per_gear=False), dict(break_in_h=0.0)]:
        alt = condition_correct(m, **kw)
        assert list(alt.columns) == list(base.columns)
        assert not np.allclose(alt[FEATS].to_numpy(), base[FEATS].to_numpy())


def test_healthy_model_feats_default_and_subset():
    z = condition_correct(synth_minutes())
    a = HealthyModel().fit(z).score(z)
    b = HealthyModel(TOOTH, FEATS).fit(z).score(z)
    pd.testing.assert_frame_equal(a, b)
    c = HealthyModel(TOOTH, TOOTH).fit(z).score(z)
    assert c.maha.notna().all() and not np.allclose(c.maha, a.maha)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: `test_condition_correct_defaults_unchanged` PASSES (no change yet); the other two FAIL with `TypeError: ... unexpected keyword argument`.

- [ ] **Step 3: Implement**

In `models/spall_hi.py`, replace `load_minutes`'s filter lines:

```python
def load_minutes(data: Path, split: str, steady: bool = True) -> pd.DataFrame:
    lf = pd.read_parquet(data / split / "lf_1s.parquet")
    ci = [c for c in lf if c.startswith(("ci_", "ctx_"))]
    # Exp-A Run-2 is a byte copy of Run-1 at the source: keep the hour axis, drop the values
    lf.loc[(lf.experiment == "Exp-A") & (lf.run_num == 2), ci] = np.nan
    if steady:
        nom_t = lf.groupby("experiment").ctx_pau_torque.transform("median")
        keep = (lf.ctx_pau_speed.sub(1400).abs() < 20) & (lf.ctx_pau_torque.sub(nom_t).abs() < 0.05 * nom_t)
        lf = lf[keep]
    lf = lf.assign(minute=(lf.test_hours * 60).astype(int))
    m = lf.groupby(["experiment", "run_num", "minute"])[FEATS + COND].median().reset_index()
    m["hours"] = (m.minute + 0.5) / 60
    return m.dropna(subset=FEATS + COND)
```

Replace `condition_correct` (keep its docstring, add one line about the switches):

```python
def condition_correct(
    m: pd.DataFrame,
    base_h: float = BASE_H,
    break_in_h: float = BREAK_IN_H,
    correct: bool = True,
    per_gear: bool = True,
) -> pd.DataFrame:
    """log|CI| with temperature/torque regressed out and scaled on each gear's own healthy window.

    Gears run at different loads (Exp-A at 120 ft-lbf, the rest at 100) and every gear has its own
    healthy vibration signature, so each gear is referenced to itself; the pooled models then only
    see departures from that gear's normal. correct=False skips the regression and per_gear=False
    uses one reference pooled over all gears in `m` (both for ablation only).
    """

    def prep(g):
        X = np.log(g[FEATS].abs().to_numpy() + 1e-9)
        base = ((g.hours > break_in_h) & (g.hours < base_h)).to_numpy()
        return X, g[COND].to_numpy(), base

    def reference(X, C, base):
        mu, sd = C[base].mean(0), C[base].std(0) + 1e-9
        T = np.c_[np.ones(len(C)), (C - mu) / sd]
        beta = np.linalg.lstsq(T[base], X[base], rcond=None)[0] if correct else np.zeros((T.shape[1], X.shape[1]))
        R = X - T @ beta
        med = np.median(R[base], 0)
        mad = np.median(np.abs(R[base] - med), 0) * 1.4826
        return mu, sd, beta, med, np.where(mad > 0, mad, 1)

    def apply(X, C, ref):
        mu, sd, beta, med, mad = ref
        T = np.c_[np.ones(len(C)), (C - mu) / sd]
        return (X - T @ beta - med) / mad

    pooled = None if per_gear else reference(*prep(m))
    out = []
    for _, g in m.groupby("experiment"):
        X, C, base = prep(g)
        Z = apply(X, C, pooled if pooled is not None else reference(X, C, base))
        z = pd.DataFrame(Z, columns=FEATS, index=g.index)
        z[["experiment", "run_num", "hours"]] = g[["experiment", "run_num", "hours"]]
        z["base"] = base
        out.append(z)
    return pd.concat(out)
```

In `HealthyModel`, add `feats` and use it in place of the module constant:

```python
class HealthyModel:
    """Healthy-state models fit on pooled baseline minutes of the training gears only."""

    def __init__(self, tooth: list[str] = TOOTH, feats: list[str] = FEATS):
        self.tooth, self.feats = tooth, feats

    def fit(self, z: pd.DataFrame):
        B = z.loc[z.base, self.feats].to_numpy()
        self.lw = LedoitWolf().fit(B)
        self.lw_t = LedoitWolf().fit(z.loc[z.base, self.tooth].to_numpy())
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(B)
        R = B - self.pca.inverse_transform(self.pca.transform(B))
        self.spe_scale = np.median((R**2).sum(1))
        return self

    def score(self, z: pd.DataFrame) -> pd.DataFrame:
        X = z[self.feats].to_numpy()
        R = X - self.pca.inverse_transform(self.pca.transform(X))
        s = z[["experiment", "run_num", "hours"]].copy()
        s["maha"] = np.sqrt(self.lw.mahalanobis(X) / len(self.feats))
        # ... remaining lines unchanged
```

(The remaining lines of `score` — `spe`, `zpos`, `zmax`, `tooth`, `tooth_maha` — stay as they are; they use `X`, `T`, `self.tooth`.)

- [ ] **Step 4: Run the new and existing tests**

Run: `.venv/bin/python -m pytest tests -v -m "not slow"`
Expected: all PASS, including the 6 existing tests in `tests/test_spall_hi.py`.

- [ ] **Step 5: Confirm the shipped outputs are unchanged**

Run: `.venv/bin/python models/spall_hi.py > /dev/null && PYTHONPATH=models .venv/bin/python models/spall_supervised.py > /dev/null && git diff --stat models/out`
Expected: no changes listed under `models/out/` (outputs byte-identical).

- [ ] **Step 6: Commit**

```bash
git add models/spall_hi.py tests/test_eval_harness.py
git commit -m "Expose ablation switches on load_minutes, condition_correct and HealthyModel

Defaults unchanged; a frozen copy of the old condition_correct pins that in tests."
```

---

### Task 2: Evaluation harness

**Files:**
- Create: `models/eval/harness.py`
- Test: `tests/test_eval_harness.py` (append)

**Interfaces:**
- Consumes: `condition_correct`, `load_minutes`, `load_debris`, `run_ends`, `photo_damage`, `challenge_score`, `EYE`, `SPLITS`, `BASE_H`, `BREAK_IN_H` from `spall_hi`; `load_labels`, `LABELLED` from `spall_supervised`.
- Produces:
  - `SEED: int = 696`, `TRAIN: list[str]`, `OUT: Path`
  - `@dataclass Dataset(minutes, z, ends, lab, debris)` (all `pd.DataFrame`)
  - `load_dataset(data=Path("data"), labels=Path("data/labels/spall_scores.csv"), *, base_h=BASE_H, break_in_h=BREAK_IN_H, steady=True, correct=True, per_gear=True, minutes=None) -> Dataset`
  - `with_debris_feature(ds: Dataset) -> Dataset` (adds column `debris_z` to `z`)
  - `folds(gears=TRAIN) -> list[tuple[list[str], str]]`
  - `@dataclass FoldResult(runs: pd.DataFrame, models: dict[str, object], fit_s: float, score_s: float)`; `runs` columns: `experiment, run_num, end_h, score, threshold, photo, spall, mean_ex`
  - `run_folds(make: Callable[[], Approach], ds: Dataset) -> FoldResult`
  - `pooled_residuals(score, truth) -> np.ndarray`
  - `evaluate_runs(runs, target="photo") -> tuple[dict, pd.Series]` (dict keys `mse_pooled, spearman, mse_per_gear`; Series = gear error indexed by experiment)
  - `sign_flip_p(diffs) -> float`, `p_floor(n: int) -> float`
  - `bootstrap_range(values, n=2000) -> tuple[float, float]`
  - `detection(runs) -> dict` (keys `caught, missed, false_alarms, f1`)
  - `lead_times(runs) -> pd.Series` (hours, indexed by experiment; NaN = never flagged)
  - `record(**numbers)`, `save_fig(fig, name: str) -> Path`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_eval_harness.py`)

```python
from pathlib import Path

from models.eval import harness as H


class Spy:
    """Approach that records what it was given."""

    n_tuned, kind, name = 0, "baseline", "spy"

    def __init__(self, log):
        self.log, self.threshold = log, 0.0

    def fit(self, z, ends, lab):
        self.log.append(("fit", set(z.experiment), set(ends.experiment), set(lab.experiment)))
        return self

    def score(self, z, ends):
        self.log.append(("score", set(z.experiment), set(ends.experiment)))
        return ends[["experiment", "run_num", "end_h"]].assign(score=ends.end_h.to_numpy())


def synth_dataset(drop=()):
    m = synth_minutes(drop=drop)
    return H.Dataset(
        minutes=m,
        z=condition_correct(m),
        ends=synth_ends(),
        lab=synth_labels(),
        debris=pd.DataFrame(dict(experiment=np.repeat(GEARS, 4), run_num=np.tile([1, 2, 3, 4], 4), debris_rate=1.0)),
    )


def test_run_folds_never_shows_the_held_out_gear_to_fit():
    log = []
    fr = H.run_folds(lambda: Spy(log), synth_dataset())
    fits = [e for e in log if e[0] == "fit"]
    scores = [e for e in log if e[0] == "score"]
    assert len(fits) == 4
    for (_, zg, eg, lg), (_, sg, seg) in zip(fits, scores):
        (held,) = sg
        assert held not in zg | eg | lg and seg == {held}
    assert set(fr.runs.columns) >= {"experiment", "run_num", "end_h", "score", "threshold", "photo", "spall", "mean_ex"}
    assert fr.runs.mean_ex.notna().sum() == 8  # only Exp-A/B folds carry T2


def test_sign_flip_p_exact_values():
    assert H.sign_flip_p([1, 2, 3, 4]) == 1 / 16 == H.p_floor(4)
    assert H.sign_flip_p([1, 2]) == 1 / 4 == H.p_floor(2)
    assert H.sign_flip_p([0, 0, 0, 0]) == 1.0
    assert H.sign_flip_p([1, -1, 2, 3]) == 3 / 16


def test_bootstrap_range_is_seeded_and_ordered():
    lo, hi = H.bootstrap_range([0.01, 0.02, 0.03, 0.10])
    assert lo <= hi and (lo, hi) == H.bootstrap_range([0.01, 0.02, 0.03, 0.10])


def _runs(scores, photo, spall, threshold=0.5, gear="Exp-X"):
    n = len(scores)
    return pd.DataFrame(
        dict(experiment=gear, run_num=range(1, n + 1), end_h=np.arange(1, n + 1) * 6.0, score=scores,
             threshold=threshold, photo=photo, spall=spall, mean_ex=np.nan)
    )


def test_detection_counts_and_lead_time():
    a = _runs([0.1, 0.9, 0.9, 0.9], [0, 0, 1, 2], [False, False, False, True])  # alarm 12 h early, 1 false alarm
    b = _runs([0.1, 0.1, 0.1, 0.1], [0, 0, 0, 2], [False, False, False, True], gear="Exp-Y")  # never flagged
    c = _runs([0.1, 0.1, 0.1, 0.9], [0, 0, 2, 2], [False, False, True, True], gear="Exp-Z")  # 6 h late
    r = pd.concat([a, b, c])
    d = H.detection(r)
    assert (d["caught"], d["missed"], d["false_alarms"]) == (2, 2, 1)
    assert np.isclose(d["f1"], 4 / 7)
    lt = H.lead_times(r)
    assert lt["Exp-X"] == 12.0 and np.isnan(lt["Exp-Y"]) and lt["Exp-Z"] == -6.0


def test_evaluate_runs_constant_score_gives_nan_spearman_finite_mse():
    r = pd.concat([_runs([1.0] * 4, [0, 0, 1, 2], [False] * 3 + [True], gear=g) for g in ["Exp-X", "Exp-Y"]])
    s, gear = H.evaluate_runs(r, "photo")
    assert np.isnan(s["spearman"]) and np.isfinite(s["mse_pooled"]) and len(gear) == 2


def test_gear_error_uses_pooled_map_so_hours_is_not_perfect():
    # hours rises identically on both gears but damage arrives at different times: pooled map cannot fit both
    a = _runs([6, 12, 18, 24.0], [0, 0, 0, 2], [False] * 3 + [True], gear="Exp-X")
    b = _runs([6, 12, 18, 24.0], [0, 2, 2, 2], [False, True, True, True], gear="Exp-Y")
    s, gear = H.evaluate_runs(pd.concat([a, b]), "photo")
    assert s["mse_per_gear"] == 0.0 and gear.min() > 0


def test_load_dataset_missing_parquet_names_the_recreate_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/flatten_hdf5.py"):
        H.load_dataset(tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'harness'`.

- [ ] **Step 3: Implement `models/eval/harness.py`**

```python
"""Common evaluation protocol: one set of folds, metrics, gear-level uncertainty and detection for every approach.

The independent unit is the gear (4 training gears, 2 labelled). Per-gear numbers are the primary evidence;
bootstrap ranges are descriptive and sign-flip p-values are reported next to their floor 1/2**n_gears.
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
    EYE,
    SPLITS,
    challenge_score,
    condition_correct,
    load_debris,
    load_minutes,
    photo_damage,
    run_ends,
)
from spall_supervised import LABELLED, load_labels  # noqa: E402

SEED = 696
TRAIN = SPLITS["train"]
OUT = Path(__file__).resolve().parent / "out"
RECREATE = "python scripts/phm_data.py small && python scripts/flatten_hdf5.py"


@dataclass
class Dataset:
    minutes: pd.DataFrame  # raw steady minutes (load_minutes), all training gears
    z: pd.DataFrame  # condition-corrected minutes
    ends: pd.DataFrame  # experiment, run_num, end_h
    lab: pd.DataFrame  # run-level label targets for Exp-A/B (load_labels)
    debris: pd.DataFrame  # experiment, run_num, debris_rate


def load_dataset(
    data=Path("data"),
    labels=Path("data/labels/spall_scores.csv"),
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
    d["debris_z"] = np.log(d.debris_rate.clip(lower=1e-3) / d.groupby("experiment").debris_rate.transform("first").clip(lower=1e-3))
    z = ds.z.merge(d[["experiment", "run_num", "debris_z"]], on=["experiment", "run_num"], how="left")
    z.index = ds.z.index
    return replace(ds, z=z.assign(debris_z=z.debris_z.fillna(0.0)))


def folds(gears=TRAIN) -> list[tuple[list[str], str]]:
    return [([g for g in gears if g != h], h) for h in gears]


def spall_flags(ends: pd.DataFrame) -> np.ndarray:
    spall = {(e, r) for (e, _, r), v in EYE.items() if v == 2}
    return np.array([(e, r) in spall for e, r in zip(ends.experiment, ends.run_num)])


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
        parts.append(r[["experiment", "run_num", "end_h", "score"]].assign(threshold=a.threshold))
        models[held] = a
    runs = pd.concat(parts, ignore_index=True).sort_values(["experiment", "run_num"], ignore_index=True)
    runs["photo"] = photo_damage(runs)
    runs["spall"] = spall_flags(runs)
    lab = ds.lab[ds.lab.experiment.isin(LABELLED)][["experiment", "run_num", "mean_ex"]]
    runs = runs.merge(lab, on=["experiment", "run_num"], how="left")
    return FoldResult(runs, models, fit_s, score_s)


def pooled_residuals(score, truth) -> np.ndarray:
    """Residual of each run end under one isotonic map shared by all gears (truth scaled to [0, 1])."""
    score, truth = np.asarray(score, float), np.asarray(truth, float)
    truth = truth / max(truth.max(), 1e-9)
    return IsotonicRegression(out_of_bounds="clip").fit(score, truth).predict(score) - truth


def evaluate_runs(runs: pd.DataFrame, target: str = "photo") -> tuple[dict, pd.Series]:
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


def detection(runs: pd.DataFrame) -> dict:
    """Spall run ends caught/missed; false alarms are flags on run ends with no photo damage at all."""
    flag = runs.score > runs.threshold
    caught = int((flag & runs.spall).sum())
    missed = int((~flag & runs.spall).sum())
    false = int((flag & (runs.photo == 0)).sum())
    denom = 2 * caught + missed + false
    return dict(caught=caught, missed=missed, false_alarms=false, f1=2 * caught / denom if denom else np.nan)


def lead_times(runs: pd.DataFrame) -> pd.Series:
    """Hours from the first flagged run end to the first spall run end, per gear with a spall (NaN = never flagged)."""
    out = {}
    for g, r in runs.groupby("experiment"):
        if not r.spall.any():
            continue
        flagged = r.loc[r.score > r.threshold, "end_h"]
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: all PASS. If `test_detection_counts_and_lead_time` fails on F1, recompute by hand: caught 2 (X run 4, Z run 4), missed 2 (Y run 4, Z run 3), false 1 (X run 2) → F1 = 4/(4+2+1) = 4/7.

- [ ] **Step 5: Commit**

```bash
git add models/eval/harness.py tests/test_eval_harness.py
git commit -m "Add evaluation harness: folds, pooled gear error, sign-flip test, detection and lead time"
```

---

### Task 3: Adapters for every approach

**Files:**
- Create: `models/eval/adapters.py`
- Test: `tests/test_eval_harness.py` (append)

**Interfaces:**
- Consumes: `HealthyModel`, `run_end_scores`, `condition_correct`, `challenge_score`, `photo_damage`, `FEATS`, `TOOTH`, `TAIL_MIN` (spall_hi); `minute_targets`, `RidgeDamage`, `GRID`, `AGGS`, `LABELLED` (spall_supervised); `SEED`, `Dataset` (harness).
- Produces (all have `name: str`, `kind: str`, `n_tuned: int`, `threshold: float`, `fit(z, ends, lab) -> self`, `score(z, ends) -> DataFrame[experiment, run_num, end_h, score]`):
  - `HI(col, tooth=TOOTH, feats=FEATS, tail=TAIL_MIN, monotone="iso")` — also `features: list[str]`, `predict_rows(X: np.ndarray) -> np.ndarray`, `hm: HealthyModel`
  - `IForest(feats=FEATS, tail=TAIL_MIN, monotone="iso", seed=SEED, **params)` — also `features`, `predict_rows`, `m: IsolationForest`
  - `RidgeApproach(feats=TOOTH, alpha=10.0, target="mean_ex", sign_split=True, log=True, positive=True, tail=TAIL_MIN)` — also `features`, `predict_rows`, `design(Z) -> np.ndarray`, `design_names: list[str]`, `m: Ridge`, `background: np.ndarray`
  - `LabelTuned(minutes, nested=True, fixed=None)` — also `selected: dict`, `features`, `predict_rows`, `hm`
  - `Hours()`, `Debris(debris: pd.DataFrame)`
  - `registry(ds: Dataset) -> dict[str, Callable[[], approach]]` with keys `hi_maha, hi_spe, hi_zpos, hi_zmax, hi_tooth, hi_tooth_maha, iforest, ridge_tooth, ridge_all, tuned_nonnested, tuned_nested, hours, debris`

- [ ] **Step 1: Write the failing tests** (append)

```python
from models.eval import adapters as A
from spall_supervised import RidgeDamage


def test_every_adapter_returns_one_row_per_run_end_even_without_minutes():
    ds = synth_dataset(drop={("Exp-A", 2)})  # mimic the nulled Exp-A Run-2
    reg = A.registry(ds)
    for name, make in reg.items():
        if name.startswith("tuned"):
            continue  # needs real-scale data; covered by the slow test
        fr = H.run_folds(make, ds)
        a2 = fr.runs[(fr.runs.experiment == "Exp-A")]
        assert len(fr.runs) == 16 and fr.runs.score.notna().all(), name
        assert a2.set_index("run_num").score[2] >= a2.set_index("run_num").score[1] - 1e-12, name


def test_iforest_is_deterministic_for_a_fixed_seed():
    ds = synth_dataset()
    a = H.run_folds(lambda: A.IForest(), ds).runs.score
    b = H.run_folds(lambda: A.IForest(), ds).runs.score
    assert np.array_equal(a, b)


def test_ridge_fits_with_one_labelled_gear_and_refuses_none():
    ds = synth_dataset()
    z, e, lab = ds.z, ds.ends, ds.lab
    one = A.RidgeApproach().fit(z[z.experiment != "Exp-A"], e[e.experiment != "Exp-A"], lab[lab.experiment != "Exp-A"])
    assert np.isfinite(one.threshold)
    with pytest.raises(ValueError, match="no labelled gear"):
        A.RidgeApproach().fit(z[z.experiment.isin(["Exp-C", "Exp-F"])], e, lab.iloc[:0])


def test_ridge_default_design_matches_existing_model():
    z = synth_dataset().z
    r = A.RidgeApproach(TOOTH)
    assert np.allclose(r.design(z[TOOTH].to_numpy()), RidgeDamage(TOOTH).design(z))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v -k "adapter or iforest or ridge"`
Expected: FAIL with `ImportError: cannot import name 'adapters'`.

- [ ] **Step 3: Implement `models/eval/adapters.py`**

```python
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
from spall_hi import FEATS, TAIL_MIN, TOOTH, HealthyModel, challenge_score, condition_correct, photo_damage, run_end_scores
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

    def __init__(self, feats=TOOTH, alpha=10.0, target="mean_ex", sign_split=True, log=True, positive=True, tail=TAIL_MIN):
        self.features, self.alpha, self.target = list(feats), alpha, target
        self.sign_split, self.log, self.positive, self.tail = sign_split, log, positive, tail
        self.name = "ridge_tooth" if feats == TOOTH else "ridge_all"
        self.design_names = (
            [f"{c}+" for c in feats] + [f"{c}-" for c in feats] if sign_split else list(feats)
        )

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
        self.fixed = fixed or json.loads(TUNED_CONFIG.read_text())

    @staticmethod
    def _setting(d):
        return dict(tooth=["ci_" + c for c in d["tooth"].split("/")], form=d["form"], base_h=float(d["base_h"]),
                    tail=int(d["tail"]), monotone=d["monotone"])

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
                            rows.append(dict(
                                tooth="/".join(c[3:] for c in tooth), form=form, base_h=base_h, tail=tail, monotone=mono,
                                AB=np.mean([challenge_score(ab[form], ab[k], ab.experiment.to_numpy())["mse_pooled"] for k in AGGS]),
                                CF=challenge_score(cf[form], photo_damage(cf))["mse_pooled"],
                            ))
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
        # tooth-only rows: HealthyModel.score needs FEATS columns, so pad the others with 0 (they do not enter tooth forms)
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
        self.threshold = float(ends.loc[ends.run_num == ends.groupby("experiment").run_num.transform("min"), "end_h"].max())
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests -v -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Smoke-run the nested selector on real data**

Run:
```bash
.venv/bin/python -c "
import models.eval
from models.eval import harness as H, adapters as A
ds = H.load_dataset()
fr = H.run_folds(lambda: A.LabelTuned(ds.minutes, nested=True), ds)
print({g: m.selected['tooth']+'/'+m.selected['form']+'/'+str(m.selected['tail']) for g, m in fr.models.items()})
print(H.evaluate_runs(fr.runs)[0])
"
```
Expected: four selected settings printed (one per held-out gear) and a metrics dict; runs in under ~2 minutes.

- [ ] **Step 6: Commit**

```bash
git add models/eval/adapters.py tests/test_eval_harness.py
git commit -m "Add adapters: HI candidates, Isolation Forest, ridge, label-tuned (nested and not), hours, debris"
```

---

### Task 4: Step 2 comparison (`compare.py`)

**Files:**
- Create: `models/eval/compare.py`

**Interfaces:**
- Consumes: `registry`; `run_folds`, `evaluate_runs`, `sign_flip_p`, `p_floor`, `bootstrap_range`, `detection`, `lead_times`, `record`, `save_fig`, `pooled_residuals`, `load_dataset`, `OUT`.
- Produces: `main(ds: Dataset | None = None) -> pd.DataFrame` (the comparison table); writes `out/runs.csv` (columns: `approach, kind, experiment, run_num, end_h, score, threshold, photo, spall, mean_ex`), `out/comparison.csv`, `out/gear_errors.csv`, `out/lead_time.csv`, `fig_per_gear_dots.png`, `fig_trajectories.png`, `fig_lead_time.png`; numbers `winner`, `winner_k_of_4`, `winner_mse_pooled_t1`, `selection_bias_gap_t1`, `p_floor_4`, `p_floor_2`.
- `rank_approaches(table: pd.DataFrame) -> list[str]` (winner rule, non-baselines only).

- [ ] **Step 1: Write the test** (append to `tests/test_eval_harness.py`)

```python
def test_winner_rule_order():
    from models.eval.compare import rank_approaches

    t = pd.DataFrame(
        dict(kind=["unsupervised", "supervised", "baseline", "semi-supervised"], k_beats_hours=[3, 3, 4, 2],
             t1_mse_pooled=[0.02, 0.01, 0.0, 0.001], n_tuned=[0, 1, 0, 96]),
        index=["a", "b", "hours", "c"],
    )
    assert rank_approaches(t) == ["b", "a", "c"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py::test_winner_rule_order -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'models.eval.compare'`.

- [ ] **Step 3: Implement `models/eval/compare.py`**

```python
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
    """Winner rule fixed in the spec: most gears beating hours, then lowest T1 mse_pooled, then fewest tuned settings."""
    t = table[table.kind != "baseline"]
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
            approach=name, kind=a.kind, n_tuned=a.n_tuned,
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
        for name in top:
            r = runs[runs.approach == name]
            fitted = pooled_residuals(r.score, r.photo) + r.photo / max(r.photo.max(), 1e-9)
            k = (r.experiment == g).to_numpy()
            ax.step(r.end_h[k], fitted[k], where="post", label=name)
        t = runs[(runs.approach == top[0]) & (runs.experiment == g)]
        ax.step(t.end_h, t.photo / runs.photo.max(), where="post", color="k", ls="--", label="photo truth")
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
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lead time (h, + = early)")
    ax.set_title("Hours between first alarm and first spall photo (missing bar = never flagged)")
    save_fig(fig, "fig_lead_time.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the script on real data**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py::test_winner_rule_order -v` → PASS
Run: `.venv/bin/python -m models.eval.compare`
Expected: prints a 13-row table and a winner order; `models/eval/out/` holds `comparison.csv`, `gear_errors.csv`, `lead_time.csv`, `runs.csv`, three PNGs, `report_numbers.json`. Sanity checks: the `hours` row has `t1_mse_per_gear == 0` (degenerate metric, as the spec predicts) and `k_beats_hours == 0`; `tuned_nonnested.t1_mse_pooled <= tuned_nested.t1_mse_pooled` is expected (selection bias) — if not, note it in the write-up rather than changing code.

- [ ] **Step 5: Open the three PNGs and check them by eye**

Each has labelled axes, a title, a legend, and no overlapping tick labels. Fix sizes in code if not.

- [ ] **Step 6: Commit**

```bash
git add models/eval/compare.py tests/test_eval_harness.py models/eval/out
git commit -m "Add Step 2 comparison: master table, per-gear dots, trajectories, lead time"
```

---

### Task 5: Ablation (`ablation.py`)

**Files:**
- Create: `models/eval/ablation.py`

**Interfaces:**
- Consumes: `HI`, `IForest`, `RidgeApproach`, `load_dataset`, `with_debris_feature`, `run_folds`, `evaluate_runs`, `save_fig`, `record`, `OUT`; `FEATS`, `TOOTH`.
- Produces: `main(ds=None) -> pd.DataFrame`; writes `out/ablation.csv` (columns `approach, ablation, t1_mse_pooled, t2_mse_pooled, d_t1, d_t2`), `out/fig_ablation_heatmap.png`; numbers `ablation_largest_t1_gain` (str), `ablation_largest_t1_loss` (str).

- [ ] **Step 1: Implement**

```python
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
```

- [ ] **Step 2: Run on real data**

Run: `.venv/bin/python -m models.eval.ablation`
Expected: about 50 printed rows, `out/ablation.csv`, `out/fig_ablation_heatmap.png`. The `full` rows' `t1_mse_pooled` for `hi_tooth`, `hi_maha`, `iforest` equal the same approaches in `out/comparison.csv` (same folds, same data) — check two of them by eye; a mismatch means the ablation used a different dataset and is a bug.

- [ ] **Step 3: Check the heatmap by eye** (readable labels, colour bar, title). Commit.

```bash
git add models/eval/ablation.py models/eval/out
git commit -m "Add Step 1.1 ablation: features, pipeline steps, ridge internals"
```

---

### Task 6: Hyperparameter sensitivity (`sensitivity.py`)

**Files:**
- Create: `models/eval/sensitivity.py`

**Interfaces:**
- Consumes: `HI`, `IForest`, `RidgeApproach`, `LabelTuned`, `load_dataset`, `run_folds`, `evaluate_runs`, `save_fig`, `record`, `OUT`, `SEED`; `FEATS`, `TOOTH`.
- Produces: `main(ds=None) -> pd.DataFrame`; writes `out/sensitivity.csv` (columns `approach, param, value, gear, gear_err, t1_mse_pooled, t2_mse_pooled, coef_norm`), `out/sensitivity_tuned_winners.csv`, `out/fig_sensitivity.png`; numbers `ridge_alpha_flat_range` (str), `iforest_seed_sd_t1` (float), `tuned_distinct_winners` (int).

- [ ] **Step 1: Implement**

```python
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
            ax.plot(dg.value.astype(str), dg.gear_err, marker="o", label=g)
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
```

- [ ] **Step 2: Run on real data**

Run: `.venv/bin/python -m models.eval.sensitivity`
Expected: `out/sensitivity.csv`, `out/sensitivity_tuned_winners.csv`, `out/fig_sensitivity.png`. The ridge `coef_norm` column explains the flat alpha region: identical norms across small alphas mean the penalty is negligible next to the data term (thousands of labelled minutes); norms of 0 mean the positivity constraint zeroes the fit. Note which one holds; it goes in the report's sensitivity paragraph.

- [ ] **Step 3: Check the figure by eye; commit**

```bash
git add models/eval/sensitivity.py models/eval/out
git commit -m "Add Step 1.2 hyperparameter sensitivity, ridge coefficient norms and nested winner stability"
```

---

### Task 7: Learning curves (`curves.py`)

**Files:**
- Create: `models/eval/curves.py`

**Interfaces:**
- Consumes: `HI`, `IForest`, `RidgeApproach`, `load_dataset`, `run_folds`, `evaluate_runs`, `pooled_residuals`, `save_fig`, `record`, `OUT`, `TRAIN`; `challenge_score`, `photo_damage` (spall_hi); `LABELLED` (spall_supervised).
- Produces: `main(ds=None)`; writes `out/curves_gears.csv` (`approach, k, held, combo, err`), `out/curves_baseline_hours.csv` (`approach, base_h, t1_mse_pooled`), `out/curves_ridge.csv` (`train_gear, test_gear, k, t2_mse, spearman`), `out/fig_learning_curves.png`; numbers `curve_gain_2_to_3_gears_hi_tooth` (float).

- [ ] **Step 1: Implement**

```python
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

APPROACHES = {"hi_tooth": lambda: HI("tooth"), "hi_maha": lambda: HI("maha"), "iforest": lambda: IForest()}


def gear_curve(ds) -> pd.DataFrame:
    """Healthy model fit on every 1-, 2- and 3-gear subset of the non-held-out gears; one pooled map per k."""
    rows = []
    for name, make in APPROACHES.items():
        preds = []
        for held in TRAIN:
            others = [g for g in TRAIN if g != held]
            for k in (1, 2, 3):
                for combo in itertools.combinations(others, k):
                    a = make().fit(ds.z[ds.z.experiment.isin(combo)], ds.ends[ds.ends.experiment.isin(combo)], ds.lab.iloc[:0])
                    r = a.score(ds.z[ds.z.experiment == held], ds.ends[ds.ends.experiment == held])
                    preds.append(r.assign(k=k, held=held, combo="+".join(c[-1] for c in combo)))
        p = pd.concat(preds, ignore_index=True)
        p["photo"] = photo_damage(p)
        for k, pk in p.groupby("k"):
            pk = pk.assign(sq=pooled_residuals(pk.score, pk.photo) ** 2)
            rows += [dict(approach=name, k=k, held=h, combo=c, err=e) for (h, c), e in pk.groupby(["held", "combo"]).sq.mean().items()]
    return pd.DataFrame(rows)


def baseline_hours_curve(ds) -> pd.DataFrame:
    rows = []
    for base_h in [1.0, 2.0, 3.0, 4.5, 6.0]:
        d = load_dataset(minutes=ds.minutes, base_h=base_h)
        for name, make in APPROACHES.items():
            rows.append(dict(approach=name, base_h=base_h, t1_mse_pooled=evaluate_runs(run_folds(make, d).runs)[0]["mse_pooled"]))
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
        axes[0].errorbar(m.index, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]], marker="o", capsize=3, label=name)
    axes[0].set(xlabel="training gears for the healthy model", ylabel="gear error (T1)", xticks=[1, 2, 3], title="Gears")
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

    m = g[g.approach == "hi_tooth"].groupby("k").err.mean()
    record(curve_gain_2_to_3_gears_hi_tooth=float(m[2] - m[3]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on real data**

Run: `.venv/bin/python -m models.eval.curves`
Expected: three CSVs and `fig_learning_curves.png`. `curves_gears.csv` has, per approach, 12 rows for k=1 (4 held × 3), 12 for k=2, 4 for k=3.

- [ ] **Step 3: Check the figure by eye; commit**

```bash
git add models/eval/curves.py models/eval/out
git commit -m "Add Step 1.3 learning curves over gears, healthy hours and labelled inspections"
```

---

### Task 8: Failure analysis (`failures.py`)

**Files:**
- Create: `models/eval/failures.py`
- Test: `tests/test_eval_harness.py` (append)

**Interfaces:**
- Consumes: `out/runs.csv` (Task 4); `registry`, `HI`, `IForest`, `RidgeApproach`, `LabelTuned`; `folds`, `load_dataset`, `pooled_residuals`, `save_fig`, `record`, `OUT`, `SEED`; `TAIL_MIN` (spall_hi).
- Produces:
  - `candidates(runs: pd.DataFrame) -> pd.DataFrame` (columns `approach, kind, target, experiment, run_num, end_h, score, truth, fitted, residual, rank`; `rank` 1 = largest |residual| within approach and target)
  - `HYPOTHESES: list[dict]` (keys `approach, experiment, target, category, why`)
  - `select_examples(cands, max_rank=5) -> pd.DataFrame` (one row per hypothesis: worst run end for that approach/gear, its rank, `confirmed = rank <= max_rank`)
  - `exact_attribution(a, x: np.ndarray) -> pd.Series | None`
  - `main(ds=None)`; writes `out/failure_candidates.csv`, `out/failure_examples.csv`, `out/fig_failure_<n>.png`; numbers `n_failure_examples`, `n_failure_categories`, `lime_agreement_median`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_candidates_rank_by_absolute_residual_within_approach():
    from models.eval.failures import candidates

    # pooled map at score 1 averages truths (1, 0, 0, 0) -> 0.25, so Exp-X run 4 misses by 0.75, Exp-Y runs by 0.25
    r = pd.concat([
        _runs([0.0, 0.0, 0.0, 1.0], [0, 0, 0, 2], [False] * 3 + [True], gear="Exp-X"),
        _runs([0.0, 1.0, 1.0, 1.0], [0, 0, 0, 0], [False] * 4, gear="Exp-Y"),
    ]).assign(approach="m", kind="unsupervised")
    c = candidates(r)
    top = c[(c.approach == "m") & (c.target == "photo")].sort_values("rank").iloc[0]
    assert (top.experiment, top.run_num, top["rank"]) == ("Exp-X", 4, 1) and np.isclose(top.residual, -0.75)


def test_exact_attribution_sums_to_the_score():
    from models.eval.failures import exact_attribution

    ds = synth_dataset()
    x = ds.z[FEATS].to_numpy()[-1]
    for col in ["tooth", "maha", "tooth_maha"]:
        a = A.HI(col).fit(ds.z, ds.ends, ds.lab)
        att = exact_attribution(a, x[[a.features.index(f) for f in a.features]])
        target = a.predict_rows(x[None])[0]
        if col == "tooth":
            assert np.isclose(att.sum(), target)
        elif col == "maha":
            assert np.isclose(att.sum(), target**2)  # shares of squared distance
        else:
            assert np.isclose(att.sum(), target**2) or target == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v -k "candidates or attribution"`
Expected: FAIL, `ModuleNotFoundError: No module named 'models.eval.failures'`.

- [ ] **Step 3: Implement `models/eval/failures.py`**

```python
"""Step 1.4: rank failures, pick examples by a fixed rule, and explain them (exact, SHAP, LIME).

Usage (repo root):  .venv/bin/python -m models.eval.failures   (needs out/runs.csv from models.eval.compare)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from spall_hi import TAIL_MIN, TOOTH

from .adapters import HI, IForest, LabelTuned, RidgeApproach, registry
from .harness import OUT, SEED, folds, load_dataset, pooled_residuals, record, save_fig

# Hypotheses from the design, each tied to a category; select_examples confirms or rejects them against the data.
HYPOTHESES = [
    dict(approach="hi_maha", experiment="Exp-A", target="photo", category="systematic",
         why="all-CI distance jumps after rig restarts while the photo shows no damage"),
    dict(approach="ridge_all", experiment="Exp-B", target="mean_ex", category="systematic",
         why="all-CI ridge learns Exp-B's restart level shifts as wear (overfitting with 2 labelled gears)"),
    dict(approach="tuned_nested", experiment="Exp-C", target="photo", category="edge case",
         why="Exp-C runs twice as long, damages only at run 12, and its photos come from two camera setups"),
    dict(approach="iforest", experiment="Exp-A", target="photo", category="operational",
         why="Exp-A Run-2 is a byte copy of Run-1 and is nulled; its value is carried forward from Run-1"),
    dict(approach="ridge_tooth", experiment="Exp-A", target="mean_ex", category="edge case",
         why="Exp-A's late spalls vs Exp-B's gradual wear: trained on one damage regime, tested on the other"),
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
            out.append(x[["approach", "kind", "target", "experiment", "run_num", "end_h", "score", "truth", "fitted", "residual", "rank"]])
    return pd.concat(out, ignore_index=True)


def select_examples(cands: pd.DataFrame, max_rank: int = 5) -> pd.DataFrame:
    rows = []
    for h in HYPOTHESES:
        c = cands[(cands.approach == h["approach"]) & (cands.experiment == h["experiment"]) & (cands.target == h["target"])]
        top = c.sort_values("rank").iloc[0]
        rows.append({**h, **top[["run_num", "end_h", "score", "truth", "fitted", "residual", "rank"]].to_dict(),
                     "confirmed": bool(top["rank"] <= max_rank)})
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
    ex = LimeTabularExplainer(background, feature_names=a.features, mode="regression", discretize_continuous=False,
                              random_state=SEED)
    e = ex.explain_instance(x, a.predict_rows, num_features=len(a.features), num_samples=2000)
    return pd.Series(dict(e.as_list())).reindex(a.features).fillna(0.0)


def _instance(a, ds, gear, run_num):
    """Median feature vector over the run's last TAIL_MIN steady minutes (the run-end state the score summarises)."""
    z = ds.z if not isinstance(a, LabelTuned) else _tuned_z(a, ds)
    r = z[(z.experiment == gear) & (z.run_num == run_num)].sort_values("hours").tail(TAIL_MIN)
    return r[a.features].median().to_numpy(), z


def _tuned_z(a, ds):
    from spall_hi import condition_correct

    return condition_correct(ds.minutes, a.st["base_h"])


def explain(ex_row, ds, n):
    reg = registry(ds)
    train, held = next((t, h) for t, h in folds() if h == ex_row.experiment)
    a = reg[ex_row.approach]().fit(ds.z[ds.z.experiment.isin(train)], ds.ends[ds.ends.experiment.isin(train)],
                                   ds.lab[ds.lab.experiment.isin(train)])
    x, z = _instance(a, ds, held, int(ex_row.run_num))
    if np.isnan(x).any():
        return dict(explained=False, note="no minutes in this run (carried forward from the previous run)")
    background = z.loc[z.experiment.isin(train), a.features].to_numpy()
    ref = exact_attribution(a, x)
    ref = shap_attribution(a, x) if ref is None else ref
    lime = lime_attribution(a, x, background)
    agree = float(spearmanr(ref.reindex(a.features), lime)[0]) if ref is not None else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 2]))
    att = pd.DataFrame({"exact/SHAP": ref, "LIME": lime}) if ref is not None else pd.DataFrame({"LIME": lime})
    att.reindex(a.features).plot.barh(ax=ax1)
    ax1.set_title(f"Attribution, {ex_row.approach} {held} run {int(ex_row.run_num)}", fontsize=9)
    zh = z[z.experiment == held].sort_values("hours")
    ax2.plot(zh.hours, a.predict_rows(zh[a.features].to_numpy()), lw=0.6, label="minute score")
    ax2.axhline(a.threshold, color="r", ls=":", label="alarm threshold")
    for h in ds.ends.loc[ds.ends.experiment == held, "end_h"]:
        ax2.axvline(h, color="grey", lw=0.5)
    mh = ds.minutes[ds.minutes.experiment == held]
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
    ex.round(5).to_csv(OUT / "failure_examples.csv", index=False)
    conf = ex[ex.confirmed]
    record(n_failure_examples=int(len(conf)), n_failure_categories=int(conf.category.nunique()),
           lime_agreement_median=float(ex.get("lime_agreement", pd.Series(dtype=float)).median()))
    print(ex[["approach", "experiment", "run_num", "category", "rank", "confirmed", "top_exact_or_shap", "top_lime", "lime_agreement"]].to_string())
    return ex


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests -v -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Run on real data and apply the selection rule**

Run: `.venv/bin/python -m models.eval.failures`
Expected: `failure_candidates.csv`, `failure_examples.csv`, `fig_failure_1..5.png`.

Then apply the spec §6.5 rule: the confirmed examples must number ≥ 3, include ≥ 1 supervised (`ridge_*`) and ≥ 1 unsupervised (`hi_*`/`iforest`), and span ≥ 3 categories. If they do not:
1. Open `failure_candidates.csv`, sort by `rank`, and take the highest-ranked run end of a missing kind or category that no confirmed example already covers.
2. Add it to `HYPOTHESES` with its approach, experiment, target, category and a one-line `why` grounded in what the time plot shows.
3. Re-run until the rule holds. Record each rejected hypothesis (with its rank) for the write-up — a rejected hypothesis is evidence too.

- [ ] **Step 6: Check each failure figure by eye; commit**

```bash
git add models/eval/failures.py tests/test_eval_harness.py models/eval/out
git commit -m "Add Step 1.4 failure analysis: residual ranking, rule-based selection, exact/SHAP/LIME attributions"
```

---

### Task 9: `run_all.py`, notebook, README, slow test

**Files:**
- Create: `models/eval/run_all.py`, `models/eval/cells.py`, `models/eval/build_notebook.py`, `models/eval/spall_eval.ipynb` (generated)
- Modify: `README.md` (new section), `tests/test_eval_harness.py` (slow test)

**Interfaces:**
- Consumes: `main(ds)` of `compare`, `ablation`, `sensitivity`, `curves`, `failures`; `load_dataset`, `OUT`.
- Produces: `run_all.main()`; `cells.CELLS: list[tuple[str, str]]` (`("md" | "code", source)`); notebook file.

- [ ] **Step 1: Write the slow test** (append)

```python
EXPECTED = [
    "comparison.csv", "gear_errors.csv", "lead_time.csv", "runs.csv", "ablation.csv", "sensitivity.csv",
    "sensitivity_tuned_winners.csv", "curves_gears.csv", "curves_baseline_hours.csv", "curves_ridge.csv",
    "failure_candidates.csv", "failure_examples.csv", "report_numbers.json", "fig_per_gear_dots.png",
    "fig_trajectories.png", "fig_lead_time.png", "fig_ablation_heatmap.png", "fig_sensitivity.png",
    "fig_learning_curves.png", "fig_failure_1.png",
]


@pytest.mark.slow
def test_run_all_writes_every_output():
    if not Path("data/train/lf_1s.parquet").exists():
        pytest.skip("parquet tables not downloaded")
    from models.eval import run_all

    run_all.main()
    missing = [f for f in EXPECTED if not (H.OUT / f).exists()]
    assert not missing, missing
```

Register the marker so pytest does not warn: create `pytest.ini` at the repo root if absent:

```ini
[pytest]
markers =
    slow: needs the real parquet tables; takes minutes
```

- [ ] **Step 2: Implement `run_all.py`**

```python
"""Regenerate every evaluation output in models/eval/out/ from the parquet tables.

Usage (repo root):  .venv/bin/python -m models.eval.run_all
"""

import time

from . import ablation, compare, curves, failures, sensitivity
from .harness import OUT, load_dataset


def main():
    t0 = time.perf_counter()
    (OUT / "report_numbers.json").unlink(missing_ok=True)
    ds = load_dataset()
    for mod in [compare, ablation, sensitivity, curves, failures]:  # failures reads compare's runs.csv
        print(f"\n=== {mod.__name__} ===")
        mod.main(ds)
    print(f"\ndone in {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement `cells.py` and `build_notebook.py`**

`models/eval/cells.py`:

```python
"""Notebook cell sources for spall_eval.ipynb: one section per section of the PDF report. Reads out/ only."""

SETUP = r'''# Setup: read the evaluation outputs (regenerate with: python -m models.eval.run_all)
import json
from pathlib import Path
import pandas as pd
try:
    from IPython.display import Image, display
except ImportError:  # plain python (used by the headless check)
    def Image(filename, width=None): return filename
    display = print

OUT = Path("models/eval/out") if Path("models/eval/out").exists() else Path("out")
N = json.loads((OUT / "report_numbers.json").read_text())
pd.set_option("display.width", 200)
def fig(name): display(Image(filename=str(OUT / name), width=900))
def table(name, **kw): return pd.read_csv(OUT / name, **kw)
'''

CELLS = [
    ("md", "# Spall models: evaluation and failure analysis\n\nEvery number and figure here is read from "
           "`models/eval/out/`, the same files the PDF report uses. Run **Setup** first."),
    ("code", SETUP),
    ("md", "## 1. Evaluation method\n\nLeave-one-gear-out over Exp-A, B, C, F. T1 = photo damage (all 4 gears); "
           "T2 = tooth labels (`mean_ex`, Exp-A/B folds only). Gear error = squared residual under one pooled "
           "isotonic map, averaged within gear. p-values are exact sign-flip tests over gears; with 4 gears the "
           "smallest possible p is 1/16."),
    ("code", 'print("p floor, 4 gears:", N["p_floor_4"], "| 2 gears:", N["p_floor_2"])'),
    ("md", "## 2. Step 1.1 — Ablation"),
    ("code", 'fig("fig_ablation_heatmap.png")\nprint(N["ablation_largest_t1_gain"]); print(N["ablation_largest_t1_loss"])\n'
             'table("ablation.csv").sort_values("d_t1")'),
    ("md", "## 3. Step 1.2 — Hyperparameter sensitivity"),
    ("code", 'fig("fig_sensitivity.png")\nprint("ridge alpha flat over:", N["ridge_alpha_flat_range"])\n'
             'print("Isolation Forest seed SD (T1):", N["iforest_seed_sd_t1"])\n'
             'print("distinct nested winners across folds:", N["tuned_distinct_winners"])\n'
             'table("sensitivity_tuned_winners.csv")'),
    ("md", "## 4. Step 1.3 — Learning curves"),
    ("code", 'fig("fig_learning_curves.png")\nprint("hi_tooth error drop from 2 to 3 gears:", N["curve_gain_2_to_3_gears_hi_tooth"])'),
    ("md", "## 5. Step 2 — Comparison"),
    ("code", 'cmp = table("comparison.csv", index_col=0)\ncmp.round(4)'),
    ("code", 'fig("fig_per_gear_dots.png"); fig("fig_trajectories.png"); fig("fig_lead_time.png")\n'
             'print("winner:", N["winner"], "| beats hours on", N["winner_k_of_4"], "of 4 gears, p =", '
             'N["winner_p_vs_hours"], "(floor", N["p_floor_4"], ")")\n'
             'print("selection bias, nested - non-nested T1 MSE:", N["selection_bias_gap_t1"])\n'
             'print("ranking changes on checked-only inspections:", N["ranking_changes_on_checked_only"])'),
    ("md", "## 6. Failure examples"),
    ("code", 'ex = table("failure_examples.csv")\nex[["approach","experiment","run_num","category","rank","confirmed",'
             '"why","top_exact_or_shap","top_lime","lime_agreement"]]'),
    ("code", 'for i in range(1, len(ex) + 1):\n    if (OUT / f"fig_failure_{i}.png").exists():\n        fig(f"fig_failure_{i}.png")'),
    ("md", "## 7. Failure candidates (full ranked list, for auditing the selection)"),
    ("code", 'table("failure_candidates.csv").sort_values(["approach", "target", "rank"]).groupby(["approach", "target"]).head(5)'),
    ("md", "## 8. Regenerate everything (optional, needs the parquet tables; several minutes)"),
    ("code", '# import models.eval.run_all as r; r.main()'),
]
```

`models/eval/build_notebook.py`:

```python
"""Build models/eval/spall_eval.ipynb from cells.py.

Usage (repo root):  .venv/bin/python -m models.eval.build_notebook
"""

from pathlib import Path

import nbformat as nbf

from .cells import CELLS


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s) for kind, s in CELLS]
    p = Path(__file__).resolve().parent / "spall_eval.ipynb"
    nbf.write(nb, p)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run everything**

Run: `.venv/bin/python -m models.eval.run_all && .venv/bin/python -m models.eval.build_notebook`
Expected: all outputs regenerate; notebook written.

Run: `.venv/bin/python -m pytest tests -v` (includes the slow test)
Expected: all PASS.

Execute every code cell headless (no Jupyter dependency needed; image cells print file paths instead):
`.venv/bin/python -c "import json; nb=json.load(open('models/eval/spall_eval.ipynb')); ns={}; [exec(''.join(c['source']), ns) for c in nb['cells'] if c['cell_type']=='code']; print('ok')"`
Expected: the figure paths print, then `ok`.

- [ ] **Step 5: Add a README section** after "Using the Exp-A / Exp-B spall labels":

````markdown
## Evaluation and failure analysis

`models/eval/` evaluates every approach (health-index candidates, Isolation Forest, ridge, label-tuned HI nested and
not, elapsed hours, debris) with one leave-one-gear-out protocol and writes all figures, tables and quoted numbers
for the report to `models/eval/out/`. Design: `docs/superpowers/specs/2026-10-09-model-evaluation-design.md`.

```bash
.venv/bin/python -m models.eval.run_all          # regenerate models/eval/out/ (several minutes)
.venv/bin/python -m models.eval.build_notebook   # rebuild models/eval/spall_eval.ipynb
.venv/bin/python -m pytest tests -m "not slow"   # fast tests
```
````

- [ ] **Step 6: Lint and commit**

Run: `ruff check models/eval tests && ruff format --check models/eval tests`
Expected: no errors (run `ruff format models/eval tests` if formatting differs, then re-run tests).

```bash
git add models/eval tests/test_eval_harness.py pytest.ini README.md
git commit -m "Add run_all, evaluation notebook, README section and end-to-end test"
```

---

### Task 10: Report-ready check

**Files:** none created; verification only.

- [ ] **Step 1: Check every rubric item has its evidence in `out/`**

| Rubric item | Evidence file |
|---|---|
| Examples (2) | `failure_examples.csv` (≥3 confirmed, both kinds), `fig_failure_*.png` |
| Reasons (2) | attributions + agreement in `failure_examples.csv`, ablation/sensitivity/curves figures |
| Categories (1) | `failure_examples.csv` `category` column with ≥3 distinct values |
| Improvements (1) | written in the PDF from the causes found; not code |
| Ethics, reproducibility appendix | written in the PDF; versions from `.venv/bin/pip freeze`, `SEED`, parameters from `out/` CSVs |

Run: `.venv/bin/python -c "import json; n=json.load(open('models/eval/out/report_numbers.json')); assert n['n_failure_examples']>=3 and n['n_failure_categories']>=3, n; print(n)"`
Expected: prints the numbers dict with no assertion error.

- [ ] **Step 2: Confirm the shipped submissions are untouched**

Run: `git diff --stat main -- models/out`
Expected: empty.

- [ ] **Step 3: Final commit of any regenerated outputs**

```bash
git add models/eval/out
git commit -m "Regenerate evaluation outputs" || echo "nothing to commit"
```
