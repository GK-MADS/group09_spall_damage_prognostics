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
    for kw in [dict(correct=False), dict(per_gear=False), dict(break_in_h=1.0)]:
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
