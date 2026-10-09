from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models.eval  # noqa: F401  (puts models/ on sys.path)
from models.eval import adapters as A
from models.eval import harness as H

# isort: split
from spall_hi import BASE_H, BREAK_IN_H, COND, FEATS, TOOTH, HealthyModel, condition_correct
from spall_supervised import RidgeDamage

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
    return pd.DataFrame([dict(experiment=g, run_num=r, end_h=r * 6.0) for g in gears for r in range(1, runs + 1)])


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
        dict(
            experiment=gear,
            run_num=range(1, n + 1),
            end_h=np.arange(1, n + 1) * 6.0,
            score=scores,
            threshold=threshold,
            photo=photo,
            spall=spall,
            mean_ex=np.nan,
        )
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


def test_winner_rule_order():
    from models.eval.compare import rank_approaches

    t = pd.DataFrame(
        dict(
            kind=["unsupervised", "supervised", "baseline", "semi-supervised"],
            k_beats_hours=[3, 3, 4, 2],
            t1_mse_pooled=[0.02, 0.01, 0.0, 0.001],
            n_tuned=[0, 1, 0, 96],
        ),
        index=["a", "b", "hours", "c"],
    )
    assert rank_approaches(t) == ["b", "a", "c"]


def test_winner_rule_excludes_in_sample_rows():
    from models.eval.compare import rank_approaches

    t = pd.DataFrame(
        dict(
            kind=["semi-supervised", "semi-supervised"],
            k_beats_hours=[4, 4],
            t1_mse_pooled=[0.01, 0.02],
            n_tuned=[96, 96],
            in_sample=[True, False],
        ),
        index=["tuned_nonnested", "tuned_nested"],
    )
    assert rank_approaches(t) == ["tuned_nested"]


def test_candidates_rank_by_absolute_residual_within_approach():
    from models.eval.failures import candidates

    # pooled map at score 1 averages truths (1, 0, 0, 0) -> 0.25, so Exp-X run 4 misses by 0.75, Exp-Y runs by 0.25
    r = pd.concat(
        [
            _runs([0.0, 0.0, 0.0, 1.0], [0, 0, 0, 2], [False] * 3 + [True], gear="Exp-X"),
            _runs([0.0, 1.0, 1.0, 1.0], [0, 0, 0, 0], [False] * 4, gear="Exp-Y"),
        ]
    ).assign(approach="m", kind="unsupervised")
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


def test_lime_contributions_match_shap_on_a_linear_model():
    from models.eval.failures import lime_attribution, shap_attribution

    ds = synth_dataset()
    a = A.RidgeApproach(FEATS, sign_split=False, log=False, positive=False).fit(ds.z, ds.ends, ds.lab)
    bg = ds.z[FEATS].to_numpy()
    # mixed-sign deviations from the mean: a slope and a contribution then disagree in sign for some features
    x = bg.mean(0) + np.random.default_rng(1).normal(0, 3, len(FEATS)) * bg.std(0)
    lime, shap_ = lime_attribution(a, x, bg), shap_attribution(a, x)
    assert np.corrcoef(lime.to_numpy(), shap_.reindex(lime.index).to_numpy())[0, 1] > 0.99


def test_seed_residual_sd_is_zero_for_deterministic_and_positive_for_iforest():
    from models.eval.failures import seed_residual_sd

    ds = synth_dataset()
    det = seed_residual_sd(ds, lambda seed: A.HI("tooth"), "Exp-A", 4, seeds=range(3))
    rnd = seed_residual_sd(ds, lambda seed: A.IForest(seed=seed, n_estimators=10), "Exp-A", 4, seeds=range(3))
    assert det == 0.0 and rnd > 0.0


def test_agreement_uses_features_both_methods_cover():
    from models.eval.failures import agreement

    ref = pd.Series({"a": 3.0, "b": -1.0, "c": 2.0})
    lime = pd.Series({"a": 0.3, "b": -0.2, "c": 0.4, "d": 9.0, "e": -9.0})
    assert np.isclose(agreement(ref, lime), 0.5)
    assert np.isnan(agreement(None, lime)) and np.isnan(agreement(ref[["a"]], lime))


EXPECTED = [
    "comparison.csv",
    "gear_errors.csv",
    "lead_time.csv",
    "runs.csv",
    "ablation.csv",
    "sensitivity.csv",
    "sensitivity_tuned_winners.csv",
    "curves_gears.csv",
    "curves_baseline_hours.csv",
    "curves_ridge.csv",
    "failure_candidates.csv",
    "failure_examples.csv",
    "report_numbers.json",
    "fig_per_gear_dots.png",
    "fig_trajectories.png",
    "fig_lead_time.png",
    "fig_ablation_heatmap.png",
    "fig_sensitivity.png",
    "fig_learning_curves.png",
    "fig_failure_1.png",
]


@pytest.mark.slow
def test_run_all_writes_every_output():
    if not Path("data/train/lf_1s.parquet").exists():
        pytest.skip("parquet tables not downloaded")
    from models.eval import run_all

    run_all.main()
    missing = [f for f in EXPECTED if not (H.OUT / f).exists()]
    assert not missing, missing
