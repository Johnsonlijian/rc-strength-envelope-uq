"""
a29_physics_arms.py — pre-empt the two strongest remaining reviewer attacks.

Attack 1: "You starved the ML of physics. Give the learner the mechanical model
and the extrapolation problem disappears."  ->  Two physics-anchored arms, same
protocol and metric as the manuscript's UQ matrix:
  (a) residual-target learning: the learner fits the ln model-error of a
      validated mechanical model, y* = ln(V_test / V_mech(x)); the prediction is
      V_mech(x) * exp(f_hat(x)).  The physics carries the extrapolation, the
      learner only corrects in-envelope bias.
  (b) physics-feature learning: ln V_mech(x) is appended as an input feature but
      the target remains ln V_test (the common "hybrid" in the shear-ML
      literature).  The learner may still saturate beyond its trained target
      range.
Anchors: MCFT (canonical steel), ACI 440.1R-15 (FRP).

Attack 2: "Covariate-shift-weighted conformal already solves this."  ->
Estimated-likelihood-ratio weighted split conformal (Tibshirani et al. 2019):
a logistic classifier between the training pool and the deployment (large-
member) covariates gives w(x) = p/(1-p); the conformal quantile is the weighted
quantile with the test point's own weight on the +infinity atom.  We report how
often the interval is INFINITE (the method abstains), the coverage, and the
width inflation of the finite intervals.

Outputs -> ../data/processed/physics_arms.csv, weighted_conformal.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from uq_utils import split_conformal_qhat, coverage
from a12_mech_models import mcft
from a08_mechanical_fallback import aci440_Vc_kN

PROC = Path("../data/processed")
ALPHA = 0.10
THRESHOLDS = (0.70, 0.75, 0.80)
SEEDS = (0, 1, 2)


def models(seed):
    return {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=seed),
        "RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
              max_features=0.7, random_state=seed, n_jobs=-1),
    }


def fit_arm(model, Xtr, ln_target_tr, Xcal, ln_y_cal, ln_base_cal,
            Xq_list, ln_base_q_list):
    """Fit on ln target, bias-shift on calibration, return calibrated ln
    predictions of ln V for cal and each query set. ln_base_* is the anchor
    term added back (zeros for the unanchored arms)."""
    model.fit(Xtr, ln_target_tr)
    ln_pred_cal = model.predict(Xcal) + ln_base_cal
    shift = np.mean(ln_y_cal - ln_pred_cal)
    out = [model.predict(Xq) + b + shift for Xq, b in zip(Xq_list, ln_base_q_list)]
    return ln_pred_cal + shift, out


def m_stats(ln_y, ln_v):
    M = np.exp(ln_y - ln_v)
    return dict(bias=float(M.mean()), cov=float(M.std(ddof=1) / M.mean()),
                unsafe=float(np.mean(M < 1.0)))


def weighted_conformal(res_cal_abs, w_cal, res_test_abs, w_test):
    """Tibshirani-2019 weighted split conformal, per test point.
    Returns (covered_bool, qhat_or_inf) arrays."""
    order = np.argsort(res_cal_abs)
    r_sorted = res_cal_abs[order]
    w_sorted = w_cal[order]
    covered, qhats = [], []
    for r_t, w_t in zip(res_test_abs, w_test):
        tot = w_sorted.sum() + w_t
        cum = np.cumsum(w_sorted) / tot
        k = np.searchsorted(cum, 1 - ALPHA, side="left")
        if k >= len(r_sorted):          # quantile falls on the +inf atom
            qhats.append(np.inf)
            covered.append(True)
        else:
            q = r_sorted[k]
            qhats.append(q)
            covered.append(r_t <= q)
    return np.array(covered), np.array(qhats)


def run(label, df, feats, ycol, anchor_col):
    df = df.dropna(subset=feats + [ycol, anchor_col]).sort_values("d").reset_index(drop=True)
    ln_anchor_all = np.log(np.clip(df[anchor_col].values, 1e-9, None))
    arm_recs, wc_recs = [], []
    for thr in THRESHOLDS:
        d_hi = df.d.quantile(thr)
        pool_m = df.d.values < d_hi
        pool = df[pool_m].reset_index(drop=True)
        big = df[~pool_m].reset_index(drop=True)
        lnA_pool, lnA_big = ln_anchor_all[pool_m], ln_anchor_all[~pool_m]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(pool))
            a, b = int(.5 * len(pool)), int(.75 * len(pool))
            tr, cal, ind = idx[:a], idx[a:b], idx[b:]
            X = pool[feats].values
            ln_y = np.log(pool[ycol].values)
            Xe, ln_ye = big[feats].values, np.log(big[ycol].values)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Z = lambda M: (M - mu) / sd

            for mname, mk in models(seed).items():
                arms = {
                    "baseline": (X, ln_y, np.zeros(len(pool)), Xe, np.zeros(len(big))),
                    "residual-target": (X, ln_y - lnA_pool, lnA_pool, Xe, lnA_big),
                    "physics-feature": (np.column_stack([X, lnA_pool]), ln_y,
                                        np.zeros(len(pool)),
                                        np.column_stack([Xe, lnA_big]), np.zeros(len(big))),
                }
                for aname, (Xa, ta, base_pool, Xa_e, base_e) in arms.items():
                    ln_cal, (ln_ind, ln_big_pred) = fit_arm(
                        mk, Xa[tr], ta[tr], Xa[cal], ln_y[cal], base_pool[cal],
                        [Xa[ind], Xa_e], [base_pool[ind], base_e])
                    q = split_conformal_qhat(ln_y[cal] - ln_cal, ALPHA)
                    rec = dict(ds=label, thr=thr, seed=seed, model=mname, arm=aname,
                               cov_ind=coverage(ln_y[ind] - ln_ind, q),
                               cov_big=coverage(ln_ye - ln_big_pred, q),
                               width_factor=float(np.exp(2 * q)),
                               **{f"ooe_{k}": v
                                  for k, v in m_stats(ln_ye, ln_big_pred).items()})
                    arm_recs.append(rec)

                # weighted conformal on the baseline arm (HistGB only: one row per split)
                if mname == "HistGB":
                    mk2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                          max_leaf_nodes=8, min_samples_leaf=15, random_state=seed)
                    ln_cal, (ln_big_pred,) = fit_arm(
                        mk2, X[tr], ln_y[tr], X[cal], ln_y[cal],
                        np.zeros(len(cal)), [Xe], [np.zeros(len(big))])
                    clf = LogisticRegression(max_iter=2000).fit(
                        np.vstack([Z(X[tr]), Z(Xe)]),
                        np.r_[np.zeros(len(tr)), np.ones(len(big))])
                    def w_of(M):
                        p = np.clip(clf.predict_proba(Z(M))[:, 1], 1e-9, 1 - 1e-9)
                        return np.clip(p / (1 - p), 1e-3, 1e3)
                    res_cal = np.abs(ln_y[cal] - ln_cal)
                    res_big = np.abs(ln_ye - ln_big_pred)
                    covered, qh = weighted_conformal(res_cal, w_of(X[cal]),
                                                     res_big, w_of(Xe))
                    finite = np.isfinite(qh)
                    q_unw = split_conformal_qhat(ln_y[cal] - ln_cal, ALPHA)
                    wc_recs.append(dict(
                        ds=label, thr=thr, seed=seed,
                        frac_infinite=float(1 - finite.mean()),
                        cov_counting_inf=float(covered.mean()),
                        cov_finite_only=float(covered[finite].mean()) if finite.any() else np.nan,
                        med_width_factor_finite=float(np.exp(2 * np.median(qh[finite]))) if finite.any() else np.nan,
                        unweighted_width_factor=float(np.exp(2 * q_unw)),
                        n_big=len(big)))
    return pd.DataFrame(arm_recs), pd.DataFrame(wc_recs)


def main():
    steel = pd.read_csv(PROC / "steel_zhang_clean.csv")
    steel = steel[steel.a_d >= 2.5].dropna(
        subset=["bw", "d", "a_d", "rho", "fc", "ag", "fy", "Vu_kN"]).reset_index(drop=True)
    steel["Vmcft_kN"] = [mcft(r.bw, r.d, r.a_d, r.rho, r.fc) / 1e3
                         for r in steel.itertuples()]
    A1, W1 = run("STEEL", steel, ["bw", "d", "a_d", "rho", "fc", "ag", "fy"],
                 "Vu_kN", "Vmcft_kN")

    frp = pd.read_csv(PROC / "frp_clean.csv")
    frp = frp[(frp.a_d >= 2.5) & (frp.rho_l < 0.1)].reset_index(drop=True)
    frp["V440_kN"] = aci440_Vc_kN(frp)
    A2, W2 = run("FRP", frp, ["bw", "d", "fc", "rho_l", "Ef", "a_d"],
                 "V_test_kN", "V440_kN")

    A = pd.concat([A1, A2], ignore_index=True)
    W = pd.concat([W1, W2], ignore_index=True)
    A.to_csv(PROC / "physics_arms.csv", index=False)
    W.to_csv(PROC / "weighted_conformal.csv", index=False)

    print("== physics-anchored arms: OOE ln-conformal coverage (target 0.90) ==")
    g = (A.groupby(["ds", "arm", "model"])
           .agg(cov_ind=("cov_ind", "mean"), cov_big=("cov_big", "mean"),
                ooe_bias=("ooe_bias", "mean"), ooe_cov=("ooe_cov", "mean"),
                ooe_unsafe=("ooe_unsafe", "mean"),
                width=("width_factor", "mean")).round(3))
    print(g.to_string())
    print("\n== estimated-ratio weighted conformal (baseline HistGB) ==")
    gw = (W.groupby("ds").agg(frac_infinite=("frac_infinite", "mean"),
                              cov_counting_inf=("cov_counting_inf", "mean"),
                              cov_finite_only=("cov_finite_only", "mean"),
                              med_width_factor_finite=("med_width_factor_finite", "mean"),
                              unweighted_width_factor=("unweighted_width_factor", "mean"))
           .round(3))
    print(gw.to_string())


if __name__ == "__main__":
    main()
