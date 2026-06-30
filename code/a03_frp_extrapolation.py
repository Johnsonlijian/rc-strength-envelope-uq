"""
a03_frp_extrapolation.py — the size-EXTRAPOLATION decision-admissibility test.

Hypothesis (reframed after F2 was falsified): the size effect does not show up as
residual heteroscedasticity under random CV (a flexible ML model fits it away and
looks well-calibrated). It shows up when the ML model is DEPLOYED beyond its
training size range — the real situation, since shear labs test mostly small/mid
beams but real members are larger. There the ML model has no size-effect guarantee
and its coverage-calibrated interval gives no warning.

Protocol: train+calibrate on the smaller beams, TEST on the largest beams (a true
size-extrapolation hold-out). Compare model uncertainty M=V_test/V_pred and
conformal coverage between INTERPOLATION (random CV on the small pool) and
EXTRAPOLATION (the large hold-out). 'Unconservative' = V_pred > V_test (M<1).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed")
FEATURES = ["bw", "d", "fc", "rho_l", "Ef", "a_d"]


def load_slender():
    df = pd.read_csv(PROC / "frp_clean.csv")
    df = df[(df["a_d"] >= 2.5) & (df["rho_l"] < 0.1)].copy()
    return df.dropna(subset=FEATURES + ["V_test_kN"]).reset_index(drop=True)


def fit_pred(model, Xtr, ytr, Xte):
    model.fit(Xtr, np.log(ytr))
    return np.exp(model.predict(Xte))


def summ(tag, M, cov_abs, cov_rel):
    unders = float(np.mean(M < 1.0))
    return (f"  {tag:14s} n={len(M):3d}  mean M={np.mean(M):.3f}  COV(M)={np.std(M,ddof=1)/np.mean(M):.3f}  "
            f"P(unconservative M<1)={unders:.2f}  cov_abs={cov_abs:.3f}  cov_rel={cov_rel:.3f}")


def main():
    df = load_slender().sort_values("d").reset_index(drop=True)
    n = len(df)
    d_hi = df["d"].quantile(0.75)
    big = df["d"] >= d_hi
    pool = df[~big].reset_index(drop=True)     # small/mid beams (train+cal)
    test = df[big].reset_index(drop=True)      # largest 25% (extrapolation)
    print(f"== size-extrapolation test (n={n}) ==")
    print(f"train/cal pool: d<{d_hi:.0f}mm n={len(pool)} (d {pool.d.min():.0f}-{pool.d.max():.0f}); "
          f"extrap test: d>={d_hi:.0f}mm n={len(test)} (d {test.d.min():.0f}-{test.d.max():.0f})")

    models = {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=0),
        "RF": RandomForestRegressor(n_estimators=800, min_samples_leaf=3,
                  max_features=0.7, random_state=0, n_jobs=-1),
        "Linear": LinearRegression(),
    }

    for mname, model in models.items():
        print(f"\n#### {mname}")
        # ---- INTERPOLATION: random 10-fold OOF on the small/mid pool ----
        Xp, yp = pool[FEATURES].values, pool["V_test_kN"].values
        cv = KFold(10, shuffle=True, random_state=0)
        pred_oof = np.exp(cross_val_predict(model, Xp, np.log(yp), cv=cv))
        M_in = yp / pred_oof
        res_in = yp - pred_oof
        # conformal from a held-out half of the pool
        rng = np.random.default_rng(0); idx = rng.permutation(len(pool))
        ca, te = idx[:len(pool)//2], idx[len(pool)//2:]
        q_abs = split_conformal_qhat(res_in[ca], 0.10)
        q_rel = split_conformal_qhat((res_in/pred_oof)[ca], 0.10)
        cov_in_abs = coverage(res_in[te], q_abs)
        cov_in_rel = float(np.mean(np.abs((res_in/pred_oof)[te]) <= q_rel))
        print(summ("INTERP (CV)", M_in, cov_in_abs, cov_in_rel))

        # ---- EXTRAPOLATION: fit on whole pool, predict the large hold-out ----
        # conformal calibrated on the pool (small beams) — the honest deploy case
        pred_pool = fit_pred(model, Xp, yp, Xp)       # in-pool for cal residuals
        res_pool = yp - pred_pool
        q_abs_e = split_conformal_qhat(res_pool, 0.10)
        q_rel_e = split_conformal_qhat(res_pool/pred_pool, 0.10)
        pred_te = fit_pred(model, Xp, yp, test[FEATURES].values)
        M_ex = test["V_test_kN"].values / pred_te
        res_ex = test["V_test_kN"].values - pred_te
        cov_ex_abs = coverage(res_ex, q_abs_e)
        cov_ex_rel = float(np.mean(np.abs(res_ex/pred_te) <= q_rel_e))
        print(summ("EXTRAP (large)", M_ex, cov_ex_abs, cov_ex_rel))


if __name__ == "__main__":
    main()
