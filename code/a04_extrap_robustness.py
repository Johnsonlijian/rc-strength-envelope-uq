"""
a04_extrap_robustness.py — is the extrapolation coverage-collapse robust?

Vary the size threshold and seeds; report interpolation vs size-extrapolation
conformal coverage (90% target) and mean model uncertainty M for RF & HistGB.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed")
FEATURES = ["bw", "d", "fc", "rho_l", "Ef", "a_d"]


def load_slender():
    df = pd.read_csv(PROC / "frp_clean.csv")
    df = df[(df["a_d"] >= 2.5) & (df["rho_l"] < 0.1)].copy()
    return df.dropna(subset=FEATURES + ["V_test_kN"]).reset_index(drop=True)


def mk(seed):
    return {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=seed),
        "RF": RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                  max_features=0.7, random_state=seed, n_jobs=-1),
    }


def main():
    df = load_slender().sort_values("d").reset_index(drop=True)
    rows = []
    for thr in [0.65, 0.70, 0.75, 0.80]:
        d_hi = df["d"].quantile(thr)
        pool = df[df.d < d_hi].reset_index(drop=True)
        test = df[df.d >= d_hi].reset_index(drop=True)
        for seed in [0, 1, 2]:
            for mname, model in mk(seed).items():
                Xp, yp = pool[FEATURES].values, pool["V_test_kN"].values
                # interpolation coverage (CV OOF on pool)
                oof = np.exp(cross_val_predict(model, Xp, np.log(yp),
                                               cv=KFold(10, shuffle=True, random_state=seed)))
                res_in = yp - oof
                q = split_conformal_qhat(res_in, 0.10)
                cov_in = coverage(res_in, q)   # in-sample-ish (cal=all pool) -> optimistic upper bound
                # extrapolation
                model.fit(Xp, np.log(yp))
                pred_pool = np.exp(model.predict(Xp)); res_pool = yp - pred_pool
                qe = split_conformal_qhat(res_pool, 0.10)
                pred_te = np.exp(model.predict(test[FEATURES].values))
                M_ex = test["V_test_kN"].values / pred_te
                cov_ex = coverage(test["V_test_kN"].values - pred_te, qe)
                rows.append(dict(thr=thr, n_test=len(test), model=mname, seed=seed,
                                 cov_interp=cov_in, cov_extrap=cov_ex, meanM_extrap=np.mean(M_ex)))
    r = pd.DataFrame(rows)
    print("== per-threshold means over seeds (interp vs extrap coverage; target 0.90) ==")
    agg = r.groupby(["thr", "n_test", "model"]).agg(
        cov_interp=("cov_interp", "mean"),
        cov_extrap=("cov_extrap", "mean"),
        meanM_extrap=("meanM_extrap", "mean")).round(3)
    print(agg.to_string())
    print("\n== overall extrapolation coverage (all thr/seed/model) ==")
    print(f"  mean cov_extrap = {r.cov_extrap.mean():.3f}  (min {r.cov_extrap.min():.3f}, max {r.cov_extrap.max():.3f})")
    print(f"  mean cov_interp = {r.cov_interp.mean():.3f}")
    print(f"  VERDICT: extrapolation coverage collapse is "
          f"{'ROBUST' if r.cov_extrap.mean() < 0.6 and r.cov_interp.mean() > 0.85 else 'NOT robust'}")


if __name__ == "__main__":
    main()
