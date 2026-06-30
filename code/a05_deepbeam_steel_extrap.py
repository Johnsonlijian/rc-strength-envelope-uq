"""
a05_deepbeam_steel_extrap.py — replicate the extrapolation coverage-collapse on a
large STEEL shear database (Megahed deep-beam set, 840 specimens, d up to 2000 mm).

Different material (steel vs FRP) and different mechanism (deep-beam strut-and-tie
vs slender sectional shear): if the collapse replicates here, it is a general
property of ML shear-strength UQ, not a quirk of one dataset.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from uq_utils import split_conformal_qhat, coverage

RAW = Path("../data/raw/github_kmegahed_DeepBeamML_deep_beam2222.xlsx")
PROC = Path("../data/processed")
FEATURES = ["h", "d", "b", "a_d", "fck", "rho", "fy", "rho_v", "fyv", "rho_h", "fyh"]


def load():
    df = pd.read_excel(RAW, sheet_name="all")
    for c in FEATURES + ["V"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["V"])
    df = df[(df["V"] > 0) & (df["d"] > 0)].reset_index(drop=True)
    df.to_csv(PROC / "deepbeam_steel_clean.csv", index=False)
    return df


def mk(seed):
    return {
        "HistGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                  max_leaf_nodes=16, min_samples_leaf=10, random_state=seed),
        "RF": RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                  max_features=0.7, random_state=seed, n_jobs=-1),
    }


def main():
    df = load().sort_values("d").reset_index(drop=True)
    print(f"== STEEL deep-beam extrapolation test (n={len(df)}) ==")
    print(f"d {df.d.min():.0f}-{df.d.max():.0f} mm; V {df.V.min():.0f}-{df.V.max():.0f} kN; "
          f"a/d {df.a_d.min():.2f}-{df.a_d.max():.2f}; rho_v=0 in {int((df.rho_v==0).sum())} rows")
    rows = []
    for thr in [0.70, 0.75, 0.80, 0.85]:
        d_hi = df["d"].quantile(thr)
        pool = df[df.d < d_hi].reset_index(drop=True)
        test = df[df.d >= d_hi].reset_index(drop=True)
        for seed in [0, 1, 2]:
            for mname, model in mk(seed).items():
                Xp, yp = pool[FEATURES].values, pool["V"].values
                oof = np.exp(cross_val_predict(model, Xp, np.log(yp),
                             cv=KFold(10, shuffle=True, random_state=seed)))
                q = split_conformal_qhat(yp - oof, 0.10)
                cov_in = coverage(yp - oof, q)
                model.fit(Xp, np.log(yp))
                pred_pool = np.exp(model.predict(Xp))
                qe = split_conformal_qhat(yp - pred_pool, 0.10)
                pred_te = np.exp(model.predict(test[FEATURES].values))
                M_ex = test["V"].values / pred_te
                cov_ex = coverage(test["V"].values - pred_te, qe)
                rows.append(dict(thr=thr, n_test=len(test), model=mname,
                                 cov_interp=cov_in, cov_extrap=cov_ex, meanM_extrap=np.mean(M_ex)))
    r = pd.DataFrame(rows)
    agg = r.groupby(["thr", "n_test", "model"]).agg(
        cov_interp=("cov_interp", "mean"), cov_extrap=("cov_extrap", "mean"),
        meanM_extrap=("meanM_extrap", "mean")).round(3)
    print(agg.to_string())
    print(f"\noverall: cov_interp={r.cov_interp.mean():.3f}  cov_extrap={r.cov_extrap.mean():.3f} "
          f"(min {r.cov_extrap.min():.3f}, max {r.cov_extrap.max():.3f})")
    print("VERDICT:", "REPLICATES (collapse)" if r.cov_extrap.mean() < 0.6 and r.cov_interp.mean() > 0.85
          else "does NOT replicate")


if __name__ == "__main__":
    main()
