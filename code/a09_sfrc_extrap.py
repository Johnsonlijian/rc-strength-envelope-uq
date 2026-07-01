"""
a09_sfrc_extrap.py — honest third-material replication on the SFRC database.
Proper split-conformal (train/cal/eval), interp vs size-extrapolation coverage.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from uq_utils import split_conformal_qhat, coverage

RAW = Path("../data/raw/Zenodo_2578061_SFRC_shear_beams.xlsx")
PROC = Path("../data/processed")
# positional map (header=None): names in row 1, data from row 3
POS = {"b": 2, "h": 3, "d": 4, "rho_l": 16, "fy": 17, "a_d": 18, "fc": 21,
       "Lf": 24, "Vf": 25, "V_test_kN": 38}
FEATS = ["b", "d", "fc", "rho_l", "a_d", "fy", "Vf"]


def load():
    raw = pd.read_excel(RAW, header=None)
    df = raw.iloc[3:, list(POS.values())].copy()
    df.columns = list(POS)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["b", "d", "fc", "rho_l", "a_d", "V_test_kN"])
    df = df[(df.V_test_kN > 0) & (df.d > 0) & (df.a_d >= 2.5)].reset_index(drop=True)
    df["Vf"] = df["Vf"].fillna(df["Vf"].median())
    df["fy"] = df["fy"].fillna(df["fy"].median())
    df.to_csv(PROC / "sfrc_clean.csv", index=False)
    return df


def main():
    df = load().sort_values("d").reset_index(drop=True)
    print(f"== SFRC size-extrapolation (n={len(df)}; d {df.d.min():.0f}-{df.d.max():.0f}mm) ==")
    rows = []
    for thr in [0.70, 0.75, 0.80]:
        d_hi = df.d.quantile(thr)
        pool = df[df.d < d_hi].reset_index(drop=True); test = df[df.d >= d_hi].reset_index(drop=True)
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(seed); idx = rng.permutation(len(pool))
            a, b = int(.5*len(pool)), int(.75*len(pool))
            tr, cal, ind = idx[:a], idx[a:b], idx[b:]
            for mname, m in {
                "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10, random_state=seed),
                "RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=-1),
            }.items():
                Xtr, ytr = pool[FEATS].values[tr], pool.V_test_kN.values[tr]
                m.fit(Xtr, np.log(ytr))
                pcl = m.predict(pool[FEATS].values[cal])
                sm = np.mean(np.exp(np.log(pool.V_test_kN.values[cal]) - pcl))
                vcal = np.exp(pcl)*sm
                vind = np.exp(m.predict(pool[FEATS].values[ind]))*sm
                vte = np.exp(m.predict(test[FEATS].values))*sm
                q = split_conformal_qhat(pool.V_test_kN.values[cal]-vcal, 0.10)
                rows.append(dict(model=mname, thr=thr, seed=seed,
                                 interp=coverage(pool.V_test_kN.values[ind]-vind, q),
                                 extrap=coverage(test.V_test_kN.values-vte, q)))
    r = pd.DataFrame(rows)
    r.to_csv(PROC / "sfrc_uq_matrix.csv", index=False)
    print(r.groupby("model").agg(interp=("interp","mean"), extrap=("extrap","mean")).round(3).to_string())
    print(f"\noverall interp={r.interp.mean():.3f} extrap={r.extrap.mean():.3f} -> "
          f"{'REPLICATES collapse' if r.extrap.mean()<0.7 and r.interp.mean()>0.8 else 'no'}")
    print("saved -> sfrc_uq_matrix.csv")


if __name__ == "__main__":
    main()
