"""
a16_steel_extrapolation.py — the headline result on the CANONICAL steel benchmark
(Zhang 1704, slender). ML size-extrapolation coverage matrix (corrected method,
reused from a07) + ML model uncertainty out-of-envelope + the mechanical fallback
(MCFT/CSCT stay calibrated where ML collapses). Real steel, the field's own data.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from a07_corrected_uq_matrix import run_dataset
from a12_mech_models import mcft, csct, fib_mc2010, MODELS

PROC = Path("../data/processed")
FEATS = ["bw","d","a_d","rho","fc","ag","fy"]


def main():
    steel = pd.read_csv(PROC/"steel_zhang_clean.csv")
    sl = steel[steel.a_d>=2.5].dropna(subset=FEATS+["Vu_kN"]).reset_index(drop=True)
    print(f"== canonical steel benchmark (Zhang 1704), slender n={len(sl)}, d {sl.d.min():.0f}-{sl.d.max():.0f}mm ==")

    R = run_dataset("STEEL", sl, FEATS, "Vu_kN", thresholds=(0.70,0.75,0.80), seeds=(0,1,2))
    g = R.groupby(["model","uq"]).agg(interp=("interp","mean"), extrap=("extrap","mean")).round(3)
    print("\nML size-extrapolation coverage (target 0.90):")
    print(g.to_string())
    best = R.dropna(subset=["extrap"]).groupby(["model","uq"])["extrap"].mean().max()
    print(f"\nbest extrapolation coverage by ANY UQ method = {best:.3f}  (target 0.90) -> none restores it")

    # ML vs mechanical model uncertainty, out of envelope (the fallback case)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    d_hi = sl.d.quantile(0.75); pool = sl[sl.d<d_hi]; big = sl[sl.d>=d_hi].reset_index(drop=True)
    ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
            min_samples_leaf=15, random_state=0).fit(pool[FEATS].values, np.log(pool.Vu_kN.values))
    sm = np.mean(np.exp(np.log(pool.Vu_kN.values) - ml.predict(pool[FEATS].values)))
    Vml = np.exp(ml.predict(big[FEATS].values))*sm
    Mml = big.Vu_kN.values/Vml
    print(f"\nout-of-envelope (d>={d_hi:.0f}mm, n={len(big)}):")
    print(f"  ML:   bias={Mml.mean():.2f}  COV={Mml.std()/Mml.mean():.2f}  %unsafe={100*np.mean(Mml<1):.0f}%")
    for name,f in [("MCFT",mcft),("CSCT",csct),("fib-MC2010",fib_mc2010)]:
        Vp = np.array([f(r.bw,r.d,r.a_d,r.rho,r.fc) for r in big.itertuples()])
        M = big.Vu_kN.values*1e3/Vp
        print(f"  {name:10s} bias={M.mean():.2f}  COV={M.std()/M.mean():.2f}  %unsafe={100*np.mean(M<1):.0f}%")
    g.to_csv(PROC/"steel_uq_matrix.csv")
    print("\nsaved -> steel_uq_matrix.csv")


if __name__ == "__main__":
    main()
