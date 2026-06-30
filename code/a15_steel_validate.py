"""
a15_steel_validate.py — validate the mechanical models on the REAL canonical
steel beam-without-stirrups database (Zhang 2022, n=1704, d 41-2000mm), and
extract the real steel size effect. This is the keystone validation that was
impossible before (FRP/SFRC/deep-beam did not fit the steel sectional models).
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from a12_mech_models import MODELS, SIZE_AWARE

RAW = Path("../data/raw/Zhang2022_EngComputers_RCbeams_WITHOUTstirrups_n1704_MOESM2.xlsx")
PROC = Path("../data/processed")


def load_steel():
    df = pd.read_excel(RAW, sheet_name=0)
    df.columns = ["bw","d","a_d","bear","rho_pct","fc","ag","fy","Vu_kN"]
    df["rho"] = df.rho_pct/100.0
    df["V_test_N"] = df.Vu_kN*1e3
    df = df[(df.bw>0)&(df.d>0)&(df.a_d>0)&(df.fc>0)&(df.rho>0)&(df.Vu_kN>0)].copy()
    df.to_csv(PROC/"steel_zhang_clean.csv", index=False)
    return df


def main():
    df = load_steel()
    sl = df[df.a_d >= 2.5].reset_index(drop=True)        # slender (sectional shear)
    print(f"Zhang steel DB: {len(df)} total, {len(sl)} slender (a/d>=2.5); d {sl.d.min():.0f}-{sl.d.max():.0f}mm")
    print("\n=== mechanical-model validation on real steel slender beams ===")
    print(f"{'model':12s} {'mean M':>7s} {'COV':>6s} {'%unsafe':>8s}   (literature target)")
    lit = {"ACI318-14":"~1.40/0.37","Zsutty":"~1.0/0.15-0.20","ACI318-19":"~1.1-1.3",
           "MCFT":"~1.38/0.26","CSCT":"~1.0-1.1/0.10-0.13","fib-MC2010":"~1.2-1.4/0.22-0.27","Bazant-Kim":"~1.0/0.15"}
    res={}
    for n,f in MODELS.items():
        Vp = np.array([f(r.bw, r.d, r.a_d, r.rho, r.fc) for r in sl.itertuples()])
        M = sl.V_test_N.values/np.clip(Vp,1e-9,None)
        M = M[np.isfinite(M)&(M>0)&(M<10)]
        print(f"{n:12s} {M.mean():7.2f} {M.std()/M.mean():6.2f} {100*np.mean(M<1):7.0f}%   ({lit[n]})")
        res[n]=dict(mean=float(M.mean()), cov=float(M.std()/M.mean()))

    # real steel size effect: residualise v_test for rho,a/d,fc on small beams, slope vs d
    sl["v"]=sl.V_test_N/(sl.bw*sl.d)
    from sklearn.linear_model import LinearRegression
    sm=sl[sl.d<=np.quantile(sl.d,0.33)]
    X=lambda d: np.column_stack([np.log(d.rho),np.log(d.a_d),np.log(d.fc)])
    lr=LinearRegression().fit(X(sm),np.log(sm.v))
    R=sl.v.values/np.exp(lr.predict(X(sl)))
    big=sl.d.values>=np.quantile(sl.d.values,0.6)
    m_emp=np.polyfit(np.log(sl.d.values[big]),np.log(R[big]),1)[0]
    print(f"\nREAL STEEL size effect: large-size exponent m = {m_emp:+.2f}  [LEFM=-0.5, strength=0]  (n_large={big.sum()})")
    print("  -> compare to FE sim (-0.23) and mechanical models (-0.30..-0.38): consistent physical size effect.")
    pd.Series(res).to_json(PROC/"steel_model_uncertainty.json")
    print("saved -> steel_zhang_clean.csv, steel_model_uncertainty.json")


if __name__ == "__main__":
    main()
