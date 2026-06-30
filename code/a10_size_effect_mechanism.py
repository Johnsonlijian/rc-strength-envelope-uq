"""
a10_size_effect_mechanism.py — the fracture-mechanics mechanism (real computation).

Why does ML fail out of envelope? Because shear of members without stirrups has a
SIZE EFFECT (Bazant): small members fail in the strength regime (nominal shear
stress ~ const); large members fail by fracture-energy-controlled cracking, with
nominal stress decaying ~ d^(-1/2) (LEFM). A learner trained on a small/medium-
dominated database has no fracture mechanism to produce that decay.

We isolate the size effect rigorously (controlling rho, a/d, f'c):
  1. fit the STRENGTH-REGIME power law  v = A rho^a (a/d)^b fc^c  on SMALL beams;
  2. residual R = v_test / v_strength  vs depth d reveals the pure size effect;
  3. fit Bazant SEL  R = (1+d/d0)^(-1/2)  -> transitional size d0;
  4. free large-size exponent m (R ~ d^m) -> compare to LEFM (-0.5) and none (0);
  5. ML effective size exponent (same residualisation on ML predictions).

Slender sets only (a/d>=2.5), where sectional size effect applies: FRP + SFRC.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

PROC = Path("../data/processed")


# per-material strength-regime variables (control everything EXCEPT size)
STRENGTH_VARS = {"FRP": ["rho","a_d","fc","Ef"],     # FRP: axial stiffness rho*Ef matters
                 "SFRC": ["rho","a_d","fc","Vf"],     # SFRC: fibre volume matters
                 "STEEL-deep": ["rho","a_d","fc"]}
ML_FEATS = {"FRP": ["b","d","a_d","rho","fc","Ef"],
            "SFRC": ["b","d","a_d","rho","fc","Vf"],
            "STEEL-deep": ["b","d","a_d","rho","fc"]}


def load_unified():
    out = {}
    frp = pd.read_csv(PROC/"frp_clean.csv")
    frp = frp[(frp.a_d>=2.5)&(frp.rho_l<0.1)].dropna(subset=["bw","d","fc","rho_l","a_d","Ef","V_test_kN"])
    out["FRP"] = pd.DataFrame({"b":frp.bw,"d":frp.d,"a_d":frp.a_d,"rho":frp.rho_l,
                               "fc":frp.fc,"Ef":frp.Ef,"V":frp.V_test_kN*1e3})
    s = pd.read_csv(PROC/"sfrc_clean.csv")
    s = s[(s.a_d>=2.5)].dropna(subset=["b","d","fc","rho_l","a_d","V_test_kN"])
    s["Vf"] = pd.to_numeric(s.get("Vf", 0.5), errors="coerce").fillna(0.5).clip(0.01, None)
    out["SFRC"] = pd.DataFrame({"b":s.b,"d":s.d,"a_d":s.a_d,"rho":s.rho_l,
                                "fc":s.fc,"Vf":s.Vf,"V":s.V_test_kN*1e3})
    st = pd.read_csv(PROC/"deepbeam_steel_clean.csv").dropna(subset=["b","d","a_d","fck","rho","V"])
    out["STEEL-deep"] = pd.DataFrame({"b":st.b,"d":st.d,"a_d":st.a_d,"rho":st.rho,
                                      "fc":st.fck,"V":st.V*1e3})
    for k,df in out.items():
        df["v"] = df.V/(df.b*df.d)             # nominal shear stress MPa
        out[k] = df[(df.v>0)&(df.rho>0)&(df.fc>0)&(df.d>0)].reset_index(drop=True)
    return out


def _Xstr(df, name):
    return np.column_stack([np.log(np.clip(df[c].values,1e-9,None)) for c in STRENGTH_VARS[name]])


def strength_fit(df, name, d_small):
    """fit log v on the material's strength-regime variables, on SMALL beams only."""
    sm = df[df.d <= d_small]
    lr = LinearRegression().fit(_Xstr(sm, name), np.log(sm.v.values))
    return lr, len(sm)


def v_strength(df, name, lr):
    return np.exp(lr.predict(_Xstr(df, name)))


def fit_sel(d, R):
    """fit R = R0 (1+d/d0)^(-1/2): grid d0, R0=mean(R*sqrt(1+d/d0))."""
    def ssr(logd0):
        d0=np.exp(logd0); f=(1+d/d0)**-0.5; R0=np.sum(R*f)/np.sum(f*f)
        return np.sum((R-R0*f)**2)
    res=minimize_scalar(ssr, bounds=(np.log(20),np.log(20000)), method="bounded")
    d0=np.exp(res.x); f=(1+d/d0)**-0.5; R0=np.sum(R*f)/np.sum(f*f)
    sse=np.sum((R-R0*f)**2); sst=np.sum((R-R.mean())**2)
    return d0, R0, 1-sse/sst


def large_exponent(d, R, q=0.6):
    """free power-law exponent of R vs d on the LARGER half (R ~ d^m)."""
    m_ = d >= np.quantile(d, q)
    if m_.sum() < 8: m_ = d >= np.quantile(d, 0.4)
    lr = LinearRegression().fit(np.log(d[m_]).reshape(-1,1), np.log(R[m_]))
    return float(lr.coef_[0]), int(m_.sum())


def main():
    data = load_unified()
    for name, df in data.items():
        d = df.d.values
        d_small = np.quantile(d, 0.33)
        lr, n_s = strength_fit(df, name, d_small)
        R = df.v.values / v_strength(df, name, lr)
        d0, R0, r2 = fit_sel(d, R)
        m_emp, n_l = large_exponent(d, R)
        sse_size = np.sum((R - R0*(1+d/d0)**-0.5)**2); sse_none=np.sum((R-R.mean())**2)

        # ML effective size exponent from SMALL-trained EXTRAPOLATION (the real test)
        feats = ML_FEATS[name]
        small = df[df.d <= d_small]
        ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                 max_leaf_nodes=8, min_samples_leaf=10, random_state=0)
        ml.fit(small[feats].values, np.log(small.V.values))
        v_ml = np.exp(ml.predict(df[feats].values))/(df.b.values*df.d.values)
        R_ml = v_ml / v_strength(df, name, lr)
        m_ml, _ = large_exponent(d, R_ml)

        sv = ", ".join(f"{c}^{lr.coef_[i]:.2f}" for i,c in enumerate(STRENGTH_VARS[name]))
        print(f"\n================ {name} (n={len(df)}; d {d.min():.0f}-{d.max():.0f} mm; a/d {df.a_d.min():.1f}-{df.a_d.max():.1f}) ================")
        print(f"  strength-regime power law (small beams n={n_s}):  v ~ {sv}")
        print(f"  Bazant SEL fit:  R = R0 (1+d/d0)^(-1/2),  d0 = {d0:.0f} mm,  R0={R0:.2f},  R^2={r2:.2f}")
        print(f"  size term explains {100*(1-sse_size/sse_none):.0f}% of residual variance")
        print(f"  EMPIRICAL large-size exponent m (data)      = {m_emp:+.2f}   [LEFM=-0.50, no-size=0.00]   (large n={n_l})")
        print(f"  ML (trained on small only) extrap exponent  = {m_ml:+.2f}   <-- can ML reproduce the decay?")
        verdict = ("DATA HAS WEAK/NO SIZE EFFECT" if abs(m_emp) < 0.20 else
                   "ML MISSES IT" if abs(m_ml) < 0.5*abs(m_emp) else "ML reproduces it")
        print(f"  >> {verdict}")
    data_path = PROC/"size_effect_unified.pkl"
    import pickle; pickle.dump(data, open(data_path,"wb"))
    print(f"\nsaved unified data -> {data_path}")


if __name__ == "__main__":
    main()
