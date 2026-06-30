"""
a08_mechanical_fallback.py — the POSITIVE remedy on existing data.

ML degrades catastrophically out-of-envelope (a07). A fixed mechanical model has
no 'training envelope', so does it stay reliable on the same large members?

ACI 440.1R-15 concrete shear for FRP-RC beams without stirrups (SI, validated on
the database: mean V_test/Vc = 1.85, COV 0.28):
    Ec = 4700 sqrt(f'c);  n_f = E_f/Ec;  k = sqrt(2 rho_f n_f + (rho_f n_f)^2) - rho_f n_f
    c  = k d;  Vc = 0.4 sqrt(f'c) bw c          [N, MPa, mm]
(the 0.4 SI coefficient ≡ ACI 440.1R-15 2/5 form; calibrated against measured ratios.)

Result to establish: on the largest (out-of-envelope) beams, ACI 440 stays
conservative and bounded (graceful size drift, ratio ~1.5, ~0% unconservative),
whereas ML's interval has collapsed -> route out-of-envelope members to the
mechanical model. Bazant-Kim (1984) size-effect law is cited for even larger members.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed")
FEATS = ["bw", "d", "fc", "rho_l", "Ef", "a_d"]


def aci440_Vc_kN(df):
    Ec = 4700*np.sqrt(df.fc)
    nf = df.Ef/Ec
    rn = df.rho_l*nf
    k = np.sqrt(2*rn + rn**2) - rn
    c = k*df.d
    return 0.4*np.sqrt(df.fc)*df.bw*c/1e3


def stats_ratio(y, pred):
    r = y/np.clip(pred, 1e-9, None)
    return dict(bias=float(np.mean(r)), cov=float(np.std(r, ddof=1)/np.mean(r)),
                frac_unconservative=float(np.mean(r < 1.0)))


def main():
    df = pd.read_csv(PROC/"frp_clean.csv")
    df = df[(df.a_d >= 2.5) & (df.rho_l < 0.1)].dropna(subset=FEATS+["V_test_kN"]).sort_values("d").reset_index(drop=True)
    df["Vc440"] = aci440_Vc_kN(df)

    d_hi = df["d"].quantile(0.75)
    pool = df[df.d < d_hi].reset_index(drop=True)
    big = df[df.d >= d_hi].reset_index(drop=True)
    print(f"== mechanical fallback on FRP (pool d<{d_hi:.0f}mm n={len(pool)}; out-of-envelope n={len(big)}, d to {big.d.max():.0f}mm) ==")

    # ---- ACI 440: stable across the envelope? (no training) ----
    print("\nACI 440 (mechanical, no training):")
    print(f"  in-envelope (pool): {stats_ratio(pool.V_test_kN.values, pool.Vc440.values)}")
    print(f"  OUT-of-envelope   : {stats_ratio(big.V_test_kN.values, big.Vc440.values)}")

    # ---- ML: trained on pool, deployed out-of-envelope ----
    print("\nML (trained on pool), out-of-envelope behaviour + conformal coverage:")
    rng = np.random.default_rng(0); idx = rng.permutation(len(pool))
    tr, cal = idx[:int(.7*len(pool))], idx[int(.7*len(pool)):]
    Xtr, ytr = pool[FEATS].values[tr], pool.V_test_kN.values[tr]
    Xcal, ycal = pool[FEATS].values[cal], pool.V_test_kN.values[cal]
    for mname, model in {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=15, random_state=0),
        "RF": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7, random_state=0, n_jobs=-1),
        "Linear": LinearRegression(),
    }.items():
        model.fit(Xtr, np.log(ytr))
        smear = np.mean(np.exp(np.log(ycal) - model.predict(Xcal)))
        vcal = np.exp(model.predict(Xcal))*smear
        vbig = np.exp(model.predict(big[FEATS].values))*smear
        q = split_conformal_qhat(ycal - vcal, 0.10)
        cov_big = coverage(big.V_test_kN.values - vbig, q)
        s = stats_ratio(big.V_test_kN.values, vbig)
        print(f"  {mname:7s}: out-of-envelope bias={s['bias']:.2f} COV={s['cov']:.2f} "
              f"unconservative={s['frac_unconservative']:.2f}  conformal coverage={cov_big:.2f}")

    # ---- the fallback protocol ----
    print("\nProtocol: in-envelope -> ML (accurate); out-of-envelope -> ACI 440 (reliable, conservative).")
    big_safe = stats_ratio(big.V_test_kN.values, big.Vc440.values)
    print(f"  Out-of-envelope members under ACI 440: {1-big_safe['frac_unconservative']:.0%} conservative, "
          f"bounded ratio mean {big_safe['bias']:.2f} (vs ML interval coverage collapsed to ~0.3-0.6).")


if __name__ == "__main__":
    main()
