"""
a14_reliability_across_size.py — reliability consequence of the size-effect mechanism
(controlled, mechanism-grounded demonstration).

"True" shear capacity = consensus of the validated size-aware mechanical models
(MCFT, CSCT, fib-MC2010), which agree with the first-principles FE (a13). A test-
like population is generated with realistic model-uncertainty scatter (COV 0.13).
An ML model trained on small/medium members is deployed across size; a size-aware
mechanical model is used as the alternative. We compute the realised reliability
index beta of a member designed to a target beta_T, as a function of size.

Result: ML-designed reliability degrades with size (its model error grows out of
envelope); mechanical-model reliability stays on target (size-stable error).
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from a12_mech_models import mcft, csct, fib_mc2010
from reliability_engine import beta_lognormal_RS

PROC = Path("../data/processed")
BETA_T, VS, COV_MECH = 3.8, 0.15, 0.13
rng = np.random.default_rng(0)


def consensus(b, d, ad, r, fc):
    return np.mean([mcft(b,d,ad,r,fc), csct(b,d,ad,r,fc), fib_mc2010(b,d,ad,r,fc)])


def realised_beta(bias, cov_assumed, bias_true, cov_true):
    A = np.sqrt((1+VS**2)/(1+cov_assumed**2)); B = np.sqrt(np.log((1+cov_assumed**2)*(1+VS**2)))
    muS = bias*A/np.exp(BETA_T*B)                  # design load (Rpred=1) to hit BETA_T
    return float(beta_lognormal_RS(bias_true, cov_true, muS, VS))


def main():
    N = 2500; bw = 250.0
    d = np.exp(rng.uniform(np.log(100), np.log(1500), N))
    ad = rng.uniform(2.5, 4.0, N); r = rng.uniform(0.008, 0.025, N); fc = rng.uniform(25, 45, N)
    Vc = np.array([consensus(bw, d[i], ad[i], r[i], fc[i]) for i in range(N)])
    Vtrue = Vc * np.exp(rng.normal(0, COV_MECH, N))          # test-like scatter

    feats = np.column_stack([np.full(N,bw), d, ad, r, fc])
    train = d < 300                                           # ML sees small/medium only
    ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
            min_samples_leaf=15, random_state=0).fit(feats[train], np.log(Vtrue[train]))
    Vml = np.exp(ml.predict(feats))

    bins = [(100,250),(250,400),(400,650),(650,1000),(1000,1500)]
    print("size bin (mm) | ML bias COV  realised-beta | MECH bias COV  realised-beta")
    rows=[]
    # in-envelope ML calibration (assumed by engineer)
    inb = d<300; Mml_in = Vtrue[inb]/Vml[inb]; bias_cv, cov_cv = Mml_in.mean(), Mml_in.std()/Mml_in.mean()
    for lo,hi in bins:
        m = (d>=lo)&(d<hi);
        Mml = Vtrue[m]/Vml[m]; b_ml, c_ml = Mml.mean(), Mml.std()/Mml.mean()
        Mme = Vtrue[m]/Vc[m];  b_me, c_me = Mme.mean(), Mme.std()/Mme.mean()
        beta_ml = realised_beta(bias_cv, cov_cv, b_ml, c_ml)   # designed w/ in-env ML calib
        beta_me = realised_beta(1.0, COV_MECH, b_me, c_me)     # designed w/ mech calib
        rows.append(dict(d_mid=(lo+hi)/2, beta_ml=beta_ml, beta_me=beta_me,
                         cov_ml=c_ml, cov_me=c_me, bias_ml=b_ml))
        print(f"  {lo:4d}-{hi:4d} | ML {b_ml:.2f} {c_ml:.2f}  beta={beta_ml:.2f} | MECH {b_me:.2f} {c_me:.2f}  beta={beta_me:.2f}")
    pd.DataFrame(rows).to_csv(PROC/"reliability_across_size.csv", index=False)
    print(f"\nML reliability falls from beta~{rows[0]['beta_ml']:.1f} (small) to {rows[-1]['beta_ml']:.1f} (large, d~1250mm);"
          f" mechanical stays ~{np.mean([r['beta_me'] for r in rows]):.1f} (target {BETA_T}).")
    print("saved -> data/processed/reliability_across_size.csv")


if __name__ == "__main__":
    main()
