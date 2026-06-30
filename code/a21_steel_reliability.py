"""
a21_steel_reliability.py — realised reliability vs member size on REAL steel data
(Zhang 1704). Replaces the controlled a14 demo: the 'true' capacity is the actual
measured V_test. Per size bin we compute the realised reliability index of a member
designed to beta_T with (i) the ML model calibrated in-envelope and (ii) a size-aware
mechanical model. Heavy: full Monte-Carlo beta with JCSS-style dead+live load.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from a12_mech_models import mcft
from reliability_engine import lognormal_from_mean_cov, normal_from_mean_cov, gumbel_from_mean_cov, _pf_to_beta

PROC=Path("../data/processed"); F=["bw","d","a_d","rho","fc","ag","fy"]; BETA_T=3.8


def mc_beta(R_mean, R_cov, util, n=400_000, seed=0):
    """MC beta: R=lognormal(R_mean,R_cov); load S=dead(normal)+live(gumbel) scaled so the
    nominal design (phi R_nom = gamma loads) sits at utilisation `util` of R_mean."""
    rng=np.random.default_rng(seed)
    S_mean=util*R_mean                     # mean load effect = util * mean resistance
    D=normal_from_mean_cov(0.6*S_mean,0.10).rvs(n,random_state=rng)      # dead 60%
    L=gumbel_from_mean_cov(0.4*S_mean,0.25).rvs(n,random_state=rng)      # live 40%
    R=lognormal_from_mean_cov(R_mean,R_cov).rvs(n,random_state=rng)
    g=R-(D+L); return _pf_to_beta(int((g<=0).sum()), n)


def util_for_betaT(R_cov):
    """find utilisation giving beta=BETA_T for a unit-mean resistance (bisection on MC)."""
    lo,hi=0.1,0.9
    for _ in range(22):
        mid=0.5*(lo+hi); b=mc_beta(1.0,R_cov,mid,n=200_000,seed=1)
        if b>BETA_T: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)


def main():
    df=pd.read_csv(PROC/"steel_zhang_clean.csv"); df=df[df.a_d>=2.5].dropna(subset=F+["Vu_kN"]).reset_index(drop=True)
    d_hi=df.d.quantile(0.33); pool=df[df.d<d_hi]
    ml=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=15,random_state=0)
    ml.fit(pool[F].values, np.log(pool.Vu_kN.values))
    sm=np.mean(np.exp(np.log(pool.Vu_kN.values)-ml.predict(pool[F].values)))
    df["Vml"]=np.exp(ml.predict(df[F].values))*sm
    df["Vmcft"]=[mcft(r.bw,r.d,r.a_d,r.rho,r.fc)/1e3 for r in df.itertuples()]
    # in-envelope ML calibration
    inb=df.d<d_hi; M_in=df.Vu_kN[inb]/df.Vml[inb]; cov_cv=M_in.std()/M_in.mean()
    util_ml=util_for_betaT(cov_cv)         # design utilisation (engineer uses in-env COV)
    M_me_in=df.Vu_kN[inb]/df.Vmcft[inb]; util_me=util_for_betaT(M_me_in.std()/M_me_in.mean())

    bins=[(0,150),(150,250),(250,400),(400,650),(650,2000)]
    print(f"in-env ML COV={cov_cv:.2f} -> design util={util_ml:.2f}; target beta_T={BETA_T}")
    print("size bin    | ML  bias COV  realised-beta | MCFT bias COV  realised-beta")
    rows=[]
    for lo,hi in bins:
        m=(df.d>=lo)&(df.d<hi);
        if m.sum()<8: continue
        Mml=(df.Vu_kN[m]/df.Vml[m]); Mme=(df.Vu_kN[m]/df.Vmcft[m])
        b_ml=mc_beta(Mml.mean(), Mml.std()/Mml.mean(), util_ml)     # realised: true bin M, design util
        b_me=mc_beta(Mme.mean(), Mme.std()/Mme.mean(), util_me)
        rows.append(dict(d_mid=(lo+hi)/2 if hi<2000 else 900, beta_ml=b_ml, beta_me=b_me,
                         cov_ml=Mml.std()/Mml.mean(), bias_ml=Mml.mean(), n=int(m.sum())))
        print(f"  {lo:4d}-{hi:4d} | ML {Mml.mean():.2f} {Mml.std()/Mml.mean():.2f}  beta={b_ml:.2f} | MCFT {Mme.mean():.2f} {Mme.std()/Mme.mean():.2f}  beta={b_me:.2f}")
    pd.DataFrame(rows).to_csv(PROC/"steel_reliability.csv", index=False)
    print("saved -> steel_reliability.csv")


if __name__=="__main__":
    main()
