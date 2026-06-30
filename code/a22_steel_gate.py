"""
a22_steel_gate.py — the applicability-domain gate on the canonical steel benchmark.
A kNN out-of-distribution gate (NOT using member depth d) calibrated on the training
pool: (i) abstention in-domain vs flagging out-of-envelope; (ii) does gating RESTORE
coverage on the retained set? (the deployable-safeguard test).
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from uq_utils import split_conformal_qhat, coverage
PROC=Path("../data/processed")
F=["bw","d","a_d","rho","fc","ag","fy"]; GF=["bw","a_d","rho","fc","ag","fy"]   # gate features exclude d

def main():
    df=pd.read_csv(PROC/"steel_zhang_clean.csv"); df=df[df.a_d>=2.5].dropna(subset=F+["Vu_kN"]).sort_values("d").reset_index(drop=True)
    d_hi=df.d.quantile(0.75); pool=df[df.d<d_hi].reset_index(drop=True); big=df[df.d>=d_hi].reset_index(drop=True)
    rng=np.random.default_rng(0); idx=rng.permutation(len(pool)); a,b=int(.5*len(pool)),int(.7*len(pool))
    tr,cal,ind=idx[:a],idx[a:b],idx[b:]
    ml=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=15,random_state=0)
    ml.fit(pool[F].values[tr],np.log(pool.Vu_kN.values[tr]))
    sm=np.mean(np.exp(np.log(pool.Vu_kN.values[cal])-ml.predict(pool[F].values[cal])))
    def vp(X): return np.exp(ml.predict(X))*sm
    q=split_conformal_qhat(pool.Vu_kN.values[cal]-vp(pool[F].values[cal]),0.10)

    # gate: kNN distance in standardised GF space (no d), threshold = 95th pct of training
    mu,sd=pool[GF].values[tr].mean(0),pool[GF].values[tr].std(0)+1e-9
    Z=lambda X:(X-mu)/sd
    nn=NearestNeighbors(n_neighbors=6).fit(Z(pool[GF].values[tr]))
    def ood(X): d,_=nn.kneighbors(Z(X)); return d[:,1:].mean(1)
    thr=np.quantile(ood(pool[GF].values[tr]),0.95)

    s_ind=ood(pool[GF].values[ind]); s_big=ood(big[GF].values)
    ab_in=np.mean(s_ind>thr); ab_big=np.mean(s_big>thr)
    print(f"steel gate (kNN, no d): in-domain abstention={ab_in:.2f}  out-of-envelope flagged={ab_big:.2f}")

    # retained-coverage test on a MIXED deployment set (in-domain holdout + large)
    Xmix=np.vstack([pool[F].values[ind],big[F].values]); ymix=np.concatenate([pool.Vu_kN.values[ind],big.Vu_kN.values])
    Gmix=np.vstack([pool[GF].values[ind],big[GF].values]); flag=ood(Gmix)>thr
    res=ymix-vp(Xmix)
    cov_all=coverage(res,q); cov_kept=coverage(res[~flag],q); frac_kept=np.mean(~flag)
    # feature-range applicability check (d included) catches the rest by construction
    in_range=(big.d.values<=pool.d.values[tr].max())
    print(f"mixed deployment: coverage all={cov_all:.2f}  ->  gated-retained={cov_kept:.2f} (kept {frac_kept:.0%})")
    print(f"  feature-range (d) check flags {100*(1-in_range.mean()):.0f}% of large members by construction")
    # mechanical fallback on flagged
    from a12_mech_models import mcft,csct
    Vmc=np.array([mcft(r.bw,r.d,r.a_d,r.rho,r.fc) for r in big.itertuples()])/1e3
    Mmc=big.Vu_kN.values/Vmc
    print(f"flagged (large) members under MCFT fallback: bias={Mmc.mean():.2f} COV={Mmc.std()/Mmc.mean():.2f}")
    pd.Series(dict(ab_in=ab_in,ab_big=ab_big,cov_all=cov_all,cov_kept=cov_kept,frac_kept=frac_kept)).to_json(PROC/"steel_gate.json")
    print("saved -> steel_gate.json")

if __name__=="__main__": main()
