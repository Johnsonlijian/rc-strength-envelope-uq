"""
a25_column_extrap.py — REAL RC column data: does ML capacity prediction fail beyond
its training envelope on TWO axes — size AND (the column-specific new dimension)
AXIAL-LOAD RATIO? Proper split-conformal (train/cal/eval) + Duan smearing + UQ methods.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from uq_utils import split_conformal_qhat, coverage
PROC=Path("../data/processed")
FEATS=["b","h","d1","Lc","a_d","fc","fyc","rho_l","rho_t","n_ax"]

def knn_sigma(Ztr,res,Zq,k=10):
    nn=NearestNeighbors(n_neighbors=min(k,len(Ztr))).fit(Ztr); _,idx=nn.kneighbors(Zq)
    return np.maximum([np.mean(np.abs(res[ix])) for ix in idx],1e-6)

def run_axis(df, axis, label, thr=0.75, seeds=(0,1,2)):
    df=df.sort_values(axis).reset_index(drop=True); cut=df[axis].quantile(thr)
    pool=df[df[axis]<cut].reset_index(drop=True); big=df[df[axis]>=cut].reset_index(drop=True)
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); idx=rng.permutation(len(pool)); a,b=int(.5*len(pool)),int(.75*len(pool))
        tr,cal,ind=idx[:a],idx[a:b],idx[b:]
        Xtr,ytr=pool[FEATS].values[tr],np.log(pool.V_test_kN.values[tr])
        mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9
        for mname,m in {"HistGB":HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=8,random_state=seed),
                        "RF":RandomForestRegressor(n_estimators=400,min_samples_leaf=3,max_features=0.7,random_state=seed,n_jobs=-1)}.items():
            m.fit(Xtr,ytr); pcal=m.predict(pool[FEATS].values[cal])
            smr=np.mean(np.exp(np.log(pool.V_test_kN.values[cal])-pcal))
            vcal=np.exp(pcal)*smr; vind=np.exp(m.predict(pool[FEATS].values[ind]))*smr; vbig=np.exp(m.predict(big[FEATS].values))*smr
            q=split_conformal_qhat(pool.V_test_kN.values[cal]-vcal,0.10)
            ci=coverage(pool.V_test_kN.values[ind]-vind,q); ce=coverage(big.V_test_kN.values-vbig,q)
            M=big.V_test_kN.values/vbig
            rows.append(dict(model=mname,interp=ci,extrap=ce,bias=M.mean(),cov=M.std()/M.mean(),unsafe=np.mean(big.V_test_kN.values<vbig)))
    r=pd.DataFrame(rows).groupby("model").mean(numeric_only=True)
    print(f"\n#### hold out {label} (>= {cut:.2f}; train n={len(pool)}, extrap n={len(big)}) ####")
    print(r.round(3).to_string())
    return r

def main():
    df=pd.read_csv(PROC/"column_clean.csv")
    print(f"== REAL RC column extrapolation (n={len(df)}) ==")
    rs=run_axis(df,"d1","SIZE (effective depth d1)")
    ra=run_axis(df,"n_ax","AXIAL-LOAD RATIO n=P/(f'c Ag)  [the column-specific axis]")
    print("\n>> KEY: on the AXIAL-LOAD axis, ML extrap bias/unsafe show whether ML over-predicts")
    print("   high-axial-load column capacity (the P-M interaction it cannot extrapolate).")
    pd.concat({"size":rs,"axial":ra}).to_csv(PROC/"column_extrap.csv")
    print("saved -> column_extrap.csv")

if __name__=="__main__": main()
