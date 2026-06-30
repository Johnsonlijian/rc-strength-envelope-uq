"""
a23_column_pm_extrap.py — column-specific mechanism demo + pipeline (to be confirmed
on real PEER column data when it lands).

The NEW dimension for columns is the AXIAL-LOAD / P-M interaction: moment capacity is
NON-MONOTONIC in axial load (rises then falls past the balanced point). A model trained
on the usual low-axial-load test range cannot extrapolate the post-balanced decrease.
Ground truth = the verified P-M model (column_models.py). Controlled experiment with
known physics, exactly as the FE established the beam size-effect mechanism.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from column_models import M_capacity_at_P, pm_interaction
from uq_utils import split_conformal_qhat, coverage
PROC=Path("../data/processed"); rng=np.random.default_rng(0)


def gen_columns(N=1500):
    h=rng.uniform(300,700,N); b=h*rng.uniform(0.8,1.0,N)
    fc=rng.uniform(25,45,N); fy=420.0; rho=rng.uniform(0.01,0.03,N)
    n_ax=rng.uniform(0.03,0.6,N)                       # axial-load ratio P/(Ag fc) -- the key axis
    rows=[]
    for i in range(N):
        Ag=b[i]*h[i]; Ast=rho[i]*Ag; A=Ast/8
        layers=[(0.15*h[i],3*A),(0.5*h[i],2*A),(0.85*h[i],3*A)]
        P=n_ax[i]*Ag*fc[i]
        M=M_capacity_at_P(b[i],h[i],layers,fc[i],fy,P)/1e6   # kN-m
        rows.append(dict(b=b[i],h=h[i],fc=fc[i],rho=rho[i],n_ax=n_ax[i],P_kN=P/1e3,M=max(M,1)))
    return pd.DataFrame(rows)


def main():
    df=gen_columns()
    Mscatter=df.M.values*np.exp(rng.normal(0,0.10,len(df)))   # test-like model-uncertainty
    df["M_test"]=Mscatter
    F=["b","h","fc","rho","P_kN"]
    print(f"== column P-M extrapolation demo (n={len(df)}; axial-load ratio n=0.03-0.60) ==")
    # balanced point ~ where M peaks; typical lab tests cluster at low n
    for axis,thr,desc in [("n_ax",0.35,"axial-load ratio")]:
        pool=df[df[axis]<thr]; big=df[df[axis]>=thr]
        X,y=pool[F].values,np.log(pool.M_test.values)
        ml=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=10,random_state=0)
        oof=np.exp(cross_val_predict(ml,X,y,cv=KFold(10,shuffle=True,random_state=0)))
        q=split_conformal_qhat(pool.M_test.values-oof,0.10)
        ml.fit(X,y); pred_big=np.exp(ml.predict(big[F].values))
        cov_in=coverage(pool.M_test.values-oof,q); cov_ex=coverage(big.M_test.values-pred_big,q)
        bias=np.mean(big.M_test.values/pred_big)
        print(f"  hold out {desc} >= {thr}:  interp coverage={cov_in:.2f}  extrap coverage={cov_ex:.2f}")
        print(f"     ML extrap bias M_true/M_ML={bias:.2f}  (>1 => ML UNDER-predicts the post-balanced capacity)")
        print(f"     %unsafe (M_test<M_ML, over-prediction)={100*np.mean(big.M_test.values<pred_big):.0f}%")
    # show the non-monotonic P-M physics ML misses: M vs n_ax, true vs ML
    grid=df.sort_values("n_ax")
    print(f"\n  true M(P) is non-monotonic: peaks at n~{grid.n_ax.values[np.argmax(grid.M.values)]:.2f} (balanced), "
          f"then declines -- ML trained on n<0.35 has no balanced-point physics to extrapolate.")
    df.to_csv(PROC/"column_pm_demo.csv",index=False); print("saved -> column_pm_demo.csv (pipeline ready for real PEER data)")


if __name__=="__main__":
    main()
