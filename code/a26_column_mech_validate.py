"""
a26_column_mech_validate.py — validate the column mechanical models (P-M flexure +
ACI 318-19 shear) on the REAL column data, and test the fallback: at HIGH axial load
(where ML over-predicts ~48%), does the mechanical model stay reliable/safe?
Column lateral capacity V = min(flexural Mn(P)/a, ACI shear). Model uncertainty M=Vtest/Vpred.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from column_models import M_capacity_at_P, shear_aci318_19_column
PROC=Path("../data/processed")

def V_pred(r):
    b,h,d1,a,fc,fyc,rho_l,P = r.b,r.h,r.d1,max(r.a,0.5*r.d1),r.fc,r.fyc,r.rho_l,r.P*1e3
    Ag=b*h; Ast=rho_l*Ag; As=Ast/2
    layers=[(r.cc if r.cc>0 else 0.1*h, As),(h-(r.cc if r.cc>0 else 0.1*h), As)]
    try: Mn=M_capacity_at_P(b,h,layers,fc,fyc,P)          # N-mm
    except Exception: Mn=np.nan
    Vflex=Mn/a if np.isfinite(Mn) and a>0 else np.nan     # N (lateral force = M/shear-span)
    Vsh=shear_aci318_19_column(b,d1,fc,rho_l,P,Ag)        # N
    return np.nanmin([Vflex,Vsh])/1e3                     # kN

def main():
    df=pd.read_csv(PROC/"column_clean.csv")
    df["Vp"]=[V_pred(r) for r in df.itertuples()]
    d=df.dropna(subset=["Vp"]); d=d[d.Vp>0]
    M=d.V_test_kN.values/d.Vp.values
    print(f"== column mechanical model (flexure+shear) on {len(d)} real columns ==")
    print(f"  ALL: mean M={M.mean():.2f}  COV={M.std()/M.mean():.2f}  %unsafe(M<1)={100*np.mean(M<1):.0f}%")
    # the fallback test: high axial load (n>=0.3) where ML was ~48% unsafe
    hi=d[d.n_ax>=0.30]; Mh=hi.V_test_kN.values/hi.Vp.values
    print(f"\n  HIGH axial load n>=0.30 (n={len(hi)}): mechanical mean M={Mh.mean():.2f} COV={Mh.std()/Mh.mean():.2f} %unsafe={100*np.mean(Mh<1):.0f}%")
    print(f"   -> vs ML there: ~48% unsafe. Mechanical model {'STAYS SAFER' if np.mean(Mh<1)<0.30 else 'also struggles'}.")
    # by axial-load bin
    print("\n  mechanical model uncertainty by axial-load bin:")
    rows = []
    for lo,hival in [(0,0.15),(0.15,0.3),(0.3,0.5),(0.5,1.0)]:
        m=(d.n_ax>=lo)&(d.n_ax<hival); Mm=d.V_test_kN.values[m]/d.Vp.values[m]
        if m.sum()>3:
            mean_m=float(Mm.mean()); cov_m=float(Mm.std()/Mm.mean()); unsafe=float(100*np.mean(Mm<1))
            print(f"    n[{lo},{hival}) n={m.sum():3d}: mean={mean_m:.2f} COV={cov_m:.2f} %unsafe={unsafe:.0f}%")
            rows.append({"axial_load_ratio_bin": f"{lo}--{hival}", "N": int(m.sum()),
                         "mean_M": mean_m, "COV": cov_m, "pct_unsafe": unsafe})
    d.to_csv(PROC/"column_with_mech.csv",index=False); print("\nsaved -> column_with_mech.csv")
    pd.DataFrame(rows).to_csv(PROC/"column_mech_by_axial_bin.csv", index=False)
    print("saved -> column_mech_by_axial_bin.csv")

if __name__=="__main__": main()
