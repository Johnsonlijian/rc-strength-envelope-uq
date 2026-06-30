"""
a18_mechanism_v2.py — definitive mechanism figure on the CANONICAL steel benchmark.
Real steel tests (Zhang 1704, m=-0.41) + first-principles FE (m=-0.23, robust over
384 parametric runs) + 5 field-standard mechanical models (-0.30..-0.38) all agree;
ML extrapolation (m=-1.0, the flat-extrapolation 1/d artefact) is the outlier.
"""
from __future__ import annotations
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from a12_mech_models import MODELS

FIG=Path("../figures"); PROC=Path("../data/processed")
plt.rcParams.update({
  "font.family":"sans-serif",
  "font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],
  "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
  "font.size":9,"axes.linewidth":0.8,"xtick.labelsize":8,"ytick.labelsize":8,
  "legend.fontsize":6.6,"savefig.dpi":450,"savefig.bbox":"tight"})


def save_all(fig, stem: str):
    fig.savefig(FIG/f"{stem}.svg")
    fig.savefig(FIG/f"{stem}.pdf")
    fig.savefig(FIG/f"{stem}.png")


def steel_size_effect():
    df=pd.read_csv(PROC/"steel_zhang_clean.csv"); df=df[df.a_d>=2.5].copy()
    df["v"]=df.Vu_kN*1e3/(df.bw*df.d)
    sm=df[df.d<=np.quantile(df.d,0.33)]
    X=lambda d: np.column_stack([np.log(d.rho),np.log(d.a_d),np.log(d.fc)])
    lr=LinearRegression().fit(X(sm),np.log(sm.v))
    R=df.v.values/np.exp(lr.predict(X(df)))
    big=df.d.values>=np.quantile(df.d.values,0.6)
    m=np.polyfit(np.log(df.d.values[big]),np.log(R[big]),1)[0]
    return df.d.values, R, m


def main():
    fe=json.load(open(PROC/"fe_size_effect.json"))
    fed=np.array([r["d"] for r in fe["rows"]]); fev=np.array([r["v_nom"] for r in fe["rows"]])
    par=json.load(open(PROC/"parametric_fe.json"))
    sd_, Rd, m_steel = steel_size_effect()
    bw,a_d,rho,fc=250.,3.,0.015,30.; ds=np.geomspace(100,2000,30); dref=250.
    def nrm(d,v): return v/np.exp(np.interp(np.log(dref),np.log(d),np.log(v)))

    fig=plt.figure(figsize=(7.4,3.2)); gs=fig.add_gridspec(1,2,width_ratios=[1.35,1],wspace=0.32)
    ax=fig.add_subplot(gs[0,0])
    # real steel binned
    bins=np.geomspace(sd_.min(),sd_.max(),9); idx=np.digitize(sd_,bins)
    bx=[np.median(sd_[idx==i]) for i in range(1,len(bins)) if (idx==i).sum()>5]
    by=[np.median(Rd[idx==i]) for i in range(1,len(bins)) if (idx==i).sum()>5]
    by=np.array(by)/np.interp(np.log(dref),np.log(bx),np.log(np.array(by)),)*0+np.array(by)
    by=np.array(by)/np.exp(np.interp(np.log(dref),np.log(bx),np.log(np.array(by))))
    ax.plot(bx,by,"s",c="k",ms=6,label="real steel tests (n=1177)",zorder=6)
    # mechanical models
    cmap={"MCFT":"#1b7837","CSCT":"#5aae61","fib-MC2010":"#a6dba0","Bazant-Kim":"#762a83","ACI318-19":"#9970ab"}
    for n in cmap:
        v=np.array([MODELS[n](bw,d,a_d,rho,fc)/(bw*d) for d in ds]); ax.plot(ds,nrm(ds,v),"-",c=cmap[n],lw=1.2,label=n,alpha=0.9)
    ax.plot(fed,nrm(fed,fev),"o",c="#2166ac",ms=5,mec="k",mew=0.4,label="FE crack-band sim.",zorder=5)
    ax.plot(ds,(ds/dref)**-1.0,"-",c="#c1352c",lw=2.0,label="ML extrapolation (m=−1.0)")
    ax.plot(ds,(ds/dref)**-0.5,"--",c="k",lw=0.8); ax.text(1900,(1900/dref)**-0.5,"LEFM −1/2",fontsize=6.3,va="top",ha="right")
    # direct slope annotations
    ax.text(1750,(1750/dref)**-1.0*1.07,"m=−1.0",fontsize=6.2,color="#c1352c",va="bottom",ha="right")
    ax.text(1750,(1750/dref)**-0.23*0.85,"m≈−0.3",fontsize=6.2,color="#555",va="top",ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("effective depth $d$ (mm)")
    ax.set_ylabel("normalised nominal shear stress"); ax.set_ylim(0.09,1.7)
    ax.legend(frameon=False,fontsize=6.1,loc="lower left",ncol=1); ax.set_title("Real data, simulation, and theory converge (~−0.3);\nML extrapolation diverges (m≈−1.0)",fontsize=8.2)
    ax.text(-0.15,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)

    ax=fig.add_subplot(gs[0,1])
    items=[("real steel",m_steel,0,"#000000"),("FE sim.",fe["large_slope"],par["slope_sd"],"#2166ac")]
    for n in ["MCFT","CSCT","fib-MC2010","Bazant-Kim","ACI318-19"]:
        v=np.array([MODELS[n](bw,d,a_d,rho,fc)/(bw*d) for d in [300,500,800,1200,2000.]])
        items.append((n.replace("fib-","").replace("Bazant-Kim","Baz-Kim").replace("ACI318-19","ACI-19"),
                      np.polyfit(np.log([300,500,800,1200,2000.]),np.log(v),1)[0],0,"#1b7837"))
    items.append(("ML extrap.",-1.0,0,"#c1352c"))
    names=[i[0] for i in items]; vals=[i[1] for i in items]; errs=[i[2] for i in items]; cols=[i[3] for i in items]
    ax.barh(range(len(names)),vals,xerr=errs,color=cols,error_kw=dict(lw=0.8))
    ax.axvline(-0.5,ls="--",c="k",lw=0.8); ax.text(-0.5,len(names)-0.4,"LEFM",fontsize=6.3,ha="center")
    ax.axvline(0,c="k",lw=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("large-size exponent $m$"); ax.set_title("ML is the outlier",fontsize=8.2)
    ax.text(-0.30,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

    save_all(fig, "fig6_mechanism_convergence"); plt.close(fig)
    print(f"fig6 v2 done. real steel m={m_steel:.2f}; FE {fe['large_slope']:.2f}±{par['slope_sd']:.2f}")


if __name__=="__main__":
    main()
