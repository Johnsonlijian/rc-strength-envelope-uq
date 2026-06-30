"""
a10_fig.py — honest size-effect mechanism figure.
Shows: the data's modest, physical size effect vs ML's spurious, architectural
extrapolation exponent (tree saturation). Real numbers from a10.
"""
from __future__ import annotations
import warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from a10_size_effect_mechanism import (load_unified, strength_fit, v_strength,
                                       fit_sel, large_exponent, ML_FEATS, STRENGTH_VARS)

FIG = Path("../figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.linewidth":0.8,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
    "savefig.dpi":300,"savefig.bbox":"tight"})
C_DATA,C_ML,C_REF="#2c6fbb","#c1352c","#7a7a7a"


def ml_exp(df,name,d_small):
    small=df[df.d<=d_small]; feats=ML_FEATS[name]
    ml=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,
        min_samples_leaf=10,random_state=0).fit(small[feats].values,np.log(small.V.values))
    return np.exp(ml.predict(df[feats].values))/(df.b.values*df.d.values)


def main():
    data=load_unified()
    res={}
    for name,df in data.items():
        d=df.d.values; d_small=np.quantile(d,0.33)
        lr,_=strength_fit(df,name,d_small)
        R=df.v.values/v_strength(df,name,lr)
        m_emp,_=large_exponent(d,R)
        R_ml=ml_exp(df,name,d_small)/v_strength(df,name,lr)
        m_ml,_=large_exponent(d,R_ml)
        res[name]=dict(d=d,R=R,R_ml=R_ml,d_small=d_small,m_emp=m_emp,m_ml=m_ml,lr=lr)

    fig=plt.figure(figsize=(7.2,3.0)); gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=0.32)

    # Panel A: FRP size-effect scatter (log-log), data vs ML extrapolation
    ax=fig.add_subplot(gs[0,0]); r=res["FRP"]; d=r["d"]; R=r["R"]
    ax.scatter(d,R,s=12,c=C_DATA,alpha=0.5,edgecolors="none",label="FRP tests")
    # binned medians
    bins=np.geomspace(d.min(),d.max(),8); idx=np.digitize(d,bins)
    bx=[np.median(d[idx==i]) for i in range(1,len(bins)) if (idx==i).sum()>2]
    by=[np.median(R[idx==i]) for i in range(1,len(bins)) if (idx==i).sum()>2]
    ax.plot(bx,by,"o-",c=C_DATA,ms=5,lw=1.3,label="binned median")
    dd=np.geomspace(d.min(),d.max(),50); dref=150.0; Rref=np.median(R[(d>100)&(d<200)])
    ax.plot(dd,Rref*(dd/dref)**r["m_emp"],"--",c=C_DATA,lw=1.4,label=f"data slope m={r['m_emp']:.2f}")
    ax.plot(dd,Rref*(dd/dref)**r["m_ml"],"-",c=C_ML,lw=1.6,label=f"ML extrapolation m={r['m_ml']:.2f}")
    ax.plot(dd,Rref*(dd/dref)**-0.5,":",c=C_REF,lw=1.0); ax.text(dd[-1],Rref*(dd[-1]/dref)**-0.5,"LEFM −1/2",fontsize=6.5,color=C_REF,va="top")
    ax.axvspan(d.min(),r["d_small"],color="#eef4fb"); ax.text(r["d_small"],ax.get_ylim()[0],"train\n(small)",fontsize=6,va="bottom",ha="right",color=C_DATA)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("effective depth $d$ (mm)")
    ax.set_ylabel(r"$R=v_{\rm test}/v_{\rm strength}$  (size effect)")
    ax.legend(frameon=False,loc="lower left",fontsize=6.6); ax.set_title("FRP: data has a modest, physical size effect",fontsize=8.5)
    ax.text(-0.18,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)

    # Panel B: data vs ML extrapolation exponent across datasets
    ax=fig.add_subplot(gs[0,1]); names=list(res); x=np.arange(len(names))
    ax.bar(x-0.18,[res[n]["m_emp"] for n in names],width=0.36,color=C_DATA,label="data (physical)")
    ax.bar(x+0.18,[res[n]["m_ml"] for n in names],width=0.36,color=C_ML,label="ML extrapolation")
    ax.axhline(-0.5,ls=":",c=C_REF,lw=1.0); ax.text(len(names)-0.5,-0.5,"LEFM",fontsize=6.5,va="bottom",ha="right",color=C_REF)
    ax.axhline(0,c="k",lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("-deep","\ndeep") for n in names],fontsize=7)
    ax.set_ylabel("large-size exponent $m$"); ax.legend(frameon=False,loc="lower left")
    ax.set_title("ML's size response is architectural,\nnot physical",fontsize=8.5)
    ax.text(-0.22,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

    fig.savefig(FIG/"fig5_size_effect_mechanism.pdf"); fig.savefig(FIG/"fig5_size_effect_mechanism.png"); plt.close(fig)
    print("fig5 done:", {n:(round(res[n]['m_emp'],2),round(res[n]['m_ml'],2)) for n in names})


if __name__=="__main__":
    main()
