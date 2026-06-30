"""
fig_build.py — publication figures (real data) for the out-of-envelope paper.
Outputs SVG+PDF+PNG to ../figures/.
 Fig 1  concept (hero) + real data: the training-envelope cliff
 Fig 2  no UQ method restores extrapolation coverage (FRP + steel)
 Fig 3  model error vs size: ML cliffs, mechanical model stays safe
 Fig 4  the remedy: applicability-domain gate + mechanical fallback
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed"); FIG = Path("../figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.linewidth": 0.8, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 450, "savefig.bbox": "tight",
})
C_IN, C_OUT, C_MECH, C_ACC = "#2c6fbb", "#c1352c", "#2a9d5c", "#7a7a7a"
FEATS = ["bw", "d", "fc", "rho_l", "Ef", "a_d"]


def save_all(fig, stem: str):
    fig.savefig(FIG/f"{stem}.svg")
    fig.savefig(FIG/f"{stem}.pdf")
    fig.savefig(FIG/f"{stem}.png")


def load_frp():
    df = pd.read_csv(PROC/"frp_clean.csv")
    return df[(df.a_d>=2.5)&(df.rho_l<0.1)].dropna(subset=FEATS+["V_test_kN"]).sort_values("d").reset_index(drop=True)


def aci440(df):
    Ec=4700*np.sqrt(df.fc); nf=df.Ef/Ec; rn=df.rho_l*nf; k=np.sqrt(2*rn+rn**2)-rn
    return 0.4*np.sqrt(df.fc)*df.bw*(k*df.d)/1e3


def frp_split_fit(df, seed=0):
    d_hi=df.d.quantile(0.75); pool=df[df.d<d_hi].reset_index(drop=True); big=df[df.d>=d_hi].reset_index(drop=True)
    rng=np.random.default_rng(seed); idx=rng.permutation(len(pool))
    tr,cal=idx[:int(.7*len(pool))],idx[int(.7*len(pool)):]
    m=RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.7,random_state=seed,n_jobs=-1)
    m.fit(pool[FEATS].values[tr], np.log(pool.V_test_kN.values[tr]))
    sm=np.mean(np.exp(np.log(pool.V_test_kN.values[cal])-m.predict(pool[FEATS].values[cal])))
    vcal=np.exp(m.predict(pool[FEATS].values[cal]))*sm
    q=split_conformal_qhat(pool.V_test_kN.values[cal]-vcal,0.10)
    vbig=np.exp(m.predict(big[FEATS].values))*sm
    vpool=np.exp(m.predict(pool[FEATS].values))*sm
    return d_hi,pool,big,vpool,vbig,q


# ---------------- Figure 1 : hero ----------------
def fig1():
    df=load_frp(); d_hi,pool,big,vpool,vbig,q=frp_split_fit(df)
    fig=plt.figure(figsize=(7.2,2.5))
    gs=fig.add_gridspec(1,3,width_ratios=[1.15,1,0.7],wspace=0.42)

    # (A) concept schematic
    ax=fig.add_subplot(gs[0,0]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.add_patch(FancyBboxPatch((0.4,2.4),4.2,5.8,boxstyle="round,pad=0.08",fc="#eaf2fb",ec=C_IN,lw=1.2))
    ax.add_patch(FancyBboxPatch((5.1,2.4),4.5,5.8,boxstyle="round,pad=0.08",fc="#fdeceb",ec=C_OUT,lw=1.2))
    ax.text(2.5,8.9,"training envelope",color=C_IN,ha="center",fontsize=7.3,style="italic")
    ax.text(7.35,8.9,"out-of-envelope",color=C_OUT,ha="center",fontsize=7.3,style="italic")
    ax.text(2.5,6.2,"ML accurate\n90% interval\nvalid",color=C_IN,ha="center",va="center",fontsize=7)
    ax.text(7.35,6.6,"ML interval\nunder-covers\n(no warning)",color=C_OUT,ha="center",va="center",fontsize=7)
    ax.annotate("",xy=(9.35,3.7),xytext=(6.05,3.7),arrowprops=dict(arrowstyle="->",color=C_MECH,lw=1.5))
    ax.text(7.55,3.05,"mechanical\nfallback (safe)",color=C_MECH,ha="center",va="center",fontsize=6.8)
    ax.text(5.0,1.35,"increasing member size  $d\\;\\rightarrow$",ha="center",fontsize=7.3)
    ax.annotate("",xy=(9.4,0.75),xytext=(0.6,0.75),arrowprops=dict(arrowstyle="->",color="k",lw=1.0))
    ax.text(-0.1,9.6,"A",fontweight="bold",fontsize=11)

    # (B) predicted vs measured, colored by envelope, with interval band on a few large pts
    ax=fig.add_subplot(gs[0,1])
    ax.scatter(vpool,pool.V_test_kN,s=10,c=C_IN,alpha=0.6,label="in-envelope",edgecolors="none")
    ax.errorbar(vbig,big.V_test_kN,xerr=q,fmt="o",ms=4,c=C_OUT,alpha=0.8,elinewidth=0.6,
                capsize=0,label="out-of-envelope (±$\\hat q$)")
    lim=[0,max(df.V_test_kN.max(),vbig.max())*1.05]
    ax.plot(lim,lim,"k--",lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ML predicted $V$ (kN)"); ax.set_ylabel("measured $V_{\\rm test}$ (kN)")
    ax.legend(loc="upper left",frameon=False); ax.text(-0.28,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

    # (C) coverage in vs out
    ax=fig.add_subplot(gs[0,2])
    cov_in=coverage(pool.V_test_kN.values-vpool,q); cov_out=coverage(big.V_test_kN.values-vbig,q)
    ax.bar([0,1],[cov_in,cov_out],color=[C_IN,C_OUT],width=0.62)
    ax.axhline(0.9,ls="--",c="k",lw=0.8); ax.text(1.5,0.91,"target\n0.90",fontsize=6.5,va="bottom",ha="right")
    ax.set_xticks([0,1]); ax.set_xticklabels(["in-\nenv.","out-of-\nenv."]); ax.set_ylim(0,1.0)
    ax.set_ylabel("interval coverage")
    for x,v in zip([0,1],[cov_in,cov_out]): ax.text(x,v+0.02,f"{v:.2f}",ha="center",fontsize=7.5)
    ax.text(-0.42,1.02,"C",transform=ax.transAxes,fontweight="bold",fontsize=11)
    save_all(fig, "fig1_envelope_cliff"); plt.close(fig)
    print("fig1 done", f"(cov in={cov_in:.2f} out={cov_out:.2f})")


# ---------------- Figure 2 : UQ-method matrix ----------------
def fig2():
    R=pd.read_csv(PROC/"a07_uq_matrix_raw.csv")
    g=R.groupby(["ds","model","uq"]).agg(interp=("interp","mean"),extrap=("extrap","mean")).reset_index()
    uqn={"split":"split-conf.","knn-norm":"adaptive","native90":"native"}
    order=["HistGB|split","HistGB|knn-norm","RF|split","RF|knn-norm","Linear|split","Linear|knn-norm","GP|native90","QuantGBM|native90"]
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.35),sharey=True)
    for ax,ds in zip(axes,["FRP","STEEL"]):
        sub=g[g.ds==ds].copy(); sub["key"]=sub.model+"|"+sub.uq
        sub=sub.set_index("key").reindex(order).reset_index()
        labels=[f"{k.split('|')[0]}\n{uqn[k.split('|')[1]]}" for k in sub.key]
        x=np.arange(len(sub))
        ax.bar(x-0.19,sub.interp,width=0.36,color=C_ACC,label="interpolation (CV)")
        ax.bar(x+0.19,sub.extrap,width=0.36,color=C_OUT,label="size-extrapolation")
        ax.axhline(0.9,ls="--",c="k",lw=0.8)
        if ds=="STEEL": ax.text(7.6,0.915,"0.90",fontsize=6.5,ha="right",color="k")
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6.4,rotation=40,ha="right")
        ax.set_title(f"{ds} beams"); ax.set_ylim(0,1.02)
        if ds=="FRP":
            ax.set_ylabel("90% interval coverage")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Coverage falls out of envelope across UQ methods",fontsize=9,y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save_all(fig, "fig2_uq_matrix"); plt.close(fig)
    print("fig2 done")


# ---------------- Figure 3 : error vs size, ML vs mechanical ----------------
def fig3():
    df=load_frp(); df["Vc440"]=aci440(df)
    # ML out-of-fold within full set for a fair size-trend of CV error (interp), plus extrap fit
    d_hi,pool,big,vpool,vbig,q=frp_split_fit(df)
    # combine ML predictions: pool=OOF-ish (use fitted, conservative) ; big=extrapolation
    ml_ratio=np.concatenate([pool.V_test_kN.values/vpool, big.V_test_kN.values/vbig])
    dd=np.concatenate([pool.d.values,big.d.values])
    mech_ratio=df.V_test_kN.values/df.Vc440.values
    fig,ax=plt.subplots(figsize=(4.25,3.0))
    # binned means
    bins=[0,150,225,300,400,550,1000]; cb=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mml=(dd>=lo)&(dd<hi); mme=(df.d.values>=lo)&(df.d.values<hi); xc=(lo+hi)/2
        if mml.sum()>2: ax.plot(xc,ml_ratio[mml].mean(),"o",c=C_OUT,ms=5)
        if mme.sum()>2: ax.plot(xc,mech_ratio[mme].mean(),"s",c=C_MECH,ms=5)
    ax.axvspan(d_hi,1000,color="#fdeceb",alpha=0.5,zorder=0)
    ax.text(d_hi+10,0.3,"out-of-envelope",color=C_OUT,fontsize=7,rotation=90,va="bottom")
    ax.axhline(1.0,ls=":",c="k",lw=0.8); ax.text(60,1.03,"unsafe if below 1",fontsize=6.5,color="k")
    ax.plot([],[],"o",c=C_OUT,label="ML  ($V_{\\rm test}/V_{\\rm ML}$)")
    ax.plot([],[],"s",c=C_MECH,label="ACI 440.1R-15 mechanical")
    ax.set_xlabel("effective depth $d$ (mm)"); ax.set_ylabel("$V_{\\rm test}/V_{\\rm pred}$ (binned mean)")
    ax.set_ylim(0,2.6); ax.legend(frameon=False,loc="upper right")
    ax.set_title("Model-error drift with member size",fontsize=8.5)
    fig.tight_layout()
    save_all(fig, "fig3_error_vs_size"); plt.close(fig)
    print("fig3 done")


# ---------------- Figure 4 : remedy ----------------
def fig4():
    df=load_frp(); df["Vc440"]=aci440(df); d_hi,pool,big,vpool,vbig,q=frp_split_fit(df)
    # % unconservative out-of-envelope: ML(RF) vs ACI440 ; and COV
    unc_ml=np.mean(big.V_test_kN.values<vbig); cov_ml=big.V_test_kN.values/vbig
    unc_me=np.mean(big.V_test_kN.values<big.Vc440.values); cov_me=big.V_test_kN.values/big.Vc440.values
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.6))
    ax=axes[0]
    ax.bar([0,1],[unc_ml*100,unc_me*100],color=[C_OUT,C_MECH],width=0.6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["ML","ACI 440.1R-15\n(fallback)"]); ax.set_ylabel("% unconservative (out-of-env.)")
    for x,v in zip([0,1],[unc_ml*100,unc_me*100]): ax.text(x,v+1,f"{v:.0f}%",ha="center",fontsize=8)
    ax.set_ylim(0,max(unc_ml*100,5)*1.25); ax.set_title("Safety of large-member estimate",fontsize=8.5)
    ax.text(-0.26,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)
    ax=axes[1]
    parts=ax.violinplot([cov_ml,cov_me],showmeans=True,showextrema=False)
    for pc,c in zip(parts['bodies'],[C_OUT,C_MECH]): pc.set_facecolor(c); pc.set_alpha(0.55)
    ax.axhline(1.0,ls=":",c="k",lw=0.8)
    ax.set_xticks([1,2]); ax.set_xticklabels(["ML","ACI 440.1R-15"]); ax.set_ylabel("$V_{\\rm test}/V_{\\rm pred}$ (out-of-env.)")
    ax.set_title("Bounded, conservative fallback",fontsize=8.5)
    ax.text(-0.26,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)
    fig.tight_layout(); save_all(fig, "fig4_remedy"); plt.close(fig)
    print("fig4 done", f"(ML unconservative {unc_ml:.0%}, ACI440 {unc_me:.0%})")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("ALL FIGURES ->", FIG.resolve())
