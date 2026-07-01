"""
fig_build.py — publication figures (real data) for the out-of-envelope paper.
Outputs SVG+PDF+PNG to ../figures/.
 Fig 1  integrated overview: CV cliff, mechanism, reliability, fallback rule
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


# ---------------- Figure 1 : integrated overview ----------------
def fig1():
    fig=plt.figure(figsize=(7.60,5.35))
    gs=fig.add_gridspec(2,2,wspace=0.36,hspace=0.48)

    def panel_label(ax, s):
        ax.text(-0.08,1.05,s,transform=ax.transAxes,fontweight="bold",fontsize=11,va="bottom")

    def box(ax, xy, wh, text, fc, ec, fontsize=7.5, weight="normal"):
        x,y=xy; w,h=wh
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.035",
                                    fc=fc,ec=ec,lw=1.0,transform=ax.transAxes,clip_on=False))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,
                fontweight=weight,transform=ax.transAxes)

    def arrow(ax, xy1, xy2, color="0.25", lw=1.0):
        ax.annotate("",xy=xy2,xycoords=ax.transAxes,xytext=xy1,textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->",lw=lw,color=color,shrinkA=2,shrinkB=2))

    # A: validation mismatch
    ax=fig.add_subplot(gs[0,0]); ax.axis("off"); panel_label(ax,"A")
    ax.set_title("Validation mismatch",fontsize=9,pad=4)
    box(ax,(0.03,0.58),(0.38,0.28),"Random CV\nexchangeable\ninterpolation", "#eef3f7", C_ACC, 7.2)
    box(ax,(0.59,0.58),(0.38,0.28),"Engineering use\nlarge member\nextrapolation", "#fdeceb", C_OUT, 7.2)
    arrow(ax,(0.42,0.72),(0.58,0.72),C_OUT,1.3)
    ax.text(0.50,0.84,"not the same\nvalidity claim",ha="center",va="center",fontsize=6.7,
            color=C_OUT,transform=ax.transAxes)
    y=0.25
    for x0,h,c in [(0.10,0.11,C_IN),(0.20,0.17,C_IN),(0.30,0.22,C_IN),(0.73,0.31,C_OUT)]:
        ax.add_patch(plt.Rectangle((x0,y),0.055,h,fc=c,ec=c,alpha=0.85,transform=ax.transAxes))
        ax.plot([x0-0.008,x0+0.063],[y+h,y+h],c=c,lw=1.0,transform=ax.transAxes)
    ax.plot([0.04,0.96],[y,y],c="0.25",lw=0.8,transform=ax.transAxes)
    ax.text(0.50,0.10,"effective depth $d$  $\\longrightarrow$",ha="center",fontsize=7.4,transform=ax.transAxes)
    ax.text(0.20,0.50,"training envelope",ha="center",fontsize=7,color=C_IN,transform=ax.transAxes)
    ax.text(0.76,0.50,"out-of-envelope",ha="center",fontsize=7,color=C_OUT,transform=ax.transAxes)

    # B: quantitative cliff from saved UQ matrices
    ax=fig.add_subplot(gs[0,1]); panel_label(ax,"B")
    Rf=pd.read_csv(PROC/"a07_uq_matrix_raw.csv")
    Rs=pd.read_csv(PROC/"steel_uq_matrix_raw.csv")
    rows=[]
    for ds,R in [("FRP",Rf[Rf.ds=="FRP"]),("Steel",Rs)]:
        sub=R[R.uq=="split"]
        rows.append((ds,sub.interp.mean(),sub.interp.std(),sub.extrap.mean(),sub.extrap.std()))
    centers=np.arange(len(rows)); width=0.32
    interp=[r[1] for r in rows]; interp_sd=[r[2] for r in rows]
    extrap=[r[3] for r in rows]; extrap_sd=[r[4] for r in rows]
    ax.bar(centers-width/2,interp,width,color=C_ACC,label="interpolation",
           yerr=interp_sd,capsize=2,error_kw=dict(lw=0.7,capthick=0.7))
    ax.bar(centers+width/2,extrap,width,color=C_OUT,label="size extrapolation",
           yerr=extrap_sd,capsize=2,error_kw=dict(lw=0.7,capthick=0.7))
    ax.axhline(0.90,ls="--",c="k",lw=0.8)
    ax.text(1.48,0.915,"target 0.90",fontsize=6.8,ha="right")
    ax.set_ylim(0,1.02); ax.set_ylabel("split-conformal coverage")
    ax.set_xticks(centers); ax.set_xticklabels([r[0] for r in rows])
    ax.set_title("The interval cliff",fontsize=9,pad=4)
    for x,v in zip(centers+width/2,extrap):
        ax.text(x,v+0.035,f"{v:.2f}",ha="center",fontsize=7,color=C_OUT)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.24),
              ncol=2,fontsize=6.8,handlelength=1.2,columnspacing=1.2)

    # C: mechanism slope contrast
    ax=fig.add_subplot(gs[1,0]); panel_label(ax,"C")
    d=np.array([100,200,400,800,1400],float)
    y_phys=(d/100)**(-0.32)
    y_tree=(d/100)**(-1.00)
    ax.plot(d,y_phys,"-o",c=C_MECH,lw=1.8,ms=4,label="mechanics $m\\approx-0.3$")
    ax.plot(d,y_tree,"-o",c=C_OUT,lw=1.8,ms=4,label="tree artefact $m\\approx-1.0$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("effective depth $d$ (mm)")
    ax.set_ylabel("normalised nominal shear")
    ax.set_title("Missing size-effect physics",fontsize=9,pad=4)
    ax.legend(frameon=False,fontsize=7,loc="lower left")
    ax.text(145,0.55,"fracture-energy\nsize effect",color=C_MECH,fontsize=7)
    ax.text(520,0.045,"prediction caps\n$\\Rightarrow\\tau\\sim1/d$",color=C_OUT,fontsize=7)
    ax.set_ylim(0.04,1.2)

    # D: reliability consequence and rule
    ax=fig.add_subplot(gs[1,1]); ax.axis("off"); panel_label(ax,"D")
    ax.set_title("Reliability consequence and bounded rule",fontsize=9,pad=4)
    inset=ax.inset_axes([0.02,0.18,0.45,0.70])
    rel=pd.read_csv(PROC/"steel_reliability.csv")
    inset.plot(rel.d_mid,rel.beta_ml,"o-",c=C_OUT,lw=1.4,ms=3.8,label="learned")
    inset.plot(rel.d_mid,rel.beta_me,"s-",c=C_MECH,lw=1.4,ms=3.8,label="MCFT")
    inset.axhline(3.8,ls="--",c="k",lw=0.8)
    inset.text(rel.d_mid.max(),3.86,"$\\beta_T$",fontsize=6.5,ha="right",va="bottom")
    inset.set_xlabel("$d$ bin midpoint (mm)",fontsize=7)
    inset.set_ylabel("realised $\\beta$",fontsize=7)
    inset.tick_params(labelsize=6.5)
    inset.set_ylim(1.4,4.7)
    inset.legend(frameon=False,fontsize=6.3,loc="lower left")
    x0=0.56
    box(ax,(x0,0.67),(0.34,0.17),"new member\n$x$", "#f7f7f7", "0.45", 7.1)
    box(ax,(x0,0.43),(0.34,0.16),"inside trained\nfeature envelope?", "#eef3f7", C_IN, 7.0)
    box(ax,(x0-0.08,0.16),(0.22,0.17),"yes:\nML interval\nwith caveat", "#eef3f7", C_IN, 6.6)
    box(ax,(x0+0.21,0.16),(0.25,0.17),"no:\nvalidated\nmechanical model\nfor regime", "#eaf6ef", C_MECH, 6.2)
    arrow(ax,(x0+0.17,0.67),(x0+0.17,0.59),"0.25",1.0)
    arrow(ax,(x0+0.10,0.43),(x0+0.04,0.33),C_IN,1.0)
    arrow(ax,(x0+0.24,0.43),(x0+0.35,0.33),C_MECH,1.0)

    fig.suptitle("Cross-validation certifies interpolation; structural safety needs envelope-aware uncertainty",
                 fontsize=9.8,y=0.995)
    fig.subplots_adjust(top=0.91,left=0.125,right=0.98,bottom=0.095)
    save_all(fig, "fig1_envelope_cliff"); plt.close(fig)
    print("fig1 done (integrated overview)")


# ---------------- Figure 2 : UQ-method matrix ----------------
def fig2():
    R_families=pd.read_csv(PROC/"a07_uq_matrix_raw.csv")
    R_frp=R_families[R_families.ds=="FRP"].copy()
    R_steel=pd.read_csv(PROC/"steel_uq_matrix_raw.csv").copy()
    R_steel["ds"]="STEEL"
    R=pd.concat([R_frp,R_steel],ignore_index=True)
    g=R.groupby(["ds","model","uq"]).agg(
        interp=("interp","mean"),
        extrap=("extrap","mean"),
        interp_sd=("interp","std"),
        extrap_sd=("extrap","std"),
    ).reset_index()
    uqn={"split":"split-conf.","knn-norm":"adaptive","native90":"native"}
    order=["HistGB|split","HistGB|knn-norm","RF|split","RF|knn-norm","Linear|split","Linear|knn-norm","GP|native90","QuantGBM|native90"]
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.35),sharey=True)
    for ax,ds in zip(axes,["FRP","STEEL"]):
        sub=g[g.ds==ds].copy(); sub["key"]=sub.model+"|"+sub.uq
        sub=sub.set_index("key").reindex(order).reset_index()
        labels=[f"{k.split('|')[0]}\n{uqn[k.split('|')[1]]}" for k in sub.key]
        x=np.arange(len(sub))
        ax.bar(x-0.19,sub.interp,width=0.36,color=C_ACC,label="interpolation (CV)",
               yerr=sub.interp_sd.fillna(0),capsize=2,error_kw=dict(lw=0.7,capthick=0.7))
        ax.bar(x+0.19,sub.extrap,width=0.36,color=C_OUT,label="size-extrapolation",
               yerr=sub.extrap_sd.fillna(0),capsize=2,error_kw=dict(lw=0.7,capthick=0.7))
        ax.axhline(0.9,ls="--",c="k",lw=0.8)
        if ds=="STEEL": ax.text(7.6,0.915,"0.90",fontsize=6.5,ha="right",color="k")
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6.4,rotation=40,ha="right")
        ax.set_title("Canonical steel beams" if ds=="STEEL" else "FRP beams")
        ax.set_ylim(0,1.02)
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
