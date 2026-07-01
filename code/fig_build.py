"""
fig_build.py — publication figures (real data) for the out-of-envelope paper.
Outputs SVG+PDF+PNG to ../figures/.
 Fig 1  integrated overview: CV cliff, mechanism, reliability, fallback rule
 Fig 2  no UQ method restores extrapolation coverage (FRP + steel)
 Fig 5  model error vs size: ML cliff, range-bounded mechanical check
 Fig 6  the remedy: applicability-domain gate + mechanical fallback
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


def save_all(fig, stem: str, aliases=()):
    for name in (stem, *aliases):
        fig.savefig(FIG/f"{name}.svg")
        fig.savefig(FIG/f"{name}.pdf")
        fig.savefig(FIG/f"{name}.png")


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
    fig=plt.figure(figsize=(7.55,6.25))
    gs=fig.add_gridspec(2,2,wspace=0.34,hspace=0.50)

    def panel_label(ax, s):
        ax.text(-0.10,1.04,s,transform=ax.transAxes,fontweight="bold",fontsize=11,va="bottom")

    def clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3,width=0.8)

    # A: validation mismatch
    ax=fig.add_subplot(gs[0,0]); ax.axis("off"); panel_label(ax,"A")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Validation domain mismatch",fontsize=9,pad=6)
    ax.annotate("",xy=(0.92,0.20),xytext=(0.08,0.20),
                arrowprops=dict(arrowstyle="->",lw=0.9,color="0.25"))
    ax.text(0.50,0.08,"effective depth / member size $d$",ha="center",fontsize=7.4)
    ax.add_patch(plt.Rectangle((0.12,0.31),0.43,0.15,fc=C_IN,ec=C_IN,lw=0.9,alpha=0.12))
    ax.add_patch(plt.Rectangle((0.70,0.31),0.17,0.15,fc=C_OUT,ec=C_OUT,lw=0.9,alpha=0.14))
    for x,h in [(0.19,0.09),(0.29,0.13),(0.39,0.18),(0.49,0.21)]:
        ax.add_patch(plt.Rectangle((x,0.22),0.035,h,fc=C_IN,ec=C_IN,alpha=0.80))
    ax.add_patch(plt.Rectangle((0.76,0.22),0.035,0.28,fc=C_OUT,ec=C_OUT,alpha=0.85))
    ax.text(0.335,0.52,"trained feature envelope",ha="center",fontsize=7.1,color=C_IN)
    ax.text(0.785,0.52,"deployment\nquery",ha="center",fontsize=7.1,color=C_OUT)
    ax.text(0.12,0.86,"Random CV",ha="left",fontsize=8.2,fontweight="bold")
    ax.text(0.12,0.75,"validates exchangeable\ninterpolation only",ha="left",va="top",fontsize=7.3)
    ax.text(0.62,0.86,"Engineering use",ha="left",fontsize=8.2,fontweight="bold")
    ax.text(0.62,0.75,"often asks beyond\nthe training envelope",ha="left",va="top",fontsize=7.3)
    ax.annotate("",xy=(0.68,0.40),xytext=(0.56,0.40),
                arrowprops=dict(arrowstyle="->",lw=1.0,color=C_OUT))
    ax.text(0.62,0.49,"not certified",ha="center",va="center",fontsize=6.9,color=C_OUT)

    # B: quantitative cliff from saved UQ matrices
    ax=fig.add_subplot(gs[0,1]); panel_label(ax,"B")
    clean_axes(ax)
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
    ax.text(1.42,0.925,"target 0.90",fontsize=6.8,ha="right")
    ax.set_ylim(0,1.04); ax.set_ylabel("split-conformal coverage")
    ax.set_xticks(centers); ax.set_xticklabels([r[0] for r in rows])
    ax.set_title("Interval coverage collapses out of envelope",fontsize=9,pad=6)
    for x,v in zip(centers+width/2,extrap):
        ax.text(x,v+0.045,f"{v:.2f}",ha="center",fontsize=7.1,color=C_OUT)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.24),
              ncol=2,fontsize=6.8,handlelength=1.2,columnspacing=1.2)

    # C: mechanism slope contrast
    ax=fig.add_subplot(gs[1,0]); panel_label(ax,"C")
    clean_axes(ax)
    d=np.array([100,200,400,800,1400],float)
    y_phys=(d/100)**(-0.32)
    y_tree=(d/100)**(-1.00)
    ax.plot(d,y_phys,"-o",c=C_MECH,lw=1.7,ms=4.2)
    ax.plot(d,y_tree,"-o",c=C_OUT,lw=1.7,ms=4.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("effective depth $d$ (mm)")
    ax.set_ylabel("normalised nominal shear")
    ax.set_title("Tree extrapolation misses size-effect physics",fontsize=9,pad=6)
    ax.text(760,0.39,"mechanics\n$m\\approx-0.3$",color=C_MECH,fontsize=7.2,
            bbox=dict(fc="white",ec="none",alpha=0.82,pad=1.5))
    ax.text(420,0.078,"tree artefact\n$m\\approx-1.0$",color=C_OUT,fontsize=7.2,
            bbox=dict(fc="white",ec="none",alpha=0.82,pad=1.5))
    ax.text(145,0.58,"fracture-energy\nsize effect",color=C_MECH,fontsize=6.8)
    ax.text(840,0.052,"prediction caps\n$\\tau\\propto1/d$",color=C_OUT,fontsize=6.8,ha="center")
    ax.set_xlim(80,1700); ax.set_ylim(0.04,1.25)

    # D: reliability consequence and rule
    ax=fig.add_subplot(gs[1,1]); panel_label(ax,"D")
    clean_axes(ax)
    ax.set_title("Reliability drops below target",fontsize=9,pad=6)
    rel=pd.read_csv(PROC/"steel_reliability.csv")
    ax.axhspan(1.35,3.8,fc=C_OUT,alpha=0.07,zorder=0)
    ax.plot(rel.d_mid,rel.beta_ml,"o-",c=C_OUT,lw=1.7,ms=4.2)
    ax.plot(rel.d_mid,rel.beta_me,"s-",c=C_MECH,lw=1.7,ms=4.2)
    ax.axhline(3.8,ls="--",c="k",lw=0.8)
    ax.text(910,3.87,"target $\\beta_T=3.8$",fontsize=6.9,ha="right",va="bottom")
    ax.text(565,2.03,"learned interval",color=C_OUT,fontsize=7.2,
            bbox=dict(fc="white",ec="none",alpha=0.85,pad=1.5))
    ax.text(560,4.42,"validated mechanics",color=C_MECH,fontsize=7.2,
            bbox=dict(fc="white",ec="none",alpha=0.85,pad=1.5))
    ax.annotate("$\\beta\\approx1.8$",xy=(525,1.82),xytext=(380,1.54),
                color=C_OUT,fontsize=7.2,
                arrowprops=dict(arrowstyle="->",lw=0.8,color=C_OUT))
    ax.set_xlabel("$d$ bin midpoint (mm)")
    ax.set_ylabel("realised reliability index $\\beta$")
    ax.set_ylim(1.35,4.75); ax.set_xlim(40,940)

    fig.suptitle("Random cross-validation is an interpolation certificate, not a deployment-safety certificate",
                 fontsize=10.0,y=0.992)
    fig.subplots_adjust(top=0.91,left=0.105,right=0.985,bottom=0.095)
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
    if (PROC/"frp_clean.csv").exists():
        df=load_frp(); df["Vc440"]=aci440(df)
        # ML out-of-fold within full set for a fair size-trend of CV error (interp), plus extrap fit
        d_hi,pool,big,vpool,vbig,q=frp_split_fit(df)
        ml_ratio=np.concatenate([pool.V_test_kN.values/vpool, big.V_test_kN.values/vbig])
        dd=np.concatenate([pool.d.values,big.d.values])
        mech_ratio=df.V_test_kN.values/df.Vc440.values
        bins=[0,150,225,300,400,550,1000]
        rows=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            mml=(dd>=lo)&(dd<hi); mme=(df.d.values>=lo)&(df.d.values<hi)
            rows.append({
                "x_mid": (lo+hi)/2,
                "ml_ratio_mean": ml_ratio[mml].mean() if mml.sum()>2 else np.nan,
                "mech_ratio_mean": mech_ratio[mme].mean() if mme.sum()>2 else np.nan,
            })
        agg=pd.DataFrame(rows)
    else:
        agg_path=PROC/"frp_fig3_binned.csv"
        if not agg_path.exists():
            raise FileNotFoundError(f"Missing {agg_path}; run a01_frp_load_eda.py on licensed data or use the public aggregate table.")
        agg=pd.read_csv(agg_path)
        d_hi=float(agg["d_hi"].dropna().iloc[0])
    fig,ax=plt.subplots(figsize=(4.25,3.0))
    for _,row in agg.iterrows():
        xc=row["x_mid"]
        if pd.notna(row["ml_ratio_mean"]): ax.plot(xc,row["ml_ratio_mean"],"o",c=C_OUT,ms=5)
        if pd.notna(row["mech_ratio_mean"]): ax.plot(xc,row["mech_ratio_mean"],"s",c=C_MECH,ms=5)
    ax.axvspan(d_hi,1000,color="#fdeceb",alpha=0.5,zorder=0)
    ax.text(d_hi+10,0.3,"out-of-envelope",color=C_OUT,fontsize=7,rotation=90,va="bottom")
    ax.axhline(1.0,ls=":",c="k",lw=0.8); ax.text(60,1.03,"unsafe if below 1",fontsize=6.5,color="k")
    ax.plot([],[],"o",c=C_OUT,label="ML  ($V_{\\rm test}/V_{\\rm ML}$)")
    ax.plot([],[],"s",c=C_MECH,label="ACI 440.1R-15 mechanical")
    ax.set_xlabel("effective depth $d$ (mm)"); ax.set_ylabel("$V_{\\rm test}/V_{\\rm pred}$ (binned mean)")
    ax.set_ylim(0,2.6); ax.legend(frameon=False,loc="upper right")
    ax.set_title("Model-error drift with member size",fontsize=8.5)
    fig.tight_layout()
    save_all(fig, "fig5_error_vs_size", aliases=("fig3_error_vs_size",)); plt.close(fig)
    print("fig5 done (legacy alias fig3_error_vs_size)")


# ---------------- Figure 4 : remedy ----------------
def fig4():
    if (PROC/"frp_clean.csv").exists():
        df=load_frp(); df["Vc440"]=aci440(df); d_hi,pool,big,vpool,vbig,q=frp_split_fit(df)
        # % unconservative out-of-envelope: ML(RF) vs ACI440 ; and COV
        cov_ml=big.V_test_kN.values/vbig
        cov_me=big.V_test_kN.values/big.Vc440.values
    else:
        ratios_path=PROC/"frp_fig4_remedy_ratios.csv"
        if not ratios_path.exists():
            raise FileNotFoundError(f"Missing {ratios_path}; run a01_frp_load_eda.py on licensed data or use the public derived ratio table.")
        ratios=pd.read_csv(ratios_path)
        cov_ml=ratios.loc[ratios.method=="ML","v_test_over_v_pred"].to_numpy()
        cov_me=ratios.loc[ratios.method=="ACI440","v_test_over_v_pred"].to_numpy()
    unc_ml=np.mean(cov_ml<1.0)
    unc_me=np.mean(cov_me<1.0)
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
    fig.tight_layout(); save_all(fig, "fig6_remedy", aliases=("fig4_remedy",)); plt.close(fig)
    print("fig6 done (legacy alias fig4_remedy)", f"(ML unconservative {unc_ml:.0%}, ACI440 {unc_me:.0%})")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("ALL FIGURES ->", FIG.resolve())
