"""Column figure: dual-axis extrapolation failure + the unsafe axial-load mechanism."""
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
PROC=Path("../data/processed"); FIG=Path("../figures")
plt.rcParams.update({
  "font.family":"sans-serif",
  "font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],
  "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
  "font.size":9,"axes.linewidth":0.8,"xtick.labelsize":8,"ytick.labelsize":8,
  "legend.fontsize":7.5,"savefig.dpi":450,"savefig.bbox":"tight"})
C_IN,C_OUT="#7a7a7a","#c1352c"

ex=pd.read_csv(PROC/"column_extrap.csv")          # index: (axis, model)
ex.columns=[c.strip() for c in ex.columns]
ex=ex.rename(columns={ex.columns[0]:"axis",ex.columns[1]:"model"})
def val(axis,col):
    s=ex[ex.axis==axis][col]; return float(s.mean())
row_path=PROC/"column_with_mech.csv"
d=pd.read_csv(row_path) if row_path.exists() else None

fig=plt.figure(figsize=(7.4,3.0)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.05],wspace=0.34)

# Panel A: coverage interp vs extrap on the two axes + unsafe rate
ax=fig.add_subplot(gs[0,0])
axes=["size","axial"]; lab=["size\n(depth)","axial-load\nratio"]
ci=[val("size","interp"),val("axial","interp")]; ce=[val("size","extrap"),val("axial","extrap")]
x=np.arange(2)
ax.bar(x-0.19,ci,width=0.36,color=C_IN,label="interpolation (CV)")
ax.bar(x+0.19,ce,width=0.36,color=C_OUT,label="size-/axial-extrapolation")
ax.axhline(0.9,ls="--",c="k",lw=0.8); ax.text(1.50,0.935,"0.90",fontsize=6.5,ha="right")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0,1.18); ax.set_ylabel("90% interval coverage")
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.22),fontsize=6.8,ncol=1)
ax.set_title("RC columns (n=234): interval coverage falls out of envelope",fontsize=8.2)
ax.text(-0.2,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)
# value labels: interp (grey) and extrap (red)
for xi,ci_v,ce_v in zip(x,ci,ce):
    ax.text(xi-0.19,ci_v+0.035,f"{ci_v:.2f}",ha="center",va="bottom",fontsize=6.5,color="#555")
    lbl=f"{ce_v:.2f}" if ce_v>0.1 else f"{ce_v:.3f}"
    ax.text(xi+0.19,ce_v+0.012,lbl,ha="center",va="bottom",fontsize=7,color="#8b1a1a",fontweight="bold")

# Panel B: the UNSAFE mechanism -- V_test/V_ML vs axial-load ratio (ML trained on low-n)
ax=fig.add_subplot(gs[0,1])
if d is not None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    F=["b","h","d1","Lc","a_d","fc","fyc","rho_l","rho_t","n_ax"]
    pool=d[d.n_ax<0.30]
    ml=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=8,random_state=0).fit(pool[F].values,np.log(pool.V_test_kN.values))
    sm=np.mean(np.exp(np.log(pool.V_test_kN.values)-ml.predict(pool[F].values)))
    d=d.assign(M=d.V_test_kN.values/(np.exp(ml.predict(d[F].values))*sm))
    big=d[d.n_ax>=0.30]
    ax.axhspan(0,1,color="#fdecea",alpha=0.6); ax.text(0.02,0.5,"unsafe:\nML over-predicts",fontsize=6.5,color=C_OUT,va="center")
    ax.scatter(d[d.n_ax<0.30].n_ax,d[d.n_ax<0.30].M,s=12,c=C_IN,alpha=0.5,edgecolors="none",label=r"train ($n_{ax}<0.30$)")
    ax.scatter(big.n_ax,big.M,s=16,c=C_OUT,alpha=0.7,edgecolors="none",label=r"extrapolation ($n_{ax}\geq0.30$)")
    ax.axhline(1.0,c="k",lw=0.8,ls=":"); ax.axvline(0.30,c="k",lw=0.6,ls="--")
    ax.set_xlabel(r"axial-load ratio  $n_{ax}=P/(f'_c A_g)$"); ax.set_ylabel(r"$V_{\rm test}/V_{\rm ML}$")
    ax.set_ylim(0,2.6); ax.legend(frameon=False,loc="upper right",fontsize=7)
    unsafe=100*np.mean(big.M<1)
    ax.set_title("High axial-load extrapolation: over-prediction risk",fontsize=8.2)
else:
    agg=pd.read_csv(PROC/"column_mech_by_axial_bin.csv")
    high=agg[agg["axial_load_ratio_bin"].isin(["0.3--0.5","0.5--1.0"])]
    mech_unsafe=float((high["N"]*high["pct_unsafe"]).sum()/high["N"].sum())
    unsafe=100*val("axial","unsafe")
    ax.bar([0,1],[unsafe,mech_unsafe],color=[C_OUT,"#2a9d5c"],width=0.55)
    ax.set_xticks([0,1]); ax.set_xticklabels(["ML high-$n$\nextrap.","mechanical\nhigh-$n$"])
    ax.set_ylim(0,60); ax.set_ylabel("% over-prediction")
    for xi,v in zip([0,1],[unsafe,mech_unsafe]):
        ax.text(xi,v+2,f"{v:.0f}%",ha="center",fontsize=7)
    ax.set_title("High axial-load extrapolation: aggregate public view",fontsize=8.2)
ax.text(-0.2,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

fig.tight_layout()
for stem in ("fig7_columns", "fig8_columns"):
    fig.savefig(FIG/f"{stem}.svg")
    fig.savefig(FIG/f"{stem}.pdf")
    fig.savefig(FIG/f"{stem}.png")
plt.close(fig)
print(f"fig7 columns done (legacy alias fig8_columns). size cov {val('size','interp'):.2f}->{val('size','extrap'):.2f}; "
      f"axial cov {val('axial','interp'):.2f}->{val('axial','extrap'):.2f}; high-n unsafe={unsafe:.0f}%")
