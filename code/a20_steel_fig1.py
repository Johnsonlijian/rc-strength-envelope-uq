"""fig1 v2 (hero) on the canonical steel benchmark: the training-envelope cliff."""
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from uq_utils import split_conformal_qhat, coverage
PROC=Path("../data/processed"); FIG=Path("../figures")
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
  "axes.linewidth":0.8,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
  "savefig.dpi":300,"savefig.bbox":"tight"})
C_IN,C_OUT,C_MECH="#2c6fbb","#c1352c","#2a9d5c"
F=["bw","d","a_d","rho","fc","ag","fy"]

df=pd.read_csv(PROC/"steel_zhang_clean.csv"); df=df[df.a_d>=2.5].dropna(subset=F+["Vu_kN"]).sort_values("d").reset_index(drop=True)
d_hi=df.d.quantile(0.75); pool=df[df.d<d_hi].reset_index(drop=True); big=df[df.d>=d_hi].reset_index(drop=True)
rng=np.random.default_rng(0); idx=rng.permutation(len(pool)); tr,cal=idx[:int(.7*len(pool))],idx[int(.7*len(pool)):]
m=RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.7,random_state=0,n_jobs=-1)
m.fit(pool[F].values[tr],np.log(pool.Vu_kN.values[tr]))
sm=np.mean(np.exp(np.log(pool.Vu_kN.values[cal])-m.predict(pool[F].values[cal])))
q=split_conformal_qhat(pool.Vu_kN.values[cal]-np.exp(m.predict(pool[F].values[cal]))*sm,0.10)
vpool=np.exp(m.predict(pool[F].values))*sm; vbig=np.exp(m.predict(big[F].values))*sm
cov_in=coverage(pool.Vu_kN.values-vpool,q); cov_out=coverage(big.Vu_kN.values-vbig,q)

fig=plt.figure(figsize=(7.2,2.5)); gs=fig.add_gridspec(1,3,width_ratios=[1.15,1,0.7],wspace=0.42)
ax=fig.add_subplot(gs[0,0]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.add_patch(FancyBboxPatch((0.4,2.4),4.2,5.8,boxstyle="round,pad=0.08",fc="#eaf2fb",ec=C_IN,lw=1.2))
ax.add_patch(FancyBboxPatch((5.1,2.4),4.5,5.8,boxstyle="round,pad=0.08",fc="#fdeceb",ec=C_OUT,lw=1.2))
ax.text(2.5,8.9,"training envelope",color=C_IN,ha="center",fontsize=7.3,style="italic")
ax.text(7.35,8.9,"out-of-envelope",color=C_OUT,ha="center",fontsize=7.3,style="italic")
ax.text(2.5,6.2,"ML accurate\n90% interval\nvalid",color=C_IN,ha="center",va="center",fontsize=7)
ax.text(7.35,6.6,"ML interval\nCOLLAPSES\n(no warning)",color=C_OUT,ha="center",va="center",fontsize=7)
ax.annotate("",xy=(9.35,3.7),xytext=(6.05,3.7),arrowprops=dict(arrowstyle="->",color=C_MECH,lw=1.5))
ax.text(7.55,3.05,"mechanical\nfallback (safe)",color=C_MECH,ha="center",va="center",fontsize=6.8)
ax.text(5.0,1.35,r"increasing member size  $d\;\rightarrow$",ha="center",fontsize=7.3)
ax.annotate("",xy=(9.4,0.75),xytext=(0.6,0.75),arrowprops=dict(arrowstyle="->",color="k",lw=1.0))
ax.text(-0.1,9.6,"A",fontweight="bold",fontsize=11)

ax=fig.add_subplot(gs[0,1])
ax.scatter(vpool,pool.Vu_kN,s=7,c=C_IN,alpha=0.45,edgecolors="none",label="in-envelope")
ax.errorbar(vbig,big.Vu_kN,xerr=q,fmt="o",ms=3.5,c=C_OUT,alpha=0.7,elinewidth=0.4,capsize=0,label=r"out-of-env. ($\pm\hat q$)")
lim=[0,float(max(df.Vu_kN.max(),vbig.max()))*1.05]; ax.plot(lim,lim,"k--",lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ML predicted V (kN)"); ax.set_ylabel(r"measured $V_{\rm test}$ (kN)")
ax.legend(loc="upper left",frameon=False); ax.text(-0.28,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

ax=fig.add_subplot(gs[0,2])
ax.bar([0,1],[cov_in,cov_out],color=[C_IN,C_OUT],width=0.62); ax.axhline(0.9,ls="--",c="k",lw=0.8)
ax.text(1.5,0.91,"target\n0.90",fontsize=6.5,va="bottom",ha="right")
ax.set_xticks([0,1]); ax.set_xticklabels(["in-\nenv.","out-of-\nenv."]); ax.set_ylim(0,1.0); ax.set_ylabel("interval coverage")
for xx,v in zip([0,1],[cov_in,cov_out]): ax.text(xx,v+0.02,f"{v:.2f}",ha="center",fontsize=7.5)
ax.text(-0.42,1.02,"C",transform=ax.transAxes,fontweight="bold",fontsize=11)
fig.savefig(FIG/"fig1_envelope_cliff.pdf"); fig.savefig(FIG/"fig1_envelope_cliff.png"); plt.close(fig)
print(f"fig1 v2 (steel) done: cov in={cov_in:.2f} out={cov_out:.2f}, n_pool={len(pool)} n_big={len(big)}")
