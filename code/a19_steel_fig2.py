"""fig2 v2 — on the canonical steel benchmark, NO UQ method restores out-of-envelope
coverage: conformal, adaptive, GP, quantile, + GPU deep ensemble / MC-dropout."""
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
PROC=Path("../data/processed"); FIG=Path("../figures")
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
  "axes.linewidth":0.8,"xtick.labelsize":7.5,"ytick.labelsize":8,"legend.fontsize":8,
  "savefig.dpi":300,"savefig.bbox":"tight"})

m=pd.read_csv(PROC/"steel_uq_matrix.csv")          # model,uq,interp,extrap
g=pd.read_csv(PROC/"steel_gpu_uq.csv")             # method,interp,extrap
uqn={"split":"split-conf.","knn-norm":"adaptive","native90":"native"}
order=[("HistGB","split"),("HistGB","knn-norm"),("RF","split"),("RF","knn-norm"),
       ("Linear","split"),("GP","native90"),("QuantGBM","native90")]
labels,interp,extrap=[],[],[]
for mod,uq in order:
    r=m[(m.model==mod)&(m.uq==uq)]
    if len(r): labels.append(f"{mod}\n{uqn[uq]}"); interp.append(float(r.interp)); extrap.append(float(r.extrap))
for _,r in g.iterrows():
    labels.append(f"{r.method}\n(GPU)"); interp.append(float(r.interp)); extrap.append(float(r.extrap))

x=np.arange(len(labels))
fig,ax=plt.subplots(figsize=(7.4,3.2))
bars_in=ax.bar(x-0.19,interp,width=0.36,color="#7a7a7a",label="interpolation (CV)")
bars_ex=ax.bar(x+0.19,extrap,width=0.36,color="#c1352c",label="size-extrapolation")
ax.axhline(0.9,ls="--",c="k",lw=0.8); ax.text(len(labels)-0.5,0.915,"0.90",fontsize=7,ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6.4,rotation=40,ha="right")
ax.set_ylim(0,1.08); ax.set_ylabel("90% interval coverage")
ax.legend(frameon=False,loc="upper right")
# value labels on extrap bars
for xi,v in zip(x+0.19,extrap):
    ax.text(xi,v+0.014,f"{v:.2f}",ha="center",va="bottom",fontsize=5.5,color="#8b1a1a",rotation=90)
ax.set_title("Canonical steel benchmark (n=1177): no UQ method — conformal, adaptive, GP,\nquantile, or GPU deep ensemble / MC-dropout — restores coverage out of envelope",fontsize=8.2)
fig.savefig(FIG/"fig2_uq_matrix.pdf"); fig.savefig(FIG/"fig2_uq_matrix.png"); plt.close(fig)
print("fig2 v2 (steel, 6 UQ methods) done")
