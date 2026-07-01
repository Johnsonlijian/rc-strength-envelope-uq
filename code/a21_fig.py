import pandas as pd, numpy as np
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
  "font.family":"sans-serif",
  "font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],
  "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
  "font.size":9,"axes.linewidth":0.8,"savefig.dpi":450,"savefig.bbox":"tight"})
r=pd.read_csv("../data/processed/steel_reliability.csv")
fig,ax=plt.subplots(figsize=(3.7,2.9))
ax.plot(r.d_mid,r.beta_ml,"o-",c="#c1352c",lw=1.7,ms=5,label="designed with ML")
ax.plot(r.d_mid,r.beta_me,"s-",c="#2a9d5c",lw=1.7,ms=5,label="designed with mechanical model (MCFT)")
ax.axhline(3.8,ls="--",c="k",lw=0.9)
ax.text(r.d_mid.max()*0.98,3.86,r"target $\beta_T=3.8$",fontsize=6.8,ha="right",va="bottom")
ax.axhspan(0,3.8,color="#fdecea",alpha=0.35)
ax.text(r.d_mid.min()*1.1,0.4,r"below-target region ($< \beta_T$)",fontsize=6.2,color="#c1352c",va="bottom")
# annotate the ML minimum
idx_min=r.beta_ml.idxmin(); x_min=r.d_mid[idx_min]; y_min=r.beta_ml[idx_min]
ax.annotate(f"$\\beta\\approx{y_min:.1f}$",xy=(x_min,y_min),xytext=(x_min*2.0,y_min+0.6),
            fontsize=6.8,color="#c1352c",
            arrowprops=dict(arrowstyle="->",color="#c1352c",lw=0.8))
ax.set_xscale("log"); ax.set_xlabel("effective depth $d$ (mm)")
ax.set_ylabel(r"realised reliability index $\beta$")
ax.set_ylim(0,5); ax.legend(frameon=False,fontsize=6.6,loc="upper left")
ax.set_title("Reliability versus member size",fontsize=8.0)
fig.tight_layout()
for stem in ("fig4_reliability_across_size", "fig7_reliability_across_size"):
    fig.savefig(f"../figures/{stem}.svg")
    fig.savefig(f"../figures/{stem}.pdf")
    fig.savefig(f"../figures/{stem}.png")
print("fig4 v3 (real steel, annotated; legacy alias fig7_reliability_across_size) done")
