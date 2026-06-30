import numpy as np, pandas as pd
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
  "axes.linewidth":0.8,"savefig.dpi":300,"savefig.bbox":"tight"})
r = pd.read_csv("../data/processed/reliability_across_size.csv")
fig, ax = plt.subplots(figsize=(3.7, 2.8))
ax.plot(r.d_mid, r.beta_ml, "o-", c="#c1352c", lw=1.6, ms=5, label="designed with ML")
ax.plot(r.d_mid, r.beta_me, "s-", c="#2a9d5c", lw=1.6, ms=5, label="designed with mechanical model")
ax.axhline(3.8, ls="--", c="k", lw=0.9)
ax.text(1300, 3.95, r"target $\beta_T=3.8$", fontsize=7, ha="right")
ax.set_xscale("log")
ax.set_xlabel("effective depth d (mm)")
ax.set_ylabel(r"realised reliability index $\beta$")
ax.legend(frameon=False, fontsize=7, loc="center left")
ax.set_title("ML reliability uncontrolled out-of-envelope;\nmechanical model stays on target", fontsize=8.2)
ax.annotate("ML off-target out of envelope\n(here over-conservative; unsafe\nfor over-predicting models)",
            xy=(1250, 7.6), xytext=(180, 6.0), fontsize=6.2, color="#c1352c",
            arrowprops=dict(arrowstyle="->", color="#c1352c", lw=0.8))
fig.savefig("../figures/fig7_reliability_across_size.pdf")
fig.savefig("../figures/fig7_reliability_across_size.png")
print("fig7 done")
