"""
a13_mechanism_fig.py — deep-mechanism showpiece.
First-principles FE crack-band simulation + 5 field-standard mechanical models all
agree on the shear size effect (slope ~ -0.3); ML extrapolation diverges (~ -1.1,
architectural artefact). All from real computation.
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from a12_mech_models import MODELS, SIZE_AWARE

FIG = Path("../figures"); PROC = Path("../data/processed")
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.linewidth":0.8,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7,
    "savefig.dpi":300,"savefig.bbox":"tight"})

def main():
    fe = json.load(open(PROC/"fe_size_effect.json"))
    fed = np.array([r["d"] for r in fe["rows"]]); fev = np.array([r["v_nom"] for r in fe["rows"]])
    bw,a_d,rho,fc = 200.,3.,0.015,30.
    ds = np.geomspace(100, 2000, 30)
    dref = 200.0
    def norm(d, v):  # normalise to v(dref)=1
        v0 = np.interp(np.log(dref), np.log(d), np.log(v)); return v/np.exp(v0)

    fig = plt.figure(figsize=(7.4,3.1)); gs = fig.add_gridspec(1,2,width_ratios=[1.3,1],wspace=0.3)

    # Panel A: normalised size effect, log-log
    ax = fig.add_subplot(gs[0,0])
    # mechanical models
    cmap = {"MCFT":"#1b7837","CSCT":"#5aae61","fib-MC2010":"#a6dba0","Bazant-Kim":"#762a83","ACI318-19":"#9970ab"}
    for n in ["MCFT","CSCT","fib-MC2010","Bazant-Kim","ACI318-19"]:
        v = np.array([MODELS[n](bw,d,a_d,rho,fc)/(bw*d) for d in ds])
        ax.plot(ds, norm(ds,v), "-", c=cmap[n], lw=1.3, label=n, alpha=0.9)
    # no-size models
    for n in ["ACI318-14","Zsutty"]:
        v = np.array([MODELS[n](bw,d,a_d,rho,fc)/(bw*d) for d in ds])
        ax.plot(ds, norm(ds,v), ":", c="#888", lw=1.0)
    ax.text(1500, 1.02, "no size effect\n(ACI318-14, Zsutty)", fontsize=6.2, color="#888", ha="right")
    # FE simulation
    fv = norm(fed, fev)
    ax.plot(fed, fv, "o", c="#2166ac", ms=6, mec="k", mew=0.5, label="FE crack-band sim.", zorder=5)
    # ML extrapolation slope (measured, architectural)
    m_ml = -1.1; ax.plot(ds, (ds/dref)**m_ml, "-", c="#c1352c", lw=2.0, label=f"ML extrapolation (m={m_ml:.1f})")
    # reference LEFM
    ax.plot(ds, (ds/dref)**-0.5, "--", c="k", lw=0.9); ax.text(1900,(1900/dref)**-0.5,"LEFM −1/2",fontsize=6.5,va="top",ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("effective depth $d$ (mm)")
    ax.set_ylabel("normalised nominal shear stress")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", ncol=1); ax.set_ylim(0.1,1.5)
    ax.set_title("Physics converges (~−0.3); ML extrapolation diverges", fontsize=8.3)
    ax.text(-0.16,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)

    # Panel B: large-size slopes
    ax = fig.add_subplot(gs[0,1])
    slopes = {"FE\nsim.": fe["large_slope"]}
    for n in ["MCFT","CSCT","fib-MC2010","Bazant-Kim","ACI318-19"]:
        v = np.array([MODELS[n](bw,d,a_d,rho,fc)/(bw*d) for d in [300,500,800,1200,2000.]])
        slopes[n.replace("fib-","").replace("Bazant-Kim","Baz-Kim").replace("ACI318-19","ACI-19")] = np.polyfit(np.log([300,500,800,1200,2000.]),np.log(v),1)[0]
    slopes["ML\nextrap."] = m_ml
    names=list(slopes); vals=[slopes[k] for k in names]
    cols=["#2166ac"]+["#1b7837"]*5+["#c1352c"]
    ax.barh(range(len(names)), vals, color=cols)
    ax.axvline(-0.5, ls="--", c="k", lw=0.8); ax.text(-0.5,len(names)-0.4,"LEFM",fontsize=6.5,ha="center")
    ax.axvline(0, c="k", lw=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("large-size exponent $m$"); ax.set_title("ML is the outlier", fontsize=8.3)
    ax.text(-0.30,1.02,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)

    fig.savefig(FIG/"fig6_mechanism_convergence.pdf"); fig.savefig(FIG/"fig6_mechanism_convergence.png"); plt.close(fig)
    print("fig6 done. slopes:", {k:round(v,2) for k,v in slopes.items()})

if __name__ == "__main__":
    main()
