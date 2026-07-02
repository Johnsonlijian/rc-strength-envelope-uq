"""
a30_fig_closure.py — Figure: closing the loop on the canonical steel benchmark.
 (A) same-metric interval test: every predictor through the identical ln-space
     split-conformal pipeline; mechanical models retain OOE coverage, ML does not
 (B) protocol-level realised reliability: envelope gate + mechanical fallback
     restores beta at every size bin
 (C) physics-anchored learners: residual-target learning recovers most (not all)
     validity; the physics-feature hybrid does not; weighted conformal abstains
Outputs ../figures/fig8_steel_closure.{svg,pdf,png}
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

PROC = Path("../data/processed")
FIG = Path("../figures")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9,
    "axes.linewidth": 0.8, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 450, "savefig.bbox": "tight",
})
C_IN, C_OUT, C_MECH, C_ACC = "#2c6fbb", "#c1352c", "#2a9d5c", "#7a7a7a"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.85))

    # ---------------- A: same-metric interval test ----------------
    ax = axes[0]
    clean(ax)
    R = pd.read_csv(PROC / "steel_closure_raw.csv")
    order = ["ML-HistGB", "ML-RF", "MCFT", "CSCT", "fib-MC2010", "ACI318-19"]
    g = (R.groupby("predictor")
           .agg(ind=("cov_ln_ind", "mean"), ind_sd=("cov_ln_ind", "std"),
                big=("cov_ln_big", "mean"), big_sd=("cov_ln_big", "std"))
           .reindex(order))
    x = np.arange(len(order))
    cols = [C_OUT if p.startswith("ML") else C_MECH for p in order]
    ax.axhspan(g.ind.min(), g.ind.max(), fc=C_ACC, alpha=0.22, zorder=0)
    ax.text(0.03, g.ind.mean() + 0.035, "in-envelope range\n(all predictors)",
            fontsize=6.2, color="#555555", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.bar(x, g.big, width=0.62, color=cols,
           yerr=g.big_sd, capsize=1.8, error_kw=dict(lw=0.7, capthick=0.7))
    ax.axhline(0.90, ls="--", c="k", lw=0.8)
    for xi, v in zip(x, g.big):
        ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=6.4,
                color="#333333")
    ax.set_xticks(x)
    labels_a = ["HistGB", "RF", "MCFT", "CSCT", "MC2010", "318-19"]
    ax.set_xticklabels(labels_a, fontsize=6.6, rotation=35, ha="right")
    for tick, p in zip(ax.get_xticklabels(), order):
        tick.set_color(C_OUT if p.startswith("ML") else C_MECH)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("90% interval coverage, OOE")
    ax.set_title("Same pipeline, same metric:\nmechanical models keep coverage",
                 fontsize=8.2, pad=4)
    ax.text(0.16, 0.50, "learned", color=C_OUT, fontsize=6.8,
            transform=ax.transAxes, ha="center")
    ax.text(-0.24, 1.06, "A", transform=ax.transAxes, fontweight="bold",
            fontsize=11)

    # ---------------- B: protocol reliability ----------------
    ax = axes[1]
    clean(ax)
    P = pd.read_csv(PROC / "protocol_beta.csv")
    ax.axhspan(1.35, 3.8, fc=C_OUT, alpha=0.06, zorder=0)
    ax.plot(P.d_mid, P.beta_ml, "o-", c=C_OUT, lw=1.5, ms=4,
            label="learned model")
    ax.plot(P.d_mid, P.beta_mcft, "s-", c=C_MECH, lw=1.5, ms=4,
            label="MCFT everywhere")
    ax.plot(P.d_mid, P.beta_protocol, "D-", c="#1a1a1a", lw=2.2, ms=4.6,
            label="gated protocol", zorder=5)
    ax.axhline(3.8, ls="--", c="k", lw=0.8)
    ax.text(915, 3.62, "target $\\beta_T$", fontsize=6.8, ha="right", va="top")
    ax.annotate("$\\beta\\approx1.8$", xy=(525, 1.85), xytext=(300, 1.55),
                color=C_OUT, fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_OUT))
    ax.annotate("$\\beta\\geq4.0$ at every size", xy=(525, 4.42),
                xytext=(120, 4.62), color="#1a1a1a", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#1a1a1a"))
    ax.set_xlabel("$d$ bin midpoint (mm)")
    ax.set_ylabel("realised $\\beta$")
    ax.set_ylim(1.35, 5.0)
    ax.set_xlim(40, 950)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right",
              handlelength=1.4, borderaxespad=0.2)
    ax.set_title("Envelope gate + fallback\nrestores target reliability",
                 fontsize=8.2, pad=4)
    ax.text(-0.22, 1.06, "B", transform=ax.transAxes, fontweight="bold",
            fontsize=11)

    # ---------------- C: physics-anchored learners ----------------
    ax = axes[2]
    clean(ax)
    A = pd.read_csv(PROC / "physics_arms.csv")
    arms = ["baseline", "physics-feature", "residual-target"]
    labels = ["raw\nfeatures", "physics\nfeature", "residual\ntarget"]
    x = np.arange(len(arms))
    for off, ds, c in [(-0.19, "STEEL", C_OUT), (0.19, "FRP", C_IN)]:
        sub = A[A.ds == ds].groupby("arm")["cov_big"].agg(["mean", "std"]).reindex(arms)
        ax.bar(x + off, sub["mean"], width=0.36, color=c, label=ds.title()
               if ds == "STEEL" else ds,
               yerr=sub["std"], capsize=1.8, error_kw=dict(lw=0.7, capthick=0.7))
        for xi, v in zip(x + off, sub["mean"]):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.2,
                    color="#333333")
    ax.axhline(0.90, ls="--", c="k", lw=0.8)
    ax.text(0.5, 0.918, "0.90", fontsize=6.5, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("90% interval coverage (OOE)")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left",
              handlelength=1.2, borderaxespad=0.2)
    ax.set_title("Physics-anchored learning helps;\nthe gate is still needed",
                 fontsize=8.2, pad=4)
    ax.text(0.98, 0.06,
            "shift-weighted conformal:\n83–100% infinite intervals",
            transform=ax.transAxes, ha="right", fontsize=6.2, style="italic",
            color="#444444")
    ax.text(-0.24, 1.06, "C", transform=ax.transAxes, fontweight="bold",
            fontsize=11)

    fig.tight_layout(w_pad=1.6)
    for name in ("fig7_steel_closure", "fig8_steel_closure"):
        for ext in ("svg", "pdf", "png"):
            fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("fig7_steel_closure done (legacy alias fig8_steel_closure) ->", FIG.resolve())


if __name__ == "__main__":
    main()
