"""Emit a compilable LaTeX Supplementary Information (SI.tex) from saved outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("../data/processed")
OUT = Path("../outputs")
TITLE = (
    "Machine-learned reinforced-concrete strength intervals fail beyond the "
    "training envelope: size-extrapolation hazards and a range-bounded mechanical fallback"
)


def esc(x):
    return (
        str(x)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_\\allowbreak{}")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def tex_tabular(df, colfmt=None, fmt=2):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda v: f"{v:.{fmt}f}")
    cols = colfmt or ("l" + "r" * (len(df.columns) - 1))
    head = " & ".join(f"\\textbf{{{esc(c)}}}" for c in df.columns) + " \\\\\n\\midrule"
    body = "\n".join(" & ".join(esc(x) for x in row) + " \\\\" for row in df.values)
    return f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n{body}\n\\bottomrule\n\\end{{tabular}}"


def table(df, cap, lab, **kw):
    return (
        "\\begin{table}[h]\\centering\\small\n"
        f"\\caption{{{cap}}}\\label{{{lab}}}\n"
        f"{tex_tabular(df, **kw)}\n"
        "\\end{table}\n"
    )


def maybe_column_mech_summary():
    agg_path = PROC / "column_mech_by_axial_bin.csv"
    if agg_path.exists():
        out = pd.read_csv(agg_path).rename(
            columns={
                "axial_load_ratio_bin": "axial-load ratio n",
                "mean_M": "mean M",
                "pct_unsafe": "% unsafe",
            }
        )
        return out[["axial-load ratio n", "N", "mean M", "COV", "% unsafe"]]

    row_path = PROC / "column_with_mech.csv"
    if not row_path.exists():
        return None
    d = pd.read_csv(row_path)
    rows = []
    for lo, hi in [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0)]:
        mask = (d.n_ax >= lo) & (d.n_ax < hi)
        mm = d.loc[mask, "V_test_kN"] / d.loc[mask, "Vp"]
        rows.append([f"{lo}--{hi}", int(mask.sum()), mm.mean(), mm.std() / mm.mean(), 100 * np.mean(mm < 1)])
    out = pd.DataFrame(rows, columns=["axial-load ratio n", "N", "mean M", "COV", "% unsafe"])
    out.to_csv(agg_path, index=False)
    return out


L = [
    rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}\usepackage{{booktabs,array,amsmath}}
\title{{Supplementary Information\\\large {TITLE}}}
\author{{Lijian REN\\School of Civil Engineering, Inner Mongolia University of Technology,\\Hohhot 010051, China\\Inner Mongolia Autonomous Region Key Laboratory of Green Construction and\\Intelligent Operation and Maintenance of Civil Engineering,\\Hohhot 010051, China\\College of Civil and Transportation Engineering, Hohai University,\\Nanjing 210098, China\\\texttt{{renlijian@imut.edu.cn}}\\\texttt{{https://orcid.org/0000-0003-1629-4368}}}}\date{{}}
\begin{{document}}\maketitle
\noindent All tables are emitted directly from saved analysis outputs by \texttt{{code/generate\_SI\_tex.py}}. The protocol table at the end lists the script, threshold grid, seed grid, and saved artifact used by each analysis family.
"""
]

L.append(
    table(
        pd.DataFrame(
            [
                ["Steel (canonical)", "steel / sectional", 1704, 1177, "41--2000"],
                ["FRP slender", "FRP / sectional", 327, 260, "73--937"],
                ["Steel deep", "steel / strut-tie", 840, 840, "137--2000"],
                ["SFRC", "steel-fibre / sectional", 488, 353, "85--1118"],
                ["RC columns", "steel / column P--M", 234, 234, "66--854"],
            ],
            columns=["database", "material / mechanism", "source n", "analysed n", "d or d1 range (mm)"],
        ),
        "Real RC member test databases. Beam shear sources: Zhang et al. 2022 (doi:10.1007/s00366-020-01076-x); Sheffield ORDA (doi:10.15131/shef.data.5057527); Megahed 2024 (doi:10.1038/s41598-024-64386-w); Lantsoght (doi:10.5281/zenodo.2578061). Columns: Avgin et al. 2026 (doi:10.1061/JSENDH.STENG-14725).",
        "tab:s1",
        colfmt="llrrr",
    )
)

mu = json.load(open(PROC / "steel_model_uncertainty.json", encoding="utf-8"))
L.append(
    table(
        pd.DataFrame([[k, v["mean"], v["cov"]] for k, v in mu.items()], columns=["model", "mean M", "COV"]),
        "Mechanical-model validation on the canonical steel benchmark (1177 slender beams): resistance model uncertainty $M=V_{\\mathrm{test}}/V_{\\mathrm{pred}}$.",
        "tab:s2",
    )
)

L.append(
    table(
        pd.read_csv(PROC / "steel_uq_matrix.csv"),
        "ML interval coverage on the canonical steel slender-beam benchmark ($n=1177$): interpolation vs. size-extrapolation (target 0.90). Averaged over thresholds $\\{0.70,0.75,0.80\\}$ and seeds.",
        "tab:s3",
        fmt=3,
    )
)

raw = pd.read_csv(PROC / "a07_uq_matrix_raw.csv")
frp_deep = raw.groupby(["ds", "model", "uq"]).agg(interp=("interp", "mean"), extrap=("extrap", "mean")).round(3).reset_index()
frp_deep = frp_deep.rename(columns={"ds": "database", "uq": "UQ method", "interp": "interp cov.", "extrap": "extrap cov."})
frp_deep["database"] = frp_deep["database"].replace({"STEEL": "steel deep-beam", "FRP": "FRP slender"})
frp_deep["UQ method"] = frp_deep["UQ method"].replace({"split": "split-conf.", "knn-norm": "adaptive-conf.", "native90": "native"})
if (PROC / "sfrc_uq_matrix.csv").exists():
    sfrc = pd.read_csv(PROC / "sfrc_uq_matrix.csv")
    sfrc = sfrc.groupby(["model"]).agg(interp=("interp", "mean"), extrap=("extrap", "mean")).round(3).reset_index()
    sfrc.insert(0, "database", "SFRC slender")
    sfrc.insert(2, "UQ method", "split-conf.")
    frp_deep = pd.concat([frp_deep, sfrc.rename(columns={"interp": "interp cov.", "extrap": "extrap cov."})], ignore_index=True)
L.append(
    table(
        frp_deep,
        "ML interval coverage (target 0.90) on the FRP slender-beam, steel deep-beam, and SFRC slender-beam databases. SFRC rows are emitted from the saved split-conformal replication artifact.",
        "tab:s4",
        fmt=3,
        colfmt="lllrr",
    )
)

L.append(
    table(
        pd.read_csv(PROC / "steel_gpu_uq.csv"),
        "GPU-trained deep-UQ coverage on the canonical steel benchmark (target 0.90). Input standardisation is fitted on the training pool only; split-level metadata are saved in \\texttt{steel\\_gpu\\_uq\\_raw.csv}.",
        "tab:s5",
        fmt=3,
    )
)

fe = json.load(open(PROC / "fe_size_effect.json", encoding="utf-8"))
par = json.load(open(PROC / "parametric_fe.json", encoding="utf-8"))
L.append(
    table(
        pd.DataFrame(
            [
                ["real steel (Zhang 1704)", -0.41],
                ["FE crack-band sim.", fe["large_slope"]],
                ["FE parametric (384 runs)", par["slope_mean"]],
                ["MCFT", -0.31],
                ["CSCT", -0.38],
                ["fib-MC2010", -0.30],
                ["Bazant-Kim", -0.33],
                ["ACI318-19", -0.37],
                ["ACI318-14 / Zsutty (no size term)", 0.0],
                ["ML extrapolation (architectural)", -1.00],
            ],
            columns=["source", "large-size exponent m"],
        ),
        f"Large-size exponent $m$ (size effect): real data, mechanics-based crack-band FE simulation (parametric mean ${par['slope_mean']:.2f}\\pm{par['slope_sd']:.2f}$ over 384 runs), field-standard mechanical theories, and the ML extrapolation artefact.",
        "tab:s6",
    )
)

rel = pd.read_csv(PROC / "steel_reliability.csv")
rel_cols = ["d_mid", "beta_ml", "beta_me", "cov_ml", "bias_ml", "cov_me", "bias_me", "n"]
rel = rel[[c for c in rel_cols if c in rel.columns]]
L.append(
    table(
        rel,
        "Realised reliability index $\\beta$ vs. member size on the real steel benchmark (target $\\beta_T=3.8$, full Monte-Carlo with dead+live load). The learned-model design is calibrated on a held-out split within the $d<75$th-percentile training envelope.",
        "tab:s7",
        fmt=3,
    )
)

ce = pd.read_csv(PROC / "column_extrap.csv")
ce.columns = ["axis", "model", "interp", "extrap", "bias", "cov", "unsafe"]
ce["% unsafe"] = (100 * ce["unsafe"]).round(0)
L.append(
    table(
        ce[["axis", "model", "interp", "extrap", "bias", "cov", "% unsafe"]],
        "RC columns (234 tests): ML interval coverage under interpolation vs. extrapolation on two axes, with out-of-envelope model-error bias, COV, and unsafe (over-prediction) fraction.",
        "tab:s8",
        fmt=3,
    )
)

col_mech = maybe_column_mech_summary()
if col_mech is not None:
    L.append(
        table(
            col_mech,
            "Column mechanical model (P--M flexure + ACI 318-19 shear, lateral capacity = lesser) validated on the real columns, by axial-load bin. This aggregate table is saved as \\texttt{column\\_mech\\_by\\_axial\\_bin.csv} for public release.",
            "tab:s9",
            colfmt="lrrrr",
        )
    )

if (PROC / "fe_mesh_objectivity.json").exists():
    mesh = json.load(open(PROC / "fe_mesh_objectivity.json", encoding="utf-8"))
    mesh_df = pd.DataFrame(mesh["summary"]).rename(
        columns={"d_mm": "d (mm)", "min_v_nom_mpa": "min v_nom", "max_v_nom_mpa": "max v_nom", "relative_range": "rel. range"}
    )
    L.append(table(mesh_df, "Mesh-objectivity check for the crack-band FE model under $2\\times$ refinement.", "tab:s10", fmt=3))

protocol = pd.DataFrame(
    [
        ["Steel UQ matrix", "a16_steel_extrapolation.py", "0.70, 0.75, 0.80", "0, 1, 2", "steel_uq_matrix_raw.csv; steel_uq_matrix.csv"],
        ["FRP/deep-beam UQ matrix", "a07_corrected_uq_matrix.py", "0.70, 0.75, 0.80", "0, 1", "a07_uq_matrix_raw.csv"],
        ["SFRC replication", "a09_sfrc_extrap.py", "0.70, 0.75, 0.80", "0, 1, 2", "sfrc_uq_matrix.csv"],
        ["Deep UQ", "a17_gpu_uq.py", "0.70, 0.75, 0.80", "split seed 0; ensemble seeds 0--4", "steel_gpu_uq_raw.csv"],
        ["Reliability consequence", "a21_steel_reliability.py", "0.75 calibration boundary", "split seed 0; MC seed fixed", "steel_reliability.csv"],
        ["Column dual-axis UQ", "a25_column_extrap.py", "size and axial-load holdout", "script fixed seeds", "column_extrap.csv"],
        ["Column mechanical aggregate", "a26_column_mech_validate.py", "axial-load bins", "deterministic", "column_mech_by_axial_bin.csv"],
    ],
    columns=["analysis", "script", "threshold grid", "seed grid", "saved artifact"],
)
L.append(
    table(
        protocol,
        "Reproducibility protocol and saved artifact map.",
        "tab:s11",
        colfmt="p{0.13\\textwidth}p{0.20\\textwidth}p{0.16\\textwidth}p{0.18\\textwidth}p{0.17\\textwidth}",
    )
)

L.append(r"\end{document}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "SI.tex").write_text("\n".join(L), encoding="utf-8")
print("wrote outputs/SI.tex")
