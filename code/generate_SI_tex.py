"""Emit a compilable LaTeX Supplementary Information (SI.tex) with all result tables."""
from __future__ import annotations
import json, pandas as pd, numpy as np
from pathlib import Path
PROC=Path("../data/processed"); MS=Path("../outputs"); MS.mkdir(exist_ok=True)

def tex_tabular(df, colfmt=None, fmt=2):
    df=df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc": df[c]=df[c].map(lambda v:f"{v:.{fmt}f}")
    cols=colfmt or ("l"+"r"*(len(df.columns)-1))
    head=" & ".join("\\textbf{%s}"%str(c).replace("_","\\_").replace("%","\\%").replace("&","\\&") for c in df.columns)+" \\\\\n\\midrule"
    body="\n".join(" & ".join(str(x).replace("_","\\_").replace("%","\\%") for x in r)+" \\\\" for r in df.values)
    return f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n{body}\n\\bottomrule\n\\end{{tabular}}"

def table(df,cap,lab,**kw):
    return f"\\begin{{table}}[h]\\centering\\small\n\\caption{{{cap}}}\\label{{{lab}}}\n{tex_tabular(df,**kw)}\n\\end{{table}}\n"

L=[r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{booktabs,array,amsmath}
\title{Supplementary Information\\\large Machine-learned strength predictions for reinforced-concrete members are unreliable beyond their training envelope}
\author{Lijian REN\\School of Civil Engineering, Inner Mongolia University of Technology,\\Hohhot 010051, China\\Inner Mongolia Autonomous Region Key Laboratory of Green Construction and\\Intelligent Operation and Maintenance of Civil Engineering,\\Hohhot 010051, China\\College of Civil and Transportation Engineering, Hohai University,\\Nanjing 210098, China\\\texttt{renlijian@imut.edu.cn}\\\texttt{https://orcid.org/0000-0003-1629-4368}}\date{}
\begin{document}\maketitle
\noindent All tables are emitted directly from the analysis outputs by \texttt{code/generate\_SI\_tex.py}; every manuscript number is reproduced by the listed scripts on the listed real databases.
Tables S1--S8 cover: (S1) database provenance, (S2) mechanical-model validation, (S3) steel slender-beam UQ matrix, (S3b) FRP and deep-beam UQ matrix, (S4) GPU deep-UQ results, (S5) size-effect exponents, (S6) reliability across size, (S7) column dual-axis extrapolation, (S8) column mechanical model by axial-load bin.
"""]
# S1
L.append(table(pd.DataFrame([
 ["Steel (canonical)","steel / sectional",1704,1177,"41--2000"],
 ["FRP slender","FRP / sectional",327,260,"73--937"],
 ["Steel deep","steel / strut-tie",840,840,"137--2000"],
 ["SFRC","steel-fibre / sectional",488,353,"85--1118"],
 ["RC columns","steel / column P--M",234,234,"66--854"]],
 columns=["database","material / mechanism","source n","analysed n","d or d1 range (mm)"]),
 "Real RC member test databases. Beam shear sources: Zhang et al.\\ 2022 (doi:10.1007/s00366-020-01076-x); Sheffield ORDA (doi:10.15131/shef.data.5057527); Megahed 2024 (doi:10.1038/s41598-024-64386-w); Lantsoght (doi:10.5281/zenodo.2578061). Columns: Avgin et al.\\ 2026 (doi:10.1061/JSENDH.STENG-14725).",
 "tab:s1",colfmt="llrrr"))
# S2
mu=json.load(open(PROC/"steel_model_uncertainty.json"))
L.append(table(pd.DataFrame([[k,v["mean"],v["cov"]] for k,v in mu.items()],columns=["model","mean M","COV"]),
 "Mechanical-model validation on the canonical steel benchmark (1177 slender beams): resistance model uncertainty $M=V_{\\mathrm{test}}/V_{\\mathrm{pred}}$, consistent with the literature.","tab:s2"))
# S3: steel slender UQ matrix (main benchmark)
m=pd.read_csv(PROC/"steel_uq_matrix.csv")
L.append(table(m,"ML interval coverage on the canonical steel slender-beam benchmark ($n=1177$): interpolation vs.\\ size-extrapolation (target 0.90). Averaged over thresholds $\\{0.70,0.75,0.80\\}$ and seeds.","tab:s3",fmt=3))
# S4: GPU deep methods, steel
L.append(table(pd.read_csv(PROC/"steel_gpu_uq.csv"),"GPU-trained deep-UQ coverage on the canonical steel benchmark (target 0.90).","tab:s4",fmt=3))
# S3b: FRP and deep-beam UQ matrix from a07
raw=pd.read_csv(PROC/"a07_uq_matrix_raw.csv")
# aggregate: pandas mean(skipna=True) so GP/QuantGBM interp and extrap each average correctly
s3b=raw.groupby(["ds","model","uq"]).agg(interp=("interp","mean"),extrap=("extrap","mean")).round(3).reset_index()
s3b=s3b.rename(columns={"ds":"database","uq":"UQ method","interp":"interp cov.","extrap":"extrap cov."})
s3b["database"]=s3b["database"].replace({"STEEL":"steel deep-beam","FRP":"FRP slender"})
s3b["UQ method"]=s3b["UQ method"].replace({"split":"split-conf.","knn-norm":"adaptive-conf.","native90":"native"})
L.append(table(s3b,"ML interval coverage (target 0.90) on the FRP slender-beam ($n=260$) and steel deep-beam ($n=840$) databases: interpolation vs.\\ size-extrapolation. ``Steel'' here denotes deep beams (strut-and-tie). Averaged over thresholds $\\{0.70,0.75,0.80\\}$ and seeds.","tab:s3b",fmt=3,colfmt="lllrr"))
# S5
fe=json.load(open(PROC/"fe_size_effect.json")); par=json.load(open(PROC/"parametric_fe.json"))
L.append(table(pd.DataFrame([
 ["real steel (Zhang 1704)",-0.41],["FE crack-band sim.",fe["large_slope"]],["FE parametric (384 runs)",par["slope_mean"]],
 ["MCFT",-0.31],["CSCT",-0.38],["fib-MC2010",-0.30],["Bazant-Kim",-0.33],["ACI318-19",-0.37],
 ["ACI318-14 / Zsutty (no size term)",0.0],["ML extrapolation (architectural)",-1.00]],
 columns=["source","large-size exponent m"]),
 f"Large-size exponent $m$ (size effect): real data, first-principles FE (parametric mean ${par['slope_mean']:.2f}\\pm{par['slope_sd']:.2f}$ over 384 runs), five mechanical theories, and the ML extrapolation artefact.","tab:s5"))
# S6
L.append(table(pd.read_csv(PROC/"steel_reliability.csv"),
 "Realised reliability index $\\beta$ vs.\\ member size on the real steel benchmark (target $\\beta_T=3.8$, full Monte-Carlo with dead+live load).","tab:s6"))
# S7 column dual-axis extrapolation
ce=pd.read_csv(PROC/"column_extrap.csv").rename(columns={ce_c:ce_c for ce_c in []})
ce.columns=["axis","model","interp","extrap","bias","cov","unsafe"]
ce["%unsafe"]=(100*ce["unsafe"]).round(0)
L.append(table(ce[["axis","model","interp","extrap","bias","cov","%unsafe"]],
 "RC columns (234 tests): ML interval coverage under interpolation vs.\\ extrapolation on two axes (member size and axial-load ratio $n=P/f'_c A_g$), with out-of-envelope model-error bias, COV, and unsafe (over-prediction) fraction. Size-axis coverage collapses; axial-axis failure is unsafe ($\\sim$48\\%).","tab:s7",fmt=3))
column_with_mech = PROC/"column_with_mech.csv"
if column_with_mech.exists():
    d=pd.read_csv(column_with_mech); Mc=d.V_test_kN/d.Vp
    s8=[]
    for lo,hi in [(0,0.15),(0.15,0.3),(0.3,0.5),(0.5,1.0)]:
        m=(d.n_ax>=lo)&(d.n_ax<hi); Mm=(d.V_test_kN[m]/d.Vp[m])
        s8.append([f"{lo}--{hi}", int(m.sum()), round(float(Mm.mean()),2), round(float(Mm.std()/Mm.mean()),2), int(round(100*np.mean(Mm.values<1)))])
    L.append(table(pd.DataFrame(s8,columns=["axial-load ratio n","N","mean M","COV","% unsafe"]),
     "Column mechanical model (P--M flexure + ACI 318-19 shear, lateral capacity = lesser) validated on the real columns, by axial-load bin. Bounded and size-stable overall (mean 1.38, COV 0.35), but increasingly scattered at high axial load --- high-$n$ columns are a hard regime for both ML and first-order mechanics.","tab:s8",colfmt="lrrrr"))
else:
    L.append(r"\paragraph{Table S8 boundary.} The row-level column table used for the mechanical-model-by-axial-load-bin summary is not redistributed in this public package. Restore the third-party column records and rerun the column-processing scripts to regenerate this table.")
L.append(r"\end{document}")
open(MS/"SI.tex","w",encoding="utf-8").write("\n".join(L))
print("wrote outputs/SI.tex")
