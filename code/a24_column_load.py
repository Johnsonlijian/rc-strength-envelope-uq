"""
a24_column_load.py — load the REAL RC column shear database (235 rectangular columns,
input geometry/material/axial-load + measured capacity Vmax). Source: refined RC-column
experimental data set (ASCE J. Struct. Eng. supplementary, zip 28845476).
Header is on row index 1; map by position (encoding-garbled names). Join input+output by ID.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
RAW=Path("../data/raw_columns"); PROC=Path("../data/processed")
# positional map of the real header row (row index 1) for rectangular columns
POS={"h":3,"b":4,"Lc":5,"cc":6,"a":7,"d1":8,"d2":9,"a_d":10,"fc":11,"fyc":12,"fyt":14,
     "rho_l":15,"rho_v":16,"rho_t":17,"P":18,"n_ax":19,"s":36}

def load():
    inp=pd.read_excel(RAW/"TestParameters_input_R.xlsx", header=1)
    df=pd.DataFrame({k: pd.to_numeric(inp.iloc[:,v], errors="coerce") for k,v in POS.items()})
    df["ID"]=pd.to_numeric(inp.iloc[:,0], errors="coerce")
    out=pd.read_excel(RAW/"TestParameters_output_R.xlsx")
    out=out.rename(columns={out.columns[0]:"ID", out.columns[1]:"Vmax_kN"})
    out["ID"]=pd.to_numeric(out["ID"], errors="coerce")
    df=df.merge(out[["ID","Vmax_kN"]], on="ID", how="inner")
    df["V_test_kN"]=df.Vmax_kN.abs()
    df=df.dropna(subset=["b","h","d1","fc","a_d","n_ax","V_test_kN"])
    df=df[(df.V_test_kN>0)&(df.d1>0)&(df.fc>0)&(df.b>0)].reset_index(drop=True)
    df.to_csv(PROC/"column_clean.csv", index=False)
    return df

def main():
    df=load()
    print(f"REAL RC column database: {len(df)} rectangular columns with measured capacity")
    print("\n== ranges ==")
    print(df[["b","h","d1","Lc","a_d","fc","fyc","rho_l","rho_t","P","n_ax","V_test_kN"]]
          .describe(percentiles=[.05,.5,.95]).T[["count","min","5%","50%","95%","max"]].to_string())
    print("\n== the KEY new dimension: axial-load ratio n=P/(f'c Ag) ==")
    for lo,hi in [(0,0.05),(0.05,0.15),(0.15,0.3),(0.3,0.5),(0.5,1.0)]:
        n=((df.n_ax>=lo)&(df.n_ax<hi)).sum(); print(f"  n in [{lo},{hi}): {n}")
    print(f"\n== size (d1) spread ==  d1 {df.d1.min():.0f}-{df.d1.max():.0f}mm")
    print("saved -> column_clean.csv")

if __name__=="__main__": main()
