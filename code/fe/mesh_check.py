"""mesh-objectivity check: crack-band should make v_nom approximately mesh-independent."""
from __future__ import annotations
import json
from pathlib import Path
from damage_fe import shear_capacity_fe

PROC = Path("../../data/processed")
PROC.mkdir(parents=True, exist_ok=True)
rows = []
for d in [300, 900]:
    print(f"d={d}:")
    for nd in [10, 14, 20]:
        v,V,_=shear_capacity_fe(d=d,a_d=3.0,rho=0.015,fc=30.0,n_d=nd,steps=40)
        print(f"   n_d={nd:2d}  v_nom={v:.3f} MPa  Vpeak={V/1e3:.1f} kN")
        rows.append({"d_mm": d, "n_d": nd, "v_nom_mpa": float(v), "Vpeak_kN": float(V/1e3)})

summary = []
for d in [300, 900]:
    vals = [r["v_nom_mpa"] for r in rows if r["d_mm"] == d]
    summary.append({
        "d_mm": d,
        "min_v_nom_mpa": min(vals),
        "max_v_nom_mpa": max(vals),
        "relative_range": (max(vals) - min(vals)) / sum(vals) * len(vals),
    })

out = {"runs": rows, "summary": summary}
(PROC / "fe_mesh_objectivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved -> fe_mesh_objectivity.json")
