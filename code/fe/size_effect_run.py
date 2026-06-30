"""Run the FE on a geometrically-similar SCALED beam series -> size effect from
first-principles crack-band simulation. Saves v_nom(d)."""
import numpy as np, json, time
from damage_fe import shear_capacity_fe

sizes = [100, 150, 225, 300, 450, 600, 900, 1350]
rows = []
t0 = time.time()
for d in sizes:
    nd = 12 if d <= 600 else 14
    v, V, hist = shear_capacity_fe(d=d, a_d=3.0, rho=0.015, fc=30.0, ag=16.0,
                                   b=200.0, n_d=nd, steps=45)
    rows.append({"d": d, "v_nom": v, "Vpeak_kN": V/1e3})
    print(f"  d={d:4.0f} mm  v_nom={v:.3f} MPa  Vpeak={V/1e3:6.1f} kN  ({time.time()-t0:.0f}s)")

d = np.array([r["d"] for r in rows]); v = np.array([r["v_nom"] for r in rows])
# SEL fit v=v0/sqrt(1+d/d0)
from scipy.optimize import minimize_scalar
def ssr(ld0):
    d0=np.exp(ld0); f=(1+d/d0)**-0.5; v0=np.sum(v*f)/np.sum(f*f); return np.sum((v-v0*f)**2)
r=minimize_scalar(ssr,bounds=(np.log(50),np.log(50000)),method="bounded"); d0=np.exp(r.x)
f=(1+d/d0)**-0.5; v0=np.sum(v*f)/np.sum(f*f)
# large-size log-log slope
sl=np.polyfit(np.log(d[d>=300]), np.log(v[d>=300]), 1)[0]
print(f"\nFE size-effect:  SEL d0={d0:.0f} mm, v0={v0:.2f} MPa;  large-size log-log slope={sl:.2f} (LEFM=-0.5)")
json.dump({"rows":rows,"d0":d0,"v0":v0,"large_slope":sl},
          open("../../data/processed/fe_size_effect.json","w"), indent=2)
print("saved -> data/processed/fe_size_effect.json")
