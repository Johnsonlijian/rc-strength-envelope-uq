"""mesh-objectivity check: crack-band should make v_nom ~ mesh-independent."""
from damage_fe import shear_capacity_fe
for d in [300, 900]:
    print(f"d={d}:")
    for nd in [10, 14, 20]:
        v,V,_=shear_capacity_fe(d=d,a_d=3.0,rho=0.015,fc=30.0,n_d=nd,steps=40)
        print(f"   n_d={nd:2d}  v_nom={v:.3f} MPa  Vpeak={V/1e3:.1f} kN")
