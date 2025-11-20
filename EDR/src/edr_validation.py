# ======================================================================
# edr_validation.py — Pruebas numéricas oficiales del modelo EDR
# ======================================================================

import numpy as np
from edr_alpha import compute_alpha_flow

def test_blackhole(M_solar=30, a=0.7, omega0=0.5):
    d_omega, alpha, df_Hz = compute_alpha_flow(M_solar, a, omega0)
    print("----- VALIDACIÓN EDR -----")
    print(f"δω geom: {d_omega}")
    print(f"α_flow : {alpha}")
    print(f"Corrimiento en Hz: {df_Hz}")
