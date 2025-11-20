# ======================================================================
# edr_alpha.py — δ ω , α_flow, frecuencia corregida
# ======================================================================

import numpy as np
from scipy.integrate import simps

from edr_deltaV import deltaV
from edr_modes import psi0, normalize_psi
from edr_kerr_geometry import r_plus
from edr_constants import geom_to_Hz, Msun_to_kg

def compute_alpha_flow(M_solar, a_over_M, omega0):
    M = 1.0
    a = a_over_M

    rH = r_plus(M, a)
    r = np.linspace(rH+1e-6, 200, 20000)

    psi = normalize_psi(r, psi0(r, rH))
    dV  = deltaV(r, rH)

    integrand = psi * dV * psi
    I = simps(integrand, r)

    d_omega = 0.5*I/omega0
    alpha   = d_omega / omega0

    M_kg = Msun_to_kg(M_solar)
    df_Hz = geom_to_Hz(d_omega.real, M_kg)

    return d_omega, alpha, df_Hz
