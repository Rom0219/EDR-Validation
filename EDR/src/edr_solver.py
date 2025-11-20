# ======================================================================
# edr_solver.py — Motor de integración para diferentes perfiles
# ======================================================================

import numpy as np
from scipy.integrate import simps
from edr_deltaV import deltaV
from edr_modes import psi0, normalize_psi

def delta_omega_general(r, psi, dV, omega0):
    integrand = psi * dV * psi
    I = simps(integrand, r)
    return 0.5 * I / omega0
