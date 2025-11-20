# =============================================================
# edr_potential.py — δV_EDR(r), ψ0(r), integral para δω
# =============================================================

import numpy as np
from scipy.integrate import simps

from edr_kerr import r_plus
from edr_constants import geom_to_Hz

# Parámetros de fluido EDR
eta = 0.12
xi  = 0.08
k_flow = 0.03

def psi0(r, rH):
    """Solución radial aproximada (modo 22 GR)."""
    return np.exp(-0.4*(r - rH)) * (r/rH)**2

def deltaV(r, rH):
    """Potencial perturbativo EDR."""
    F1 = (rH/r)**3
    F2 = np.exp(-(r-rH))
    F3 = (rH/r)**2 * np.sin(r - rH)
    return eta*F1 + xi*F2 + k_flow*F3

def delta_omega(M, a, omega0):
    """Calcula δω para Kerr + EDR."""
    rH = r_plus(M, a)
    r  = np.linspace(rH+1e-6, 200, 20000)

    psi = psi0(r, rH)
    dV  = deltaV(r, rH)

    integrand = psi * dV * psi
    I = simps(integrand, r)

    return 0.5/omega0 * I

