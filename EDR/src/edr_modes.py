# ======================================================================
# edr_modes.py — Modo base ψ₀ y normalización
# ======================================================================

import numpy as np
from scipy.integrate import simps

def psi0(r, rH):
    """Modo base GR aproximado."""
    return np.exp(-0.4*(r - rH)) * (r/rH)**2

def normalize_psi(r, psi):
    N = simps(psi*psi, r)
    return psi / np.sqrt(N)
