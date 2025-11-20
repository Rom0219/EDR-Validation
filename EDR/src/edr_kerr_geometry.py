# ======================================================================
# edr_kerr_geometry.py — Métrica, horizonte, funciones Σ, Δ
# ======================================================================

import numpy as np

def Delta(r, M, a):
    return r**2 - 2*M*r + a**2

def Sigma(r, th, a):
    return r**2 + a**2*np.cos(th)**2

def r_plus(M, a):
    return M + np.sqrt(M**2 - a**2)

def Omega_H(M, a):
    rp = r_plus(M, a)
    return a / (2*M*rp)
