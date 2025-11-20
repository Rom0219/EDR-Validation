# =================================================
# edr_kerr.py — Funciones básicas de Kerr
# =================================================

import numpy as np

def r_plus(M, a):
    """Horizonte exterior de Kerr."""
    return M + np.sqrt(M**2 - a**2)

def Omega_H(M, a):
    """Velocidad angular del horizonte."""
    rp = r_plus(M, a)
    return a / (2 * M * rp)

