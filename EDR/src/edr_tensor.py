# ======================================================================
# edr_tensor.py — Tensor D_{μν} linealizado (versión computacional)
# ======================================================================

import numpy as np

# Parámetros EDR
eta = 0.12
xi  = 0.08

def Omega(r, rH):
    """Magnitud efectiva de la vorticidad de fondo."""
    return (rH/r)**2 * np.exp(-(r-rH)/3)

def div_u(r, rH):
    """Divergencia efectiva del flujo."""
    return (rH/r)**3

def D_rr(r, rH):
    """Componente diagonal δD_rr(r)."""
    Om = Omega(r, rH)
    Du = div_u(r, rH)
    return 2*eta*Om**2 - xi*Du**2

def D_tt(r, rH):
    """Componente temporal δD_tt(r)."""
    Om = Omega(r, rH)
    return -0.25*eta*Om**2

def D_mix(r, rH):
    """Componente de mezcla angular r–θ."""
    Om = Omega(r, rH)
    return -eta * Om * np.sin(r-rH)

def D_effective(r, rH):
    """Combinación final usada en δV."""
    return 0.7*D_rr(r,rH) + 0.2*D_tt(r,rH) + 0.1*D_mix(r,rH)

