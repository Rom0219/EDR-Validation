# ======================================================================
# edr_deltaV.py — Potencial efectivo δV_EDR(r)
# ======================================================================

import numpy as np
from edr_tensor import D_effective
from edr_microphysics import rho_profile, eta_rho, xi_rho

k_flow = 0.03

def deltaV(r, rH):
    rho = rho_profile(r, rH)
    eta = eta_rho(rho)
    xi  = xi_rho(rho)

    term1 = eta * (rH/r)**3
    term2 = xi  * np.exp(-(r-rH))
    term3 = k_flow * (rH/r)**2 * np.sin(r-rH)

    Dterm = D_effective(r, rH)

    return term1 + term2 + term3 + 0.2*Dterm
