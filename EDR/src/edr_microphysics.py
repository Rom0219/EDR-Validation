# ======================================================================
# edr_microphysics.py — Modelos η(ρ), ξ(ρ), presión y estabilidad
# ======================================================================

import numpy as np

def rho_profile(r, rH):
    return (rH/r)**4

def eta_rho(rho, eta0=0.12, alpha=1/3):
    return eta0 * rho**alpha

def xi_rho(rho, xi0=0.08, beta=1/2):
    return xi0 * rho**beta

def pressure(rho, K=0.22, gamma=1.3):
    return K * rho**gamma

def sound_speed(rho, K=0.22, gamma=1.3):
    return gamma*K*rho**(gamma-1)

