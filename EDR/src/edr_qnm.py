# =============================================================
# edr_qnm.py — Cálculo de α_flow y frecuencia corregida
# =============================================================

from edr_constants import geom_to_Hz, Msun_to_kg
from edr_potential import delta_omega

def compute_alpha_flow(M_solar, a_over_M, omega0):
    M_geom = 1.0
    a_geom = a_over_M
    
    # δω geométrico
    d_omega = delta_omega(M_geom, a_geom, omega0)

    # α_flow
    alpha = d_omega / omega0

    # Conversión a Hz
    M_kg = Msun_to_kg(M_solar)
    df_Hz = geom_to_Hz(d_omega.real, M_kg)
    
    return d_omega, alpha, df_Hz

