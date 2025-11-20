# ================================================
# edr_constants.py — Constantes físicas y unidades
# ================================================

import numpy as np

# Constantes físicas
G = 6.67430e-11
c = 299792458
Msun = 1.98847e30

# Conversión de frecuencia geométrica a Hz
def geom_to_Hz(omega_geom, M_kg):
    return (omega_geom * c**3) / (2*np.pi*G*M_kg)

# Conversión de masas
def Msun_to_kg(M_solar):
    return M_solar * Msun

