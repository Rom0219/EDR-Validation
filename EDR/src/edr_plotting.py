# ======================================================================
# edr_plotting.py — Gráficas físicas de δV, ψ₀, integrando modos
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from edr_deltaV import deltaV
from edr_modes import psi0
from edr_kerr_geometry import r_plus

def plot_deltaV(M=1, a=0.7):
    rH = r_plus(M, a)
    r = np.linspace(rH+0.01, 100, 5000)
    plt.plot(r, deltaV(r, rH))
    plt.title("Potencial EDR δV(r)")
    plt.xlabel("r")
    plt.ylabel("δV")
    plt.grid()
    plt.show()
