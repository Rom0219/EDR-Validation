#!/usr/bin/env python3
# sparc_fit.py
# Contiene todas las funciones de ajuste, carga y gráficos para modelos SPARC y EDR.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------------------------------------
# 1. MODELO EDR PARA CURVAS DE ROTACIÓN
# -------------------------------------------------------------
def load_rotmod_generic(path):
    """
    Lector definitivo para archivos SPARC rotmod.sin header real.
    Asigna nombres de columnas estándar:
    Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    """
    import pandas as pd
    import numpy as np

    # -----------------------------------------------------------------
    # 1. Cargar ignorando comentarios y sin header
    # -----------------------------------------------------------------
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        sep=r"\s+",
        engine="python"
    )

    # -----------------------------------------------------------------
    # 2. Validación: debe haber 8 columnas EXACTAS
    # -----------------------------------------------------------------
    if df.shape[1] < 8:
        raise ValueError(
            f"Archivo {path} tiene {df.shape[1]} columnas, se esperaban 8 columnas SPARC."
        )

    # -----------------------------------------------------------------
    # 3. Asignamos nombres estándar SPARC
    # -----------------------------------------------------------------
    df.columns = [
        "Rad", "Vobs", "errV",
        "Vgas", "Vdisk", "Vbul",
        "SBdisk", "SBbul"
    ]

    # -----------------------------------------------------------------
    # 4. Extraemos arrays correctamente
    # -----------------------------------------------------------------
    r       = df["Rad"].astype(float).values
    v_obs   = df["Vobs"].astype(float).values
    v_err   = df["errV"].astype(float).values
    v_gas   = df["Vgas"].astype(float).values
    v_disk  = df["Vdisk"].astype(float).values
    v_bul   = df["Vbul"].astype(float).values

    # Curva bariónica (modelo SPARC)
    v_bary = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)

    return {
        "r": r,
        "v_obs": v_obs,
        "v_err": v_err,
        "v_bary": v_bary,
        "name": path
    }

# -------------------------------------------------------------
# 3. AJUSTE NO LINEAL
# -------------------------------------------------------------
def fit_galaxy(data, initial=(0.05, 5.0)):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    try:
        popt, pcov = curve_fit(
            v_edr_model,
            r,
            v_obs,
            sigma=v_err,
            p0=initial,
            absolute_sigma=True,
            maxfev=100000
        )
        A, R0 = popt
    except Exception:
        A, R0 = np.nan, np.nan
        pcov = np.zeros((2, 2))

    return {
        "A": A,
        "R0": R0,
        "cov": pcov
    }

# -------------------------------------------------------------
# 4. BOOTSTRAP DE ERRORES
# -------------------------------------------------------------
def bootstrap_errors(data, fitres, nboot=200):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    Ab, Rb = [], []

    for _ in range(nboot):
        idx = np.random.randint(0, len(r), len(r))
        r_b = r[idx]
        v_b = v_obs[idx]
        e_b = v_err[idx]

        try:
            popt, _ = curve_fit(
                v_edr_model,
                r_b,
                v_b,
                sigma=e_b,
                p0=(fitres["A"], fitres["R0"]),
                absolute_sigma=True
            )
            Ab.append(popt[0])
            Rb.append(popt[1])
        except Exception:
            continue

    return {
        "A_err": np.std(Ab) if Ab else np.nan,
        "R_err": np.std(Rb) if Rb else np.nan
    }

# -------------------------------------------------------------
# 5. GENERADOR DE PLOTS
# -------------------------------------------------------------
def plot_fit(data, fitres, fname=None):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    A = fitres["A"]
    R0 = fitres["R0"]

    r_plot = np.linspace(min(r), max(r), 300)
    v_model = v_edr_model(r_plot, A, R0)

    plt.figure(figsize=(7, 5))
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label="Datos", alpha=0.8)
    plt.plot(r_plot, v_model, '-', label=f"EDR fit: A={A:.3f}, R0={R0:.3f}")
    plt.xlabel("r [kpc]")
    plt.ylabel("v [km/s]")
    plt.legend()
    plt.grid(True)

    if fname:
        plt.savefig(fname, dpi=150)
    plt.close()
