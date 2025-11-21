import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# ============================================================
# MODELO DE VELOCIDAD EDR + BARIONES (DISCO + BULBO + GAS)
# ============================================================

def v_edr_model(r, A, R0):
    """Modelo de velocidad EDR:
       v_EDR(r) = sqrt( A * (1 - exp(-r / R0)) )
    """
    return np.sqrt(A * (1.0 - np.exp(-r / R0)))


def total_velocity_model(r, A, R0, Yd, Yb, Vgas, Vdisk, Vbul):
    """Velocidad total combinando:
       - gas
       - disco estelar con masa M/L variable Yd
       - bulbo estelar con masa M/L variable Yb
       - contribución del fluido EDR
    """
    v2 = (
        (Vgas)**2
        + (Yd * Vdisk)**2
        + (Yb * Vbul)**2
        + (v_edr_model(r, A, R0))**2
    )
    return np.sqrt(v2)


# ============================================================
# DETECCIÓN DEL FORMATO SPARC E IMPORTACIÓN GENERAL
# ============================================================

def load_rotmod_generic(path):
    """
    Lee un archivo *_rotmod.dat del SPARC.
    Detecta columnas automáticamente y devuelve un dict con:
    r, Vobs, errV, Vgas, Vdisk, Vbul
    """
    df = pd.read_csv(path, comment="#", sep=r"\s+")

    cols = list(df.columns)

    if len(cols) < 6:
        raise KeyError("Formato desconocido para archivo SPARC.")

    r     = df[cols[0]].values
    Vobs  = df[cols[1]].values
    errV  = df[cols[2]].values
    Vgas  = df[cols[3]].values
    Vdisk = df[cols[4]].values
    Vbul  = df[cols[5]].values

    # Detección automática de bulbo ausente
    no_bulge = np.all(Vbul == 0.0)

    return {
        "r": r,
        "Vobs": Vobs,
        "errV": errV,
        "Vgas": Vgas,
        "Vdisk": Vdisk,
        "Vbul": Vbul,
        "has_bulge": not no_bulge
    }


# ============================================================
# AJUSTE CON RESTRICCIONES SPARC (OPCIÓN B)
# ============================================================

def fit_galaxy(data):
    """
    Ajusta:
        A, R0  → parámetros EDR
        Yd     → M/L del disco (0.3 – 1.1)
        Yb     → M/L del bulbo (0.5 – 2.0) o 0 fijo si no hay bulbo
    """
    r     = data["r"]
    Vobs  = data["Vobs"]
    errV  = data["errV"]
    Vgas  = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul  = data["Vbul"]
    has_bulge = data["has_bulge"]

    # Valores iniciales
    p0 = [100.0, 2.0, 0.6]
    bounds_lower = [0.0, 0.01, 0.3]
    bounds_upper = [500.0, 20.0, 1.1]

    if has_bulge:
        # inicial
        p0.append(1.0)
        bounds_lower.append(0.5)
        bounds_upper.append(2.0)
    else:
        p0.append(0.0)
        bounds_lower.append(0.0)
        bounds_upper.append(0.0)

    p0 = np.array(p0)

    def wrapped_model(r, *params):
        if has_bulge:
            A, R0, Yd, Yb = params
        else:
            A, R0, Yd, _ = params
            Yb = 0.0

        return total_velocity_model(r, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)

    try:
        popt, pcov = curve_fit(
            wrapped_model,
            r, Vobs,
            sigma=errV,
            absolute_sigma=True,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=50000
        )

        perr = np.sqrt(np.diag(pcov))

        # Extra uncertainty no modelada
        Vmodel = wrapped_model(r, *popt)
        sigma_extra = np.sqrt(np.mean((Vobs - Vmodel)**2))

        # χ²
        chi2 = np.sum((Vobs - Vmodel)**2 / (errV**2 + sigma_extra**2))
        Ndof = len(r) - len(popt)
        chi2_red = chi2 / Ndof

        return {
            "ok": True,
            "params": popt,
            "errors": perr,
            "chi2": chi2,
            "chi2_red": chi2_red,
            "Ndof": Ndof,
            "Ndata": len(r),
            "sigma_extra": sigma_extra
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# GRÁFICAS PROFESIONALES
# ============================================================

def plot_fit(data, result, fname):
    if not result["ok"]:
        print("No se grafica porque el ajuste falló.")
        return

    r     = data["r"]
    Vobs  = data["Vobs"]
    errV  = data["errV"]
    Vgas  = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul  = data["Vbul"]
    has_bulge = data["has_bulge"]

    p = result["params"]
    A, R0, Yd, Yb = p
    if not has_bulge:
        Yb = 0.0

    # Curvas individuales
    V_edr  = v_edr_model(r, A, R0)
    V_total = total_velocity_model(r, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)

    plt.figure(figsize=(8, 6))
    plt.title("Ajuste SPARC + EDR", fontsize=14)

    plt.errorbar(r, Vobs, yerr=errV, fmt="o", label="Observado")
    plt.plot(r, V_edr, "--", label="EDR")
    plt.plot(r, Yd * Vdisk, label="Disco escalado")
    if has_bulge:
        plt.plot(r, Yb * Vbul, label="Bulbo escalado")
    plt.plot(r, np.sqrt(Vgas**2), label="Gas")
    plt.plot(r, V_total, linewidth=2, label="Total (modelo)")

    plt.xlabel("Radio [kpc]")
    plt.ylabel("Velocidad [km/s]")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=200)
    plt.close()
