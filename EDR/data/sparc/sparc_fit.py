import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -----------------------------------------------------------
#   MODELO EDR (FENOMENOLÓGICO)
# -----------------------------------------------------------
def v_edr_model(r, A, R0):
    """
    Modelo fenomenológico simple inspirado en EDR:
        v(r) = A * (1 - exp(-r / R0))
    donde:
        A  = velocidad asintótica (km/s)
        R0 = escala radial (kpc)
    """
    r = np.asarray(r, dtype=float)
    return A * (1.0 - np.exp(-r / R0))


# -----------------------------------------------------------
#   LECTURA DE ARCHIVOS SPARC
# -----------------------------------------------------------
def load_rotmod_generic(path):
    """
    Carga archivos SPARC del tipo *_rotmod.dat
    Columnas esperadas:
        R  Vobs  Error  Vgas  Vdisk  Vbulge  SBdisk  SBbul
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe archivo: {path}")

    # Leer ignorando líneas con '#'
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        engine="python",
        header=None
    )

    # Verificar número de columnas
    if df.shape[1] < 3:
        raise ValueError(f"Formato no reconocido en {path}")

    # Asignar nombres estándar SPARC
    df.columns = [
        "R", "Vobs", "eV", "Vgas",
        "Vdisk", "Vbul", "SBdisk", "SBbul"
    ][:df.shape[1]]

    # Filtrar filas válidas
    df = df.dropna()

    # Extraer columnas mínimas necesarias
    r = df["R"].values.astype(float)
    v = df["Vobs"].values.astype(float)
    ev = df["eV"].values.astype(float)

    return {
        "R": r,
        "Vobs": v,
        "eV": ev,
        "Vgas": df["Vgas"].values if "Vgas" in df else None,
        "Vdisk": df["Vdisk"].values if "Vdisk" in df else None,
        "Vbul": df["Vbul"].values if "Vbul" in df else None,
        "file": os.path.basename(path)
    }


# -----------------------------------------------------------
#   AJUSTE CON CURVE_FIT
# -----------------------------------------------------------
def fit_edr_rotation_curve(R, Vobs, eV):
    """
    Ajusta v_EDR(r) a datos SPARC usando curve_fit.
    Retorna:
        - dict con parámetros y errores
        - chi^2 reducido
    """
    try:
        p0 = [np.max(Vobs), np.median(R)]  # Aprox inicial
        popt, pcov = curve_fit(
            v_edr_model, R, Vobs,
            sigma=eV,
            absolute_sigma=True,
            p0=p0,
            maxfev=20000
        )
        A, R0 = popt
        Aerr, R0err = np.sqrt(np.diag(pcov))

        model = v_edr_model(R, A, R0)
        chi2 = np.sum(((Vobs - model) / eV)**2)
        dof = len(R) - 2
        chi2_red = chi2 / max(dof, 1)

        return {
            "A": A,
            "R0": R0,
            "Aerr": Aerr,
            "R0err": R0err,
            "chi2": chi2,
            "chi2_red": chi2_red,
        }

    except Exception as e:
        print(f"[ERROR] Fallo en ajuste EDR: {e}")
        return None


# -----------------------------------------------------------
#   PLOT DE AJUSTE
# -----------------------------------------------------------
def plot_fit(data, fitres, fname="plot.png"):
    """
    Grafica curva de rotación y modelo EDR ajustado.
    """
    R = data["R"]
    V = data["Vobs"]
    eV = data["eV"]

    r_plot = np.linspace(min(R), max(R), 500)
    v_model = v_edr_model(r_plot, fitres["A"], fitres["R0"])

    plt.figure(figsize=(7, 5))
    plt.errorbar(R, V, yerr=eV, fmt="o", label="Datos SPARC", color="black")
    plt.plot(r_plot, v_model, label="Ajuste EDR", linewidth=2)

    plt.xlabel("Radio (kpc)")
    plt.ylabel("Velocidad (km/s)")
    plt.title(f"Ajuste EDR — {data['file']}")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------
#   EXPORTAR RESULTADOS
# -----------------------------------------------------------
def save_fit_result(result, galaxy, out_csv):
    """
    Añade una fila al archivo CSV con los parámetros del ajuste.
    """
    row = {
        "Galaxy": galaxy,
        "A": result["A"],
        "Aerr": result["Aerr"],
        "R0": result["R0"],
        "R0err": result["R0err"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"]
    }

    import csv
    header_exists = os.path.isfile(out_csv)

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)
