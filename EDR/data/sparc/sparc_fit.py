import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ----------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------

def load_rotmod_generic(path):
    """
    Carga un archivo *_rotmod.dat de SPARC.
    Lee columnas: R, Vobs, errV, Vgas, Vdisk, Vbul
    """
    df = pd.read_csv(path, comment='#', sep=r'\s+', engine='python')

    colnames = df.columns

    if len(colnames) < 5:
        raise ValueError(f"Archivo {path} no tiene columnas suficientes.")

    R       = df[colnames[0]].values.astype(float)
    Vobs    = df[colnames[1]].values.astype(float)
    errV    = df[colnames[2]].values.astype(float)
    Vgas    = df[colnames[3]].values.astype(float)
    Vdisk   = df[colnames[4]].values.astype(float)

    if len(colnames) >= 6:
        Vbul = df[colnames[5]].values.astype(float)
    else:
        Vbul = np.zeros_like(R)

    return {
        "R": R,
        "Vobs": Vobs,
        "errV": errV,
        "Vgas": Vgas,
        "Vdisk": Vdisk,
        "Vbul": Vbul
    }


# ----------------------------------------------------------------------
# MODELO DEL FLUIDO EDR
# ----------------------------------------------------------------------

def v_edr(r, A, R0):
    """Modelo EDR simple."""
    return A * (1 - np.exp(-r / R0))


def model_no_bulge(R, A, R0, Yd, Vgas, Vdisk):
    """Modelo cuando NO hay bulbo."""
    Vb = np.zeros_like(R)
    return np.sqrt(Vgas**2 + (Yd * Vdisk)**2 + v_edr(R, A, R0)**2)


def model_with_bulge(R, A, R0, Yd, Yb, Vgas, Vdisk, Vbul):
    """Modelo cuando sí hay bulbo real."""
    return np.sqrt(Vgas**2 +
                   (Yd * Vdisk)**2 +
                   (Yb * Vbul)**2 +
                   v_edr(R, A, R0)**2)


# ----------------------------------------------------------------------
# AJUSTE CON O SIN BULBO
# ----------------------------------------------------------------------

def fit_galaxy(data):

    R     = data["R"]
    Vobs  = data["Vobs"]
    errV  = data["errV"]
    Vgas  = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul  = data["Vbul"]

    has_bulge = not np.allclose(Vbul, 0.0)

    if has_bulge:
        # ----------------------------------------------------------
        # AJUSTE CON BULBO
        # ----------------------------------------------------------
        def fit_fun(R, A, R0, Yd, Yb):
            return model_with_bulge(R, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)

        p0 = [100, 1.5, 0.6, 1.0]
        bounds = (
            [0.01, 0.01, 0.3, 0.5],         # A, R0, Yd, Yb
            [500, 10,  1.1, 2.0]
        )

    else:
        # ----------------------------------------------------------
        # AJUSTE SIN BULBO
        # ----------------------------------------------------------
        def fit_fun(R, A, R0, Yd):
            return model_no_bulge(R, A, R0, Yd, Vgas, Vdisk)

        p0 = [100, 1.5, 0.6]
        bounds = (
            [0.01, 0.01, 0.3],                # A, R0, Yd
            [500, 10,  1.1]
        )

    # ------------------------------------------------------------------
    # AJUSTE
    # ------------------------------------------------------------------

    try:
        popt, pcov = curve_fit(
            fit_fun,
            R,
            Vobs,
            sigma=errV,
            p0=p0,
            bounds=bounds,
            maxfev=200000
        )
    except Exception as e:
        return None, f"ERROR en ajuste: {str(e)}"

    # Parámetros ajustados
    if has_bulge:
        A, R0, Yd, Yb = popt
    else:
        A, R0, Yd = popt
        Yb = 0.0

    modelV = fit_fun(R, *popt)

    chi2 = np.sum(((Vobs - modelV) / errV) ** 2)
    ndof = len(R) - len(popt)
    chi2_red = chi2 / ndof if ndof > 0 else np.nan

    result = {
        "A": A,
        "R0": R0,
        "Yd": Yd,
        "Yb": Yb,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "has_bulge": has_bulge
    }

    return result, modelV


# ----------------------------------------------------------------------
# PLOT PROFESIONAL
# ----------------------------------------------------------------------

def plot_fit(data, modelV, result, fname=None, galaxy_name=""):
    R     = data["R"]
    Vobs  = data["Vobs"]
    errV  = data["errV"]
    Vgas  = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul  = data["Vbul"]

    A  = result["A"]
    R0 = result["R0"]
    Yd = result["Yd"]
    Yb = result["Yb"]
    has_bulge = result["has_bulge"]

    plt.figure(figsize=(8,6))
    plt.errorbar(R, Vobs, yerr=errV, fmt='k.', label='Vobs')

    plt.plot(R, Vgas, 'c--', label='Gas')
    plt.plot(R, Yd * Vdisk, 'g-', label=f'Disco (Yd={Yd:.2f})')

    if has_bulge:
        plt.plot(R, Yb * Vbul, 'm-', label=f'Bulbo (Yb={Yb:.2f})')
    else:
        plt.plot(R, np.zeros_like(R), 'm--', label='Bulbo: No bulge')

    plt.plot(R, v_edr(R, A, R0), 'b-', label='EDR')
    plt.plot(R, modelV, 'r', linewidth=2.2, label='Total EDR')

    plt.title(f"Galaxy: {galaxy_name}")
    plt.xlabel("R (kpc)")
    plt.ylabel("V (km/s)")
    plt.legend()
    plt.grid(True)

    if fname:
        plt.savefig(fname, dpi=200, bbox_inches='tight')

    plt.close()


# ----------------------------------------------------------------------
# EXPORTACIÓN A CSV
# ----------------------------------------------------------------------

def append_result(csv_path, galaxy, result):
    exists = os.path.exists(csv_path)
    df_row = pd.DataFrame([{
        "Galaxy": galaxy,
        "A": result["A"],
        "R0": result["R0"],
        "Yd": result["Yd"],
        "Yb": result["Yb"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "fit_ok": True,
        "mode": "EDR_barions"
    }])

    if exists:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, index=False)
