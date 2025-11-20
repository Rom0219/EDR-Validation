# sparc_fit.py
# Versión corregida e integrada: suma bariones + ajuste EDR + nuisance params
# ------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------------
# Modelo EDR (fenomenológico)
# ------------------------------
def v_edr_model(r, A, R0):
    """
    Modelo fenomenológico EDR:
        v_EDR(r) = A * (1 - exp(-r / R0))
    A: velocidad asintótica [km/s]
    R0: escala radial [kpc]
    """
    r = np.asarray(r, dtype=float)
    # evitar división por cero numérica
    R0 = np.maximum(R0, 1e-6)
    return A * (1.0 - np.exp(-r / R0))


# ------------------------------
# Lector robusto SPARC *_rotmod.dat
# ------------------------------
def load_rotmod_generic(path):
    """
    Lector definitivo para archivos SPARC tipo *_rotmod.dat (sin header real).
    Define columnas estándar:
      Rad   Vobs    errV    Vgas    Vdisk   Vbul    SBdisk  SBbul
    Devuelve dict con arrays y metadatos.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe archivo: {path}")

    # Leer ignorando líneas que empiezan con '#', sin header
    df = pd.read_csv(path, comment="#", header=None, sep=r"\s+", engine="python")

    # SPARC típicamente tiene 8 columnas; aceptamos >=3
    if df.shape[1] < 3:
        raise ValueError(f"Formato no reconocido en {path} (solo {df.shape[1]} columnas)")

    # Asignar nombres (si menos de 8, tomar las primeras)
    cols = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    df.columns = cols[:df.shape[1]]

    # dropna por si hay filas con NaN
    df = df.dropna()

    # Extraer arrays (si faltan algunas columnas, rellenar con ceros)
    def get_col(name):
        return df[name].values.astype(float) if name in df.columns else np.zeros(len(df))

    r = get_col("Rad")
    v_obs = get_col("Vobs")
    v_err = get_col("errV")
    v_gas = get_col("Vgas")
    v_disk = get_col("Vdisk")
    v_bul = get_col("Vbul")

    return {
        "R": r,
        "Vobs": v_obs,
        "eV": v_err,
        "Vgas": v_gas,
        "Vdisk": v_disk,
        "Vbul": v_bul,
        "file": os.path.basename(path)
    }


# ------------------------------
# Modelo total: suma bariones + EDR
# ------------------------------
def v_total_model_from_components(r, A, R0, Yd, Yb, Vgas, Vdisk, Vbul):
    """
    Calcula v_total dado A,R0,y factores M/L (Yd, Yb) y componentes precomputadas Vgas,Vdisk,Vbul.
    Devuelve velocidad total v(r).
    """
    # componentes bariónicas multiplicadas por Y (M/L)
    v_gas = Vgas  # gas ya en km/s (no escala por Y)
    v_disk = Yd * Vdisk if Vdisk is not None else 0.0 * Vgas
    v_bul = Yb * Vbul if Vbul is not None else 0.0 * Vgas

    v_bary_sq = v_gas**2 + v_disk**2 + v_bul**2
    v_edr = v_edr_model(r, A, R0)
    v_total = np.sqrt(np.clip(v_bary_sq + v_edr**2, 0.0, np.inf))
    return v_total


# ------------------------------
# Envoltorio para curve_fit con número variable de parámetros
# ------------------------------
def _make_model_function(Vgas, Vdisk, Vbul, include_Yd=True, include_Yb=True, include_jitter=False):
    """
    Crea una función f(r, *params) -> v_model que curve_fit puede usar.
    params order:
      if include_Yd and include_Yb and include_jitter:
        A, R0, Yd, Yb, sigma_extra
      if include_Yd and include_Yb and not jitter:
        A, R0, Yd, Yb
      if only Yd:
        A, R0, Yd
      if none Y:
        A, R0
    Note: sigma_extra se usa para "inflar" errores, no para modelar v (se devuelve v).
    """
    def f(r, *params):
        p = list(params)
        A = p.pop(0)
        R0 = p.pop(0)
        Yd = p.pop(0) if include_Yd else 1.0
        Yb = p.pop(0) if include_Yb else 0.0
        # ignore sigma_extra for model; it's used in fit weighting externally
        return v_total_model_from_components(r, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)
    return f


# ------------------------------
# AJUSTE: fit_edr_rotation_curve
# ------------------------------
def fit_edr_rotation_curve(R, Vobs, eV, Vgas=None, Vdisk=None, Vbul=None,
                           fit_Yd=True, fit_Yb=True, fit_jitter=True,
                           bounds=None, p0=None, maxfev=50000):
    """
    Ajusta la curva total (bariones + EDR) a los datos observados.
    Argumentos:
      R, Vobs, eV : arrays observados (kpc, km/s, km/s)
      Vgas, Vdisk, Vbul: arrays con contribuciones rotacionales ya calculadas (km/s)
      fit_Yd, fit_Yb: si True ajusta Y_disk, Y_bulge; si False usa Yd=1,Yb=1 (o 0 si absent)
      fit_jitter: si True añade sigma_extra en cuadratura a eV para permitir ruido no modelado (sigma fitted)
    Retorna:
      diccionario con parámetros, errores y métricas.
    """
    R = np.asarray(R, dtype=float)
    Vobs = np.asarray(Vobs, dtype=float)
    eV = np.asarray(eV, dtype=float)
    N = len(R)

    # Saneamiento de componentes
    Vgas = np.asarray(Vgas, dtype=float) if Vgas is not None else np.zeros_like(R)
    Vdisk = np.asarray(Vdisk, dtype=float) if Vdisk is not None else np.zeros_like(R)
    Vbul = np.asarray(Vbul, dtype=float) if Vbul is not None else np.zeros_like(R)

    # Decide qué parámetros ajustar
    include_Yd = fit_Yd and np.any(Vdisk != 0.0)
    include_Yb = fit_Yb and np.any(Vbul != 0.0)

    # Construir función para curve_fit (sin jitter en la función)
    model_fun = _make_model_function(Vgas, Vdisk, Vbul, include_Yd=include_Yd, include_Yb=include_Yb)

    # Inicial guesses
    A0 = np.nanmax(Vobs) if np.any(np.isfinite(Vobs)) else 100.0
    R00 = np.median(R) if np.any(R>0) else 1.0
    Yd0 = 0.5
    Yb0 = 0.7

    # Parameter vector initial
    p0_list = [A0, R00]
    if include_Yd: p0_list.append(Yd0)
    if include_Yb: p0_list.append(Yb0)

    # Bounds: reasonable physical bounds
    lower = [0.0, 1e-3]
    upper = [1000.0, 1e3]
    if include_Yd:
        lower.append(0.0); upper.append(5.0)
    if include_Yb:
        lower.append(0.0); upper.append(10.0)

    # Prepare sigma for curve_fit (we'll optionally inflate with jitter if fit_jitter True)
    sigma_used = np.copy(eV)
    sigma_extra = 0.0  # track sigma_extra if used

    # Try full fit (without fitting jitter). If fit_jitter True, we will try to estimate sigma_extra by iterating.
    try:
        popt, pcov = curve_fit(
            model_fun,
            R,
            Vobs,
            sigma=sigma_used,
            absolute_sigma=True,
            p0=p0_list,
            bounds=(lower, upper),
            maxfev=maxfev
        )
        success_mode = "full_fit_no_jitter"
    except Exception as e:
        # Fallback: try simpler two-parameter fit (A,R0 only) using only EDR and ignoring Ys
        try:
            popt2, pcov2 = curve_fit(
                lambda r, A, R0: v_edr_model(r, A, R0),
                R, Vobs, sigma=sigma_used, absolute_sigma=True,
                p0=[A0, R00], maxfev=maxfev
            )
            popt = np.array([popt2[0], popt2[1]] + ([1.0] if include_Yd else []) + ([0.0] if include_Yb else []))
            pcov = np.zeros((len(popt), len(popt)))
            success_mode = "fallback_edr_only"
        except Exception as e2:
            # complete failure
            return {
                "fit_ok": False,
                "reason": f"Fit failed: {e}; fallback failed: {e2}"
            }

    # Extract parameters in canonical order
    # popt ordering: [A, R0, (Yd?), (Yb?)]
    idx = 0
    A = popt[idx]; idx += 1
    R0 = popt[idx]; idx += 1
    Yd = popt[idx] if include_Yd else 1.0; idx += 1 if include_Yd else 0
    Yb = popt[idx] if include_Yb else 0.0; idx += 1 if include_Yb else 0

    # Covariance handling and errors (if pcov all zeros, set errs to nan)
    try:
        perr = np.sqrt(np.diag(pcov))
    except Exception:
        perr = np.full(len(popt), np.nan)

    # Map errors
    idx = 0
    Aerr = perr[idx] if len(perr) > idx else np.nan; idx += 1
    R0err = perr[idx] if len(perr) > idx else np.nan; idx += 1
    Yderr = perr[idx] if include_Yd and len(perr) > idx else np.nan; idx += 1 if include_Yd else 0
    Yberr = perr[idx] if include_Yb and len(perr) > idx else np.nan; idx += 1 if include_Yb else 0

    # Compute model and chi2 with final params
    v_model = v_total_model_from_components(R, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)
    # If fit_jitter True, estimate sigma_extra so that reduced chi2 ~ 1 (iterative)
    chi2 = np.sum(((Vobs - v_model) / sigma_used)**2)
    dof = max(1, N - len(popt))
    chi2_red = chi2 / dof

    # If requested, try to fit a sigma_extra by inflating sigma_used to make chi2_red ~ 1
    sigma_extra = 0.0
    if fit_jitter and chi2_red > 1.2:
        # estimate sigma_extra such that sum(((Vobs-v)/sqrt(eV^2+sigma_extra^2))^2)/(dof)=1
        # solve for sigma_extra via simple root find (bisection)
        def chi2_with_sigma(s):
            s2 = eV**2 + s**2
            return np.sum(((Vobs - v_model) / np.sqrt(s2))**2) / dof

        # bisection between 0 and up to e.g. 500 km/s
        lo, hi = 0.0, 500.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            val = chi2_with_sigma(mid)
            if val > 1.0:
                lo = mid
            else:
                hi = mid
        sigma_extra = 0.5*(lo+hi)
        # recompute final chi2 with sigma_extra
        s2 = eV**2 + sigma_extra**2
        chi2 = np.sum(((Vobs - v_model) / np.sqrt(s2))**2)
        chi2_red = chi2 / dof

    result = {
        "fit_ok": True,
        "A": float(A), "Aerr": float(Aerr),
        "R0": float(R0), "R0err": float(R0err),
        "Yd": float(Yd) if include_Yd else np.nan,
        "Yderr": float(Yderr) if include_Yd else np.nan,
        "Yb": float(Yb) if include_Yb else np.nan,
        "Yberr": float(Yberr) if include_Yb else np.nan,
        "sigma_extra": float(sigma_extra),
        "chi2": float(chi2),
        "chi2_red": float(chi2_red),
        "Ndata": int(N),
        "Ndof": int(dof),
        "mode": success_mode
    }

    return result


# ------------------------------
# plot_fit: incluye contribuciones
# ------------------------------
def plot_fit(data, fitres, fname="plot.png", show_residuals=True):
    """
    Grafica datos (con errores), contribuciones bariónicas, contribución EDR, y total.
    data: dict retornado por load_rotmod_generic
    fitres: dict retornado por fit_edr_rotation_curve
    """
    R = data["R"]
    Vobs = data["Vobs"]
    eV = data["eV"]
    Vgas = data["Vgas"] if "Vgas" in data else np.zeros_like(R)
    Vdisk = data["Vdisk"] if "Vdisk" in data else np.zeros_like(R)
    Vbul = data["Vbul"] if "Vbul" in data else np.zeros_like(R)

    # Ensure fit successful
    if not fitres.get("fit_ok", False):
        # simple plot of data
        plt.figure(figsize=(7,5))
        plt.errorbar(R, Vobs, yerr=eV, fmt='o')
        plt.title(f"Data only — {data.get('file','')}")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        return

    A = fitres["A"]; R0 = fitres["R0"]
    Yd = fitres.get("Yd", 1.0) if not np.isnan(fitres.get("Yd", np.nan)) else 1.0
    Yb = fitres.get("Yb", 0.0) if not np.isnan(fitres.get("Yb", np.nan)) else 0.0

    r_plot = np.linspace(min(R), max(R), 600)
    v_edr_plot = v_edr_model(r_plot, A, R0)
    v_disk_plot = (Yd * Vdisk) if Vdisk is not None else np.zeros_like(r_plot)
    v_gas_plot = Vgas if Vgas is not None else np.zeros_like(r_plot)
    v_bul_plot = (Yb * Vbul) if Vbul is not None else np.zeros_like(r_plot)

    # For continuous plotting of baryonic profiles we interpolate the provided discrete components
    from scipy.interpolate import interp1d
    def interp_arr(xin, yin):
        try:
            f = interp1d(R, yin, bounds_error=False, fill_value=(yin[0], yin[-1]))
            return f(r_plot)
        except Exception:
            return np.interp(r_plot, R, yin)

    v_disk_c = interp_arr(R, Yd * Vdisk)
    v_gas_c = interp_arr(R, Vgas)
    v_bul_c = interp_arr(R, Yb * Vbul)

    v_total_plot = np.sqrt(np.clip(v_gas_c**2 + v_disk_c**2 + v_bul_c**2 + v_edr_plot**2, 0.0, np.inf))

    # Plotting
    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(2,1, height_ratios=[3,1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax.errorbar(R, Vobs, yerr=eV, fmt='o', label='Datos', color='k', alpha=0.8)
    ax.plot(r_plot, v_total_plot, label='Total (bary + EDR)', lw=2)
    ax.plot(r_plot, v_edr_plot, '--', label='EDR (A,R0)', lw=1.6)
    ax.plot(r_plot, v_disk_c, ':', label=f'Disk (Yd={Yd:.2f})')
    if np.any(Vbul != 0.0):
        ax.plot(r_plot, v_bul_c, ':', label=f'Bulge (Yb={Yb:.2f})')
    ax.plot(r_plot, v_gas_c, ':', label='Gas')

    ax.set_ylabel('v [km/s]')
    ax.set_xlim(min(r_plot), max(r_plot))
    ax.legend()
    ax.grid(True)
    ax.set_title(f"{data.get('file','')} — A={A:.2f}, R0={R0:.2f}, chi2_red={fitres.get('chi2_red',np.nan):.2f}")

    # Residuals
    axr = fig.add_subplot(gs[1], sharex=ax)
    v_model_at_R = np.interp(R, r_plot, v_total_plot)
    resid = Vobs - v_model_at_R
    axr.axhline(0, color='0.3', lw=1)
    axr.errorbar(R, resid, yerr=eV, fmt='o', color='k', alpha=0.8)
    axr.set_ylabel('Obs - Model [km/s]')
    axr.set_xlabel('R [kpc]')
    axr.grid(True)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()


# ------------------------------
# Guardar resultados a CSV (añadir fila)
# ------------------------------
def save_fit_result(result, galaxy, out_csv):
    """
    Añade una fila al CSV out_csv con los parámetros del ajuste.
    El diccionario result puede contener:
      A, Aerr, R0, R0err, Yd, Yderr, Yb, Yberr, sigma_extra, chi2, chi2_red, mode
    """
    row = {
        "Galaxy": galaxy,
        "fit_ok": result.get("fit_ok", False),
        "mode": result.get("mode", ""),
        "A": result.get("A", np.nan),
        "Aerr": result.get("Aerr", np.nan),
        "R0": result.get("R0", np.nan),
        "R0err": result.get("R0err", np.nan),
        "Yd": result.get("Yd", np.nan),
        "Yderr": result.get("Yderr", np.nan),
        "Yb": result.get("Yb", np.nan),
        "Yberr": result.get("Yberr", np.nan),
        "sigma_extra": result.get("sigma_extra", np.nan),
        "chi2": result.get("chi2", np.nan),
        "chi2_red": result.get("chi2_red", np.nan),
        "Ndata": result.get("Ndata", np.nan),
        "Ndof": result.get("Ndof", np.nan)
    }

    # Write append-safe
    import csv
    header_exists = os.path.isfile(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------
# Opcional: bootstrap errors (simple)
# ------------------------------
def bootstrap_errors(data, fitres, nboot=200, fit_kwargs=None):
    """
    Estima errores por bootstrap resampleando puntos y refiteando el modelo
    fit_kwargs se pasa a fit_edr_rotation_curve si se quiere ajustar Yd/Yb allí.
    Retorna dict con desviaciones estándar empíricas para A,R0,Yd,Yb.
    """
    R = data["R"]; Vobs = data["Vobs"]; eV = data["eV"]
    Vgas = data.get("Vgas", np.zeros_like(R))
    Vdisk = data.get("Vdisk", np.zeros_like(R))
    Vbul = data.get("Vbul", np.zeros_like(R))

    A_samples = []
    R0_samples = []
    Yd_samples = []
    Yb_samples = []

    for i in range(nboot):
        idx = np.random.randint(0, len(R), len(R))
        Ri, Vi, ei = R[idx], Vobs[idx], eV[idx]
        try:
            res = fit_edr_rotation_curve(Ri, Vi, ei, Vgas=Vgas[idx], Vdisk=Vdisk[idx], Vbul=Vbul[idx], **(fit_kwargs or {}))
            if res.get("fit_ok", False):
                A_samples.append(res["A"])
                R0_samples.append(res["R0"])
                if "Yd" in res and not np.isnan(res["Yd"]): Yd_samples.append(res["Yd"])
                if "Yb" in res and not np.isnan(res["Yb"]): Yb_samples.append(res["Yb"])
        except Exception:
            continue

    out = {}
    out["A_bs_err"] = float(np.std(A_samples)) if len(A_samples)>0 else np.nan
    out["R0_bs_err"] = float(np.std(R0_samples)) if len(R0_samples)>0 else np.nan
    out["Yd_bs_err"] = float(np.std(Yd_samples)) if len(Yd_samples)>0 else np.nan
    out["Yb_bs_err"] = float(np.std(Yb_samples)) if len(Yb_samples)>0 else np.nan
    return out


# End of file
