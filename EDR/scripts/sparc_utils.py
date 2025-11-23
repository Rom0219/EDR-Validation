#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_utils.py — Librería de Utilidades para la Validación EDR + SPARC

Contiene:
1. Constantes de ruta.
2. Funciones de Parseo para Tablas 1 y 2 (Lelli et al. 2016).
3. Función de carga de datos radiales (load_rotmod_generic).
4. Modelo EDR y algoritmo de ajuste (fit_galaxy, find_sigma_extra).
5. Funciones de ploteo para resultados locales y globales.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, brentq
from scipy import stats # Necesario para BTFR y regresiones

# -------------------------
# 1) CONFIGURACIÓN Y CONSTANTES
# -------------------------
# Nombres de archivos de datos:
# ¡CORRECCIÓN FINAL! Usando los nombres exactos que subiste:
TABLE1_FILENAME = "SPARC_Lelli2016_Table1.txt" 
TABLE2_FILENAME = "SPARC_Lelli2016_Table2.txt"

# Ruta de referencia (mencionada en sparc_fit.py)
REFERENCE_PDF = "/mnt/data/FORMULAS_V2.pdf"

# -------------------------
# 2) PARSERS DE DATOS DE SPARC (Tablas 1 y 2)
# -------------------------

def parse_table1(txt_path):
    """
    Parse Table1 (Galaxy Sample) para extraer datos globales.
    Columnas: ID, T, D, L[3.6], MHI, Vflat, etc.
    """
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    rows = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # Detectar líneas que empiezan por ID y luego un número
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+\d+\s+[-+]?\d", s):
            parts = re.split(r"\s+", s)
            rows.append(parts)

    if not rows:
        raise ValueError("No se encontraron filas tipo Table1 en el archivo")

    # Esquema de columnas: puede variar
    cols_base = ["ID","T","D","e_D","f_D","Inc","e_Inc","L[3.6]","e_L[3.6]","Reff","SBeff","Rdisk","SBdisk","MHI","RHI","Vflat","e_Vflat","Q","Ref"]
    maxcols = max(len(r) for r in rows)
    
    cols = cols_base[:maxcols]
    if maxcols > len(cols_base):
        cols = cols_base + [f"extra_{i}" for i in range(len(cols_base), maxcols)]

    norm = [r + [""]*(len(cols)-len(r)) for r in rows]
    df = pd.DataFrame(norm, columns=cols)
    
    # Convertir numéricas (ignorando errores)
    for c in df.columns:
        if c not in ["ID", "Ref"] and not c.startswith("extra"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    return df.dropna(subset=['ID']).reset_index(drop=True)


def parse_table2(txt_path):
    """
    Parse Table2 (Mass Models) desde el TXT.
    Columnas: ID, D, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul.
    """
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    start_idx = 0
    for i, ln in enumerate(lines):
        if "Table:" in ln and "Mass Models" in ln:
            start_idx = i+1
            break
        if "Byte-by-byte Description of file: datafile2" in ln:
            start_idx = i+1
            break

    data_lines = []
    for ln in lines[start_idx:]:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", s):
            parts = re.split(r"\s+", s)
            data_lines.append(parts)

    if not data_lines:
        raise ValueError("No se detectaron líneas de datos tipo Table2 en el archivo.")

    maxcols = max(len(r) for r in data_lines)
    cols_base = ["ID","D","R","Vobs","e_Vobs","Vgas","Vdisk","Vbul","SBdisk","SBbul"]
    
    cols = cols_base[:maxcols]
    if maxcols > len(cols_base):
        cols = cols_base + [f"extra_{i}" for i in range(len(cols_base), maxcols)]

    norm = [r + [""]*(len(cols)-len(r)) for r in data_lines]
    df = pd.DataFrame(norm, columns=cols)
    
    # Convertir numéricas
    for c in df.columns:
        if c != "ID" and not c.startswith("extra"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    return df.dropna(subset=['ID', 'R']).reset_index(drop=True)

def load_free_table(file_path):
    """
    Lector genérico y flexible de tablas separadas por espacios/tabs. (Usado como fallback).
    """
    p = Path(file_path)
    text_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    data_lines = []
    for ln in text_lines:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", s):
            data_lines.append(s)

    if not data_lines:
        raise ValueError(f"No se detectaron líneas de datos en {file_path}")

    rows = [re.split(r"\s+", dl) for dl in data_lines]
    maxcols = max(len(r) for r in rows)
    norm = [r + [""]*(maxcols - len(r)) for r in rows]

    headers = ["col{:02d}".format(i+1) for i in range(maxcols)]
    df = pd.DataFrame(norm, columns=headers)
    return df

# -------------------------
# 3) LECTURA SPARC rotmod (para el Fit Local)
# -------------------------

def load_rotmod_generic(path):
    """
    Carga los datos pre-procesados de una sola galaxia (*_rotmod.dat) para el fit.
    Asume un formato simple de 8 columnas.
    """
    # Intentar leer con pandas que maneja mejor los espacios
    df = pd.read_csv(path, comment="#", sep=r"\s+", skipinitialspace=True)
    
    # Si pandas lee mal, intentamos leer como texto y dividir
    if len(df.columns) < 8:
        # Fallback a lectura básica si el header es problemático
        try:
            data = np.loadtxt(path, comments="#", usecols=range(8))
            if data.ndim == 1: # Caso de una sola fila
                data = data[np.newaxis, :]
            df = pd.DataFrame(data, columns=["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"])
        except Exception:
            raise KeyError(f"Archivo SPARC con columnas insuficientes o formato incorrecto: {path}")

    # Renombrar columnas para estandarización si la lectura fue exitosa (o si era correcta)
    df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    
    # Filtro de seguridad
    mask = (~np.isnan(df["Vobs"])) & (df["errV"] > 0)

    if not np.any(mask):
        raise ValueError("No hay puntos de datos válidos con incertidumbre Vobs > 0.")
        
    return {
        "r": df["r"].values[mask].astype(float),
        "Vobs": df["Vobs"].values[mask].astype(float),
        "errV": df["errV"].values[mask].astype(float),
        "Vgas": df["Vgas"].values[mask].astype(float),
        "Vdisk": df["Vdisk"].values[mask].astype(float),
        "Vbul": df["Vbul"].values[mask].astype(float),
    }

# -------------------------
# 4) MODELO EDR + BARIONES
# -------------------------

def v_edr_component(r, A, R0):
    """Componente de velocidad de la EDR (núcleo + recorte exponencial)."""
    return A * (1.0 - np.exp(-r / R0))

def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    """Velocidad de curva de rotación total (EDR + bariones)."""
    V_edr = v_edr_component(r, A, R0)
    # Combinación de velocidades en cuadratura
    return np.sqrt((Yd * Vdisk)**2 + (Yb * Vbul)**2 + (Vgas)**2 + V_edr**2)

# -------------------------
# 5) ESTIMADOR DE JITTER CONDICIONADO
# -------------------------

def find_sigma_extra(residuals, errV, target_chi2red=1.0):
    """
    Encuentra sigma_extra >= 0 tal que chi2_red == target_chi2red.
    Si chi2_red_raw <= 1, devuelve 0.
    """
    resid = np.asarray(residuals)
    errV = np.asarray(errV)
    # El número de grados de libertad depende de los parámetros del fit_galaxy (4)
    dof = len(resid) - 4 

    if dof <= 0:
        return 0.0

    def chi2_minus_target(sig):
        # Función a la que se le busca la raíz: chi2_red(sig) - target
        denom = errV**2 + sig**2
        # Evitar división por cero o valores negativos en el denominador
        if np.any(denom <= 0):
             return np.inf # Valor grande para forzar a sig a ser positivo
        val = np.sum((resid**2) / denom)
        return val / dof - target_chi2red

    # 1. Comprobar si ya estamos por debajo del objetivo sin jitter (sig=0)
    try:
        if chi2_minus_target(0.0) <= 0:
            return 0.0
    except Exception:
        # Si el cálculo inicial falla (ej. división por cero), asumimos 0.0
        return 0.0

    # 2. Buscar la raíz para sigma_extra > 0
    sig_max = np.std(resid) * 10.0 + np.median(errV)
    try:
        # Utilizamos brentq para encontrar la raíz entre 0 y sig_max
        root = brentq(chi2_minus_target, 1e-12, max(sig_max, 1e-6), maxiter=100, disp=False)
        return float(max(0.0, root))
    except Exception:
        # Fallback si brentq falla (el intervalo no contiene la raíz)
        return 0.0

# -------------------------
# 6) FIT PRINCIPAL
# -------------------------

def fit_galaxy(data, galaxy_name="Galaxy"):
    """
    Ajusta los parámetros [A, R0, Yd, Yb] y calcula sigma_extra condicionado.
    Retorna: (result_dict, Vmodel_plot, sigma_extra, residuals_obs)
    """
    r = data["r"]
    Vobs = data["Vobs"]
    errV = data["errV"]
    Vgas = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul = data["Vbul"]

    # guess and bounds (parámetros EDR y M/L bariónicos)
    p0 = [120.0, 2.0, 0.5, 0.1]
    bounds = ([10.0, 0.01, 0.0, 0.0], [400.0, 20.0, 3.0, 1.0])

    def model_obs(r_arr, A, R0, Yd, Yb):
        return v_model_total(r_arr, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model_obs, r, Vobs,
            sigma=errV,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=30000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # Modelo interpolado para ploteo suave
    r_plot = np.linspace(np.min(r), np.max(r), 300)
    Vdisk_p = np.interp(r_plot, r, Vdisk)
    Vbul_p = np.interp(r_plot, r, Vbul)
    Vgas_p = np.interp(r_plot, r, Vgas)
    Vmodel_plot = v_model_total(r_plot, A, R0, Yd, Yb, Vdisk_p, Vbul_p, Vgas_p)
    
    # Modelo en puntos de observación
    Vmodel_obs = v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    # Residuales y cálculo de jitter
    residuals = Vobs - Vmodel_obs
    dof = max(len(r) - 4, 1)

    sigma_extra = find_sigma_extra(residuals, errV, target_chi2red=1.0)

    # Recomputar chi2 con sigma_extra
    denom = errV**2 + sigma_extra**2
    chi2 = np.sum((residuals**2) / denom)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    mode = "EDR_barions_jitter_conditioned" if sigma_extra > 0 else "EDR_barions_interpolated"

    result = {
        "ok": True, "A": float(A), "R0": float(R0), "Yd": float(Yd), "Yb": float(Yb),
        "Aerr": float(perr[0]) if perr.size > 0 else np.nan, "R0err": float(perr[1]) if perr.size > 1 else np.nan,
        "Yderr": float(perr[2]) if perr.size > 2 else np.nan, "Yberr": float(perr[3]) if perr.size > 3 else np.nan,
        "chi2": float(chi2), "chi2_red": float(chi2_red), "r_plot": r_plot, "mode": mode
    }

    return result, Vmodel_plot, float(sigma_extra), residuals

# -------------------------
# 7) FUNCIONES DE PLOTEO
# -------------------------

def plot_fit_with_residuals(data, Vmodel_plot, result, fname, galaxy_name="Galaxy"):
    """Gráfico de curva de rotación (fit + residuales)."""
    r = data["r"]
    Vobs = data["Vobs"]
    errV = data["errV"]

    # Usar interpolación para los residuales
    Vmodel_interp = np.interp(r, result["r_plot"], Vmodel_plot)
    residuals = Vobs - Vmodel_interp

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=errV, fmt="o", alpha=0.8, label="Observado")
    ax1.plot(result["r_plot"], Vmodel_plot, "-b", lw=2, label=f"EDR model ($\\chi^2_\\nu={result['chi2_red']:.2f}$)")
    ax1.set_ylabel("Velocidad (km/s)")
    ax1.set_title(f"{galaxy_name}")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=errV, fmt="o", color="darkred")
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residual_histogram_single(residuals, fname, galaxy_name="Galaxy"):
    """Histograma de residuales para una sola galaxia."""
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=20, alpha=0.85, edgecolor="black")
    plt.axvline(np.mean(residuals), color="red", linestyle="--", label=f"media={np.mean(residuals):.2f}")
    plt.title(f"Residuales — {galaxy_name}")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residuals_hist_global(residuals_list, fname="hist_global.png", figsize=(15, 12)):
    """Histograma de residuales normalizados a nivel de la muestra."""
    all_res = np.concatenate([np.asarray(x) for x in residuals_list if len(x) > 0])
    plt.figure(figsize=figsize)
    plt.hist(all_res, bins=40, alpha=0.85, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", lw=1.4)
    plt.title("Histograma global de residuales — SPARC + EDR")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    
def plot_btfr(log_M, log_V, slope, intercept, r_squared, out_path):
    """Genera el gráfico de la Relación Tully-Fisher Bariónica (BTFR)."""
    plt.figure(figsize=(7, 6))
    plt.scatter(log_M, log_V, s=20, label="SPARC (EDR/SPS Model)")
    x_fit = np.linspace(log_M.min(), log_M.max(), 100)
    y_fit = intercept + slope * x_fit
    plt.plot(x_fit, y_fit, 'r--', 
             label=f"Regresión Lineal\n$Log(V) = {slope:.2f} Log(M) + {intercept:.2f}$\n$r^2={r_squared:.3f}$")

    plt.xlabel(r'$\log_{10}(M_{bar} / 10^9 M_{\odot})$')
    plt.ylabel(r'$\log_{10}(V_{flat} / km/s)$')
    plt.title(f"Relación Tully-Fisher Bariónica (BTFR) - Validación EDR")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
