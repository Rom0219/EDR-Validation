#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_btfr_validation_175.py — Script Principal para la Validación de la EDR
usando la Relación Tully-Fisher Bariónica (BTFR) con datos SPARC.

Realiza ajustes de curvas de rotación individuales y una regresión global.
Depende del módulo de utilidades 'sparc_utils_175.py'.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
# Importar el módulo de utilidades con el sufijo 175
import sparc_utils_175 as su 

# -------------------------
# 1) CONSTANTES FÍSICAS Y CONFIGURACIÓN
# -------------------------
# Constante gravitacional G en unidades de (km/s)^2 * kpc / M_sun
G = 4.30091e-6 # (km/s)^2 * kpc / M_sun

# -------------------------
# 2) CARGA DE DATOS Y PREPARACIÓN
# -------------------------

def load_and_merge_data():
    """Carga y combina los datos globales (Table 1) y radiales (Table 2)."""
    print("Cargando datos globales (Table 1) y radiales (Table 2)...")
    try:
        # Asume que los nombres de archivo están definidos en sparc_utils_175
        df_global = su.parse_table1(su.TABLE1_FILENAME)
        df_radial = su.parse_table2(su.TABLE2_FILENAME)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de datos. Asegúrese de que '{e.filename}' esté en la ruta esperada.")
        return None, None

    # Limpieza básica: solo considerar galaxias con datos clave
    df_global = df_global.dropna(subset=['L[3.6]', 'MHI', 'ID']).reset_index(drop=True)
    
    # Agrupar datos radiales por galaxia para facilitar la iteración
    radial_groups = df_radial.groupby('ID')
    
    print(f"Total de galaxias con datos globales válidos: {len(df_global)}")
    print(f"Total de galaxias con datos radiales: {len(radial_groups.groups)}")

    return df_global, radial_groups

def calculate_btfr_regression(df_results, out_dir):
    """
    Realiza la regresión lineal para la BTFR: Log(Vflat) vs Log(Mbar).
    Genera el gráfico de la BTFR.
    """
    df_btfr = df_results.copy()
    
    # Filtrar resultados válidos
    df_btfr = df_btfr.dropna(subset=['log_Mbar', 'log_Vflat'])
    
    if len(df_btfr) < 2:
        print("ERROR: Menos de 2 puntos válidos para realizar la regresión BTFR.")
        return

    # Regresión lineal: Log(V) = a * Log(M) + b
    log_M = df_btfr['log_Mbar'].values
    log_V = df_btfr['log_Vflat'].values

    # Utilizamos stats.linregress para la regresión
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_M, log_V)
    r_squared = r_value**2
    
    print("\n--- Resultados de la Regresión BTFR (Log-Log) ---")
    print(f"Fórmula: $\\mathrm{Log}(V_{\\mathrm{flat}}) = {slope:.3f} \\times \\mathrm{Log}(M_{\\mathrm{bar}}) + {intercept:.3f}$")
    print(f"Coeficiente de determinación ($R^2$): {r_squared:.3f}")

    # Guardar gráfico BTFR (utilizando la función de utilidades)
    btfr_path = Path(out_dir) / "btfr_edr_validation.png"
    su.plot_btfr(log_M, log_V, slope, intercept, r_squared, str(btfr_path))
    print(f"Gráfico BTFR guardado en: {btfr_path}")

# -------------------------
# 3) FUNCIÓN PRINCIPAL DE EJECUCIÓN
# -------------------------

def main():
    """Función principal para ejecutar el ajuste y la validación BTFR."""
    out_dir = Path("./btfr_results")
    out_dir.mkdir(exist_ok=True)
    
    df_global, radial_groups = load_and_merge_data()
    if df_global is None:
        return

    all_results = []
    all_residuals = []

    # Identificar galaxias que tienen datos en ambas tablas
    galaxies_to_process = set(df_global['ID'].values) & set(radial_groups.groups.keys())
    
    print(f"\nIniciando ajuste para {len(galaxies_to_process)} galaxias...")

    for i, gal_id in enumerate(sorted(list(galaxies_to_process))):
        gal_data_radial_df = radial_groups.get_group(gal_id)
        gal_data_global = df_global[df_global['ID'] == gal_id].iloc[0]

        # ---------------------------------------------
        # Preparación de datos para fit_galaxy 
        # ---------------------------------------------
        try:
            # Diccionario con los datos radiales necesarios para el ajuste
            data_dict = {
                "r": gal_data_radial_df["R"].values,
                "Vobs": gal_data_radial_df["Vobs"].values,
                "errV": gal_data_radial_df["e_Vobs"].values,
                "Vgas": gal_data_radial_df["Vgas"].values,
                "Vdisk": gal_data_radial_df["Vdisk"].values,
                "Vbul": gal_data_radial_df["Vbul"].values,
            }
        except KeyError as e:
            print(f"Saltando {gal_id}: Columna faltante {e}. Fuente de datos incompleta.")
            continue
        
        # ---------------------------------------------
        # Ajuste del modelo EDR + Bariones
        # ---------------------------------------------
        try:
            result, Vmodel_plot, sigma_extra, residuals = su.fit_galaxy(data_dict, gal_id)
        except ValueError as e:
            print(f"Saltando {gal_id}: Error en los datos o en la incertidumbre de Vobs: {e}")
            continue
        except Exception as e:
            print(f"Saltando {gal_id}: El ajuste falló con el error: {e}")
            continue

        if not result["ok"]:
            # Esto maneja el caso donde la optimización no converge
            print(f"Saltando {gal_id}: El ajuste falló (error de optimización).")
            continue

        # ---------------------------------------------
        # Extracción y Cálculo de Magnitudes Físicas
        # ---------------------------------------------
        
        # A) Velocidad Asintótica (Vflat)
        # Usamos el parámetro A de la EDR como el proxy de la V_flat
        Vflat_edr = result["A"]
        log_Vflat = np.log10(Vflat_edr) if Vflat_edr > 0 else np.nan

        # B) Masa Bariónica Total (Mbar)
        
        # M_Gas (MHI en 10^9 M_sun)
        M_gas = gal_data_global["MHI"] 
        
        # M_Estelar (Aproximación para BTFR)
        # L[3.6] está en 10^9 L_sun (según la Tabla 1 original)
        L_star_9 = gal_data_global["L[3.6]"] 

        # Se usa el factor M/L del disco (Yd_fit) como un M/L estelar efectivo promedio 
        # para toda la luminosidad estelar total (Lelli et al., 2016, usan un M/L fijo, 
        # pero aquí usamos el ajustado por EDR)
        M_star_9 = L_star_9 * result["Yd"] 
        Mbar_total_9 = M_star_9 + M_gas # Mbar en 10^9 M_sun
        
        log_Mbar = np.log10(Mbar_total_9) if Mbar_total_9 > 0 else np.nan
        
        # ---------------------------------------------
        # Consolidación de Resultados
        # ---------------------------------------------
        
        # Añadir al registro global
        new_row = {
            "ID": gal_id,
            "MHI_9": M_gas,
            "L3.6_9": gal_data_global["L[3.6]"],
            "A_edr": Vflat_edr, 
            "R0": result["R0"],
            "Yd_fit": result["Yd"],
            "Yb_fit": result["Yb"],
            "log_Mbar": log_Mbar,
            "log_Vflat": log_Vflat,
            "chi2_red": result["chi2_red"],
            "sigma_extra": sigma_extra
        }
        all_results.append(new_row)
        all_residuals.append(residuals)

        # ---------------------------------------------
        # Ploteo Local
        # ---------------------------------------------
        if (i + 1) % 10 == 0 or i == len(galaxies_to_process) - 1:
             print(f"Procesando {i+1}/{len(galaxies_to_process)}: {gal_id} ($\chi^2_{{\\mathrm{{red}}}}$: {result['chi2_red']:.2f})")

        # Guardar gráficos de ajuste (Curva de rotación y residuales)
        fit_path = out_dir / f"{gal_id}_fit.png"
        su.plot_fit_with_residuals(data_dict, Vmodel_plot, result, str(fit_path), gal_id)
        
        # Guardar histograma de residuales local
        hist_path = out_dir / f"{gal_id}_hist.png"
        su.plot_residual_histogram_single(residuals, str(hist_path), gal_id)

    # -------------------------
    # 4) RESULTADOS GLOBALES Y BTFR
    # -------------------------
    df_results = pd.DataFrame(all_results)
    
    # Generar histograma global de residuales (de todas las galaxias)
    global_hist_path = out_dir / "global_residuals_hist.png"
    su.plot_residuals_hist_global(all_residuals, str(global_hist_path))
    print(f"\nHistograma global de residuales guardado en: {global_hist_path}")

    # Exportar tabla de resultados
    results_csv_path = out_dir / "edr_fit_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"Tabla de resultados guardada en: {results_csv_path}")

    # Validación BTFR
    calculate_btfr_regression(df_results, out_dir)

if __name__ == "__main__":
    main()
