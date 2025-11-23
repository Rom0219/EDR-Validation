#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_btfr_validation.py — Script Principal para la Validación de la EDR
usando la Relación Tully-Fisher Bariónica (BTFR) con datos SPARC.

Realiza ajustes de curvas de rotación individuales y una regresión global.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import sparc_utils_175 as su # Importa el módulo de utilidades

# -------------------------
# 1) CONSTANTES FÍSICAS Y CONFIGURACIÓN
# -------------------------
# Constante gravitacional G en unidades de (km/s)^2 * kpc / M_sun
G = 4.30091e-6 # (km/s)^2 * kpc / M_sun

# Unidades para la masa bariónica (Mbar)
M_UNIT = 1e9 # Normalizar Log(M) a 10^9 M_sun

# -------------------------
# 2) CARGA DE DATOS Y PREPARACIÓN
# -------------------------

def load_and_merge_data():
    """Carga y combina los datos globales (Table 1) y radiales (Table 2)."""
    print("Cargando datos globales (Table 1) y radiales (Table 2)...")
    try:
        df_global = su.parse_table1(su.TABLE1_FILENAME)
        df_radial = su.parse_table2(su.TABLE2_FILENAME)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de datos. Asegúrese de que '{e.filename}' esté en la ruta esperada.")
        return None, None

    # Limpieza básica: la Tabla 1 contiene L[3.6] (luminosidad en banda 3.6 μm)
    df_global = df_global.dropna(subset=['L[3.6]', 'MHI', 'ID']).reset_index(drop=True)
    
    # Agrupar datos radiales por galaxia
    radial_groups = df_radial.groupby('ID')
    
    print(f"Total de galaxias con datos globales válidos: {len(df_global)}")
    print(f"Total de galaxias con datos radiales: {len(radial_groups.groups)}")

    return df_global, radial_groups

def calculate_btfr_regression(df_results, out_dir):
    """
    Realiza la regresión lineal para la BTFR: Log(Vflat) vs Log(Mbar).
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

    # Utilizamos stats.linregress para la regresión simple
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_M, log_V)
    r_squared = r_value**2
    
    print("\n--- Resultados de la Regresión BTFR (Log-Log) ---")
    print(f"Fórmula: Log(V_flat) = {slope:.3f} * Log(M_bar) + {intercept:.3f}")
    print(f"Coeficiente de determinación (R^2): {r_squared:.3f}")

    # Guardar gráfico BTFR
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

    # Iterar sobre las galaxias que tienen datos globales y radiales
    galaxies_to_process = set(df_global['ID'].values) & set(radial_groups.groups.keys())
    
    print(f"\nIniciando ajuste para {len(galaxies_to_process)} galaxias...")

    for i, gal_id in enumerate(sorted(list(galaxies_to_process))):
        gal_data_radial_df = radial_groups.get_group(gal_id)
        gal_data_global = df_global[df_global['ID'] == gal_id].iloc[0]

        # ---------------------------------------------
        # Preparación de datos para fit_galaxy (simulando rotmod.dat)
        # ---------------------------------------------
        # su.fit_galaxy espera un diccionario con arrays de Numpy (r, Vobs, etc.)
        try:
            data_dict = {
                "r": gal_data_radial_df["R"].values,
                "Vobs": gal_data_radial_df["Vobs"].values,
                "errV": gal_data_radial_df["e_Vobs"].values,
                "Vgas": gal_data_radial_df["Vgas"].values,
                "Vdisk": gal_data_radial_df["Vdisk"].values,
                "Vbul": gal_data_radial_df["Vbul"].values,
            }
        except KeyError as e:
            print(f"Skipping {gal_id}: Missing column {e}. Data source incomplete.")
            continue
        
        # ---------------------------------------------
        # Ajuste del modelo EDR
        # ---------------------------------------------
        try:
            result, Vmodel_plot, sigma_extra, residuals = su.fit_galaxy(data_dict, gal_id)
        except ValueError as e:
            print(f"Skipping {gal_id}: Data error or Vobs uncertainty problem: {e}")
            continue
        except Exception as e:
            print(f"Skipping {gal_id}: Fit failed with error: {e}")
            continue

        if not result["ok"]:
            print(f"Skipping {gal_id}: Fit failed (optimization error).")
            continue

        # ---------------------------------------------
        # Extracción y Cálculo de Magnitudes Físicas
        # ---------------------------------------------
        
        # A) Velocidad Asintótica (Vflat)
        # En el modelo EDR, el parámetro 'A' es una excelente proxy para la V_flat
        Vflat_edr = result["A"]
        log_Vflat = np.log10(Vflat_edr) if Vflat_edr > 0 else np.nan

        # B) Masa Bariónica Total (Mbar)
        # Mbar = Masa Estelar (Disco + Bulbo) + Masa de Gas (HI + H2)
        
        # M_Gas = MHI (de Table 1). Lelli et al. asumen M_H2 negligible para la mayoría
        M_gas = gal_data_global["MHI"] 
        
        # M_Estelar = Luminosidad * Factor M/L
        # Luminosidad L[3.6] (en 10^9 L_sun)
        L_star_9 = gal_data_global["L[3.6]"] # Unidad 10^9 L_sun (en la Tabla 1 original)

        # Usamos los factores M/L ajustados: Yd (Disco) y Yb (Bulbo)
        # El modelo SPARC asume que L[3.6] ya es la suma de L_disk + L_bulge.
        # Para ser precisos, necesitariamos L_disk y L_bulge por separado,
        # pero como estamos ajustando Yd y Yb a las componentes de velocidad (Vdisk, Vbul),
        # podemos calcular la masa estelar total a partir de M_disk y M_bulge implícitas.
        # Los Vdisk y Vbul en Table 2 se basan en un M/L de 1, por lo que:
        # V^2 = G M / R. -> M = R V^2 / G
        # La masa equivalente a Vdisk (para Y=1) es M_disk_unit = R * Vdisk^2 / G
        # Masa Estelar Real = Yd * M_disk_unit + Yb * M_bul_unit
        
        # En el espíritu de BTFR (Mbar = Y*L_star + M_gas), usaremos una simplificación
        # basada en L[3.6] y un factor M/L promedio ponderado, o mejor:
        
        # M_bar = Luminosidad * Y_fit + M_gas (donde Y_fit es un M/L estelar efectivo)
        # Vamos a aproximar la masa estelar usando la relación SPARC original (Lelli et al.)
        # Asumimos que la masa estelar M_star_9 está implícitamente relacionada con L[3.6] y los Yd, Yb
        
        # Simplificación: Usar la masa de disco y bulbo del modelo con Yd y Yb
        # Lelli et al. obtienen M* a partir de L[3.6] usando un M/L fijo.
        # Para la BTFR, usaremos la masa bariónica total:
        
        # M_star = Y_fit * L[3.6]
        # El ajuste EDR nos da Yd y Yb, que son los factores M/L para Vdisk y Vbul.
        # Sin el desglose de Ldisk/Lbul, haremos la aproximación más común para BTFR:
        
        # M_bar_total (en 10^9 M_sun)
        M_star_9 = gal_data_global["L[3.6]"] * result["Yd"] # Aproximación simple usando Yd como M/L estelar promedio.
        Mbar_total_9 = M_star_9 + M_gas
        
        log_Mbar = np.log10(Mbar_total_9) if Mbar_total_9 > 0 else np.nan
        
        # ---------------------------------------------
        # Consolidación de Resultados
        # ---------------------------------------------
        
        # Añadir al registro global
        new_row = {
            "ID": gal_id,
            "MHI_9": M_gas,
            "L3.6_9": gal_data_global["L[3.6]"],
            "A_edr": Vflat_edr, # Vflat proxy
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
             print(f"Procesando {i+1}/{len(galaxies_to_process)}: {gal_id} (Chi2_red: {result['chi2_red']:.2f})")

        # Generar gráficos de ajuste (Curva de rotación y residuales)
        fit_path = out_dir / f"{gal_id}_fit.png"
        su.plot_fit_with_residuals(data_dict, Vmodel_plot, result, str(fit_path), gal_id)
        
        # Generar histograma de residuales local
        hist_path = out_dir / f"{gal_id}_hist.png"
        su.plot_residual_histogram_single(residuals, str(hist_path), gal_id)

    # -------------------------
    # 4) RESULTADOS GLOBALES Y BTFR
    # -------------------------
    df_results = pd.DataFrame(all_results)
    
    # Generar histograma global de residuales
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
