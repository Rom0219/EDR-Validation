#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
residuals_analysis_175.py — Análisis de los residuos (Vobs - Vfit) y validación
de la relación barionica de Tully-Fisher (BTFR) usando los resultados del ajuste EDR
para las 175 galaxias SPARC.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
import seaborn as sns

# Configuración de Matplotlib y Seaborn
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# -------------------------
# 1) CONFIGURACIÓN Y CARGA DE DATOS
# -------------------------
RESULTS_DIR = Path("./btfr_results")
# Nombre del archivo de resultados generado por 'run_btfr_validation_175.py'
RESULTS_FILENAME = RESULTS_DIR / "edr_fit_results.csv"
OUTPUT_DIR = RESULTS_DIR / "analysis_plots_175"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Carga los resultados del ajuste EDR y prepara los datos para el análisis."""
    try:
        df = pd.read_csv(RESULTS_FILENAME)
        print(f"Resultados de ajuste cargados para {len(df)} galaxias.")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de resultados en '{RESULTS_FILENAME}'.")
        print("Por favor, ejecute 'run_btfr_validation_175.py' primero.")
        return None

# -------------------------
# 2) ANÁLISIS DE RESIDUOS (Vobs - Vfit)
# -------------------------

def plot_residuals(df):
    """Genera histogramas y boxplots para visualizar la distribución de los residuos."""
    
    # Calculamos el residuo estandarizado: (Vobs - Vfit) / sigma_V
    # sigma_V = incertidumbre en Vobs (e_Vobs)
    # df['V_res'] = df['Vobs'] - df['Vfit']
    
    # El residuo de interés aquí es el promedio de todos los puntos por galaxia
    df['V_res_mean'] = df['Vobs_list'].apply(lambda x: np.mean(np.array(eval(x)) - np.array(eval(df.loc[df['ID'] == x, 'Vfit_list'].iloc[0]))))
    
    residuals = df.dropna(subset=['V_res_mean'])['V_res_mean']

    if residuals.empty:
        print("Advertencia: No hay suficientes datos de residuos para graficar.")
        return

    plt.figure(figsize=(10, 6))
    
    # Histograma de los residuos promedio por galaxia
    sns.histplot(residuals, bins=30, kde=True, color='skyblue', edgecolor='black', zorder=2)
    
    # Media y Desviación Estándar de los residuos
    mean_res = residuals.mean()
    std_res = residuals.std()
    
    plt.axvline(mean_res, color='r', linestyle='--', label=f'Media: {mean_res:.2f} km/s')
    plt.axvline(mean_res + std_res, color='g', linestyle=':', label=f'1 $\sigma$: $\pm{std_res:.2f}$ km/s')
    plt.axvline(mean_res - std_res, color='g', linestyle=':')
    
    plt.title('Distribución de los Residuos Promedio ($V_{\\mathrm{obs}} - V_{\\mathrm{fit}}$)')
    plt.xlabel('Residuo Promedio (km/s)')
    plt.ylabel('Frecuencia (Galaxias)')
    plt.legend()
    
    plot_path = OUTPUT_DIR / "hist_average_residuals.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  - Histograma de Residuos guardado: {plot_path.name}")


# -------------------------
# 3) ANÁLISIS BTFR (Relación Barionica de Tully-Fisher)
# -------------------------

def plot_btfr(df):
    """
    Genera el diagrama de la Relación Barionica de Tully-Fisher (BTFR):
    log_Mbar vs. log_Vflat
    """
    
    # Filtrar datos que son necesarios
    btfr_data = df.dropna(subset=['log_Mbar', 'Vflat_fit', 'e_Vflat_fit'])

    if btfr_data.empty:
        print("Advertencia: No hay suficientes datos para el análisis BTFR.")
        return
        
    # Calcular log_Vflat (Vflat_fit es la velocidad asintótica de la curva de ajuste EDR)
    btfr_data['log_Vflat'] = np.log10(btfr_data['Vflat_fit'])
    
    # ---------------------------
    # 3.1) Regresión Lineal (BTFR)
    # ---------------------------
    
    # Utilizamos Mbar como variable independiente (X)
    X = btfr_data['log_Mbar']
    Y = btfr_data['log_Vflat']
    
    # Realizar la regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    r_squared = r_value**2
    
    # ---------------------------
    # 3.2) Gráfico BTFR
    # ---------------------------
    plt.figure(figsize=(10, 8))
    
    # Diagrama de dispersión con barras de error
    # Usar incertidumbre en Vflat (e_Vflat_fit) como error en Y
    # Note: No podemos incluir incertidumbre en Mbar (X) con plt.errorbar fácilmente
    plt.errorbar(btfr_data['log_Mbar'], btfr_data['log_Vflat'], 
                 yerr=btfr_data['e_Vflat_fit'] / (btfr_data['Vflat_fit'] * np.log(10)), # Propagación de error para log10
                 fmt='o', color='gray', ecolor='lightgray', capsize=3, zorder=1, alpha=0.6)
                 
    sns.scatterplot(x=X, y=Y, color='navy', s=50, zorder=2)

    # Línea de regresión
    x_range = np.linspace(X.min(), X.max(), 100)
    y_fit = slope * x_range + intercept
    
    plt.plot(x_range, y_fit, 'r--', label=f'Ajuste Lineal:\n $\log V = {slope:.2f} \log M + {intercept:.2f}$\n $R^2 = {r_squared:.3f}$', zorder=3)
    
    plt.title('Relación Bariónica de Tully-Fisher (BTFR) con Velocidades EDR')
    plt.xlabel('$\log_{10} M_{\\mathrm{bar}}$ ($M_{\\odot}$)')
    plt.ylabel('$\log_{10} V_{\\mathrm{flat}}$ (km/s)')
    plt.legend(loc='lower right')
    
    plot_path = OUTPUT_DIR / "btfr_edr_fit.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n  - Gráfico BTFR guardado: {plot_path.name}")
    print(f"  - Regresión BTFR: Pendiente = {slope:.3f}, Intercepto = {intercept:.3f}, $R^2$ = {r_squared:.3f}")
    
    # Recomendación para el siguiente paso
    print("\n--- NOTA IMPORTANTE ---")
    print("Para el resumen final de los parámetros, ejecute el script:")
    print("sparc_results_summary_175.py")


# -------------------------
# 4) FUNCIÓN PRINCIPAL
# -------------------------

def main():
    """Ejecuta el pipeline de análisis de residuos y BTFR."""
    
    df = load_data()
    if df is None:
        return
        
    print("\n--- Iniciando Análisis de Residuos ---")
    plot_residuals(df)
    
    print("\n--- Iniciando Análisis BTFR ---")
    plot_btfr(df)

    print("\nAnálisis de residuos y BTFR completados. Revise la carpeta 'btfr_results/analysis_plots_175'.")

if __name__ == "__main__":
    main()
