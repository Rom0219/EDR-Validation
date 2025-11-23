#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_results_summary.py — Resumen y visualización de los parámetros
de ajuste EDR (A, R0, Yd_fit) y los diagnósticos de calidad (chi2_red, sigma_extra)
para las 175 galaxias SPARC.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de Matplotlib y Seaborn
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# -------------------------
# 1) CONFIGURACIÓN Y CARGA DE DATOS
# -------------------------
RESULTS_DIR = Path("./btfr_results")
RESULTS_FILENAME = RESULTS_DIR / "edr_fit_results.csv"
OUTPUT_DIR = RESULTS_DIR / "summary_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_fit_results():
    """Carga los resultados del ajuste EDR y prepara las columnas para el análisis."""
    try:
        df = pd.read_csv(RESULTS_FILENAME)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de resultados en '{RESULTS_FILENAME}'.")
        print("Por favor, ejecute 'run_btfr_validation_175.py' primero para generar los resultados.")
        return None

    # Las columnas clave deben existir y ser numéricas para el análisis
    required_cols = ['A_edr', 'R0', 'Yd_fit', 'chi2_red', 'sigma_extra', 'log_Mbar']
    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
    
    print(f"Datos cargados y limpios para el resumen: {len(df_clean)} galaxias.")
    
    return df_clean

# -------------------------
# 2) ESTADÍSTICAS DESCRIPTIVAS
# -------------------------

def print_summary_statistics(df):
    """Calcula y muestra estadísticas descriptivas para los parámetros clave."""
    
    parameters = ['A_edr', 'R0', 'Yd_fit', 'chi2_red', 'sigma_extra']
    
    print("\n--- Estadísticas Descriptivas de los Parámetros EDR y Diagnósticos ---")
    
    summary = df[parameters].describe().T
    summary = summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    # Formatear la tabla de resumen
    print(summary.to_markdown(floatfmt=".3f"))
    
    # Comentario sobre los resultados clave
    print(f"\nInterpretación clave:")
    print(f" - Media del $\chi^2_{\\mathrm{red}}$: {summary.loc['chi2_red', 'mean']:.3f}. Valores cercanos a 1 indican un buen ajuste en promedio.")
    print(f" - Media de $\sigma_{\\mathrm{extra}}$: {summary.loc['sigma_extra', 'mean']:.3f} km/s. Representa la dispersión intrínseca promedio.")
    print(f" - Media de $Y_d$ (M/L): {summary.loc['Yd_fit', 'mean']:.3f}. Muestra la relación masa-luz media de los discos.")


# -------------------------
# 3) VISUALIZACIÓN DE DISTRIBUCIONES (HISTOGRAMAS)
# -------------------------

def plot_distributions(df):
    """Genera histogramas y boxplots para visualizar la distribución de los parámetros."""
    
    plot_params = {
        'A_edr': {'title': 'Distribución del Parámetro $A$', 'xlabel': '$A$ (km/s)', 'color': 'navy'},
        'R0': {'title': 'Distribución del Parámetro $R_0$', 'xlabel': '$R_0$ (kpc)', 'color': 'darkgreen'},
        'Yd_fit': {'title': 'Distribución de la Razón Masa-Luz del Disco ($Y_d$)', 'xlabel': '$Y_d$ ($\Upsilon_D$)', 'color': 'purple'},
        'chi2_red': {'title': 'Distribución del $\chi^2_{\\mathrm{red}}$', 'xlabel': '$\chi^2_{\\mathrm{red}}$', 'color': 'tomato'},
        'sigma_extra': {'title': 'Distribución de $\sigma_{\\mathrm{extra}}$', 'xlabel': '$\sigma_{\\mathrm{extra}}$ (km/s)', 'color': 'sienna'},
    }
    
    print("\n--- Generando Gráficos de Distribución ---")
    
    for param, config in plot_params.items():
        plt.figure(figsize=(10, 6))
        
        # Histograma
        sns.histplot(df[param], bins=25, kde=True, color=config['color'], edgecolor='black', zorder=2)
        
        # Opcionalmente añadir media o mediana
        median_val = df[param].median()
        plt.axvline(median_val, color='r', linestyle='--', label=f'Mediana: {median_val:.2f}')
        
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel('Frecuencia')
        plt.legend()
        
        plot_path = OUTPUT_DIR / f"hist_{param}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - Histograma de {param} guardado: {plot_path.name}")

# -------------------------
# 4) VISUALIZACIÓN DE CORRELACIONES (SCATTER PLOTS)
# -------------------------

def plot_correlations(df):
    """Genera diagramas de dispersión de las correlaciones clave."""
    
    # Relaciones teóricas y diagnósticas clave
    correlation_pairs = [
        # 1. R0 vs. A_edr (Relación intrínseca del perfil EDR)
        ('R0', 'A_edr', 'Relación $R_0$ vs $A$ del Perfil EDR', '$R_0$ (kpc)', '$A$ (km/s)', 'viridis'),
        # 2. Yd_fit vs. Propiedad Física (M/L vs. escala de la galaxia)
        ('log_Mbar', 'Yd_fit', 'Razón Masa-Luz ($Y_d$) vs. Masa Bariónica', '$\log_{10} M_{\\mathrm{bar}}$ ($M_{\\odot}$)', '$Y_d$ ($\Upsilon_D$)', 'plasma'),
        # 3. Yd_fit vs. Diagnóstico (M/L y calidad de ajuste)
        ('Yd_fit', 'chi2_red', 'Calidad de Ajuste ($\chi^2_{\\mathrm{red}}$) vs. $Y_d$', '$Y_d$ ($\Upsilon_D$)', '$\chi^2_{\\mathrm{red}}$', 'magma'),
    ]
    
    print("\n--- Generando Gráficos de Correlación (Dispersión) ---")

    for x_param, y_param, title, x_label, y_label, cmap in correlation_pairs:
        plt.figure(figsize=(10, 8))
        
        # Usar chi2_red como color para la mayoría de los plots para ver si el ajuste afecta la relación
        color_data = df['chi2_red'] if 'chi2_red' not in (x_param, y_param) else df['sigma_extra']
        color_label = '$\chi^2_{\\mathrm{red}}$' if 'chi2_red' not in (x_param, y_param) else '$\sigma_{\\mathrm{extra}}$'

        sns.scatterplot(x=df[x_param], y=df[y_param], 
                        hue=color_data, 
                        palette=cmap, 
                        size=color_data,
                        sizes=(20, 200),
                        alpha=0.7, 
                        edgecolor='w', 
                        linewidth=0.5)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Mover la leyenda de la barra de color
        plt.legend(title=color_label, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_path = OUTPUT_DIR / f"corr_{x_param}_vs_{y_param}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - Gráfico de Correlación {x_param} vs {y_param} guardado: {plot_path.name}")


# -------------------------
# 5) FUNCIÓN PRINCIPAL
# -------------------------

def main():
    """Ejecuta el pipeline de resumen y visualización."""
    
    df_clean = load_fit_results()
    if df_clean is None:
        return
        
    # Paso 2: Estadísticas Descriptivas
    print_summary_statistics(df_clean)
    
    # Paso 3: Distribuciones
    plot_distributions(df_clean)
    
    # Paso 4: Correlaciones
    plot_correlations(df_clean)

    print("\nResumen y visualizaciones completadas. Revise la carpeta 'btfr_results/summary_plots'.")

if __name__ == "__main__":
    main()
