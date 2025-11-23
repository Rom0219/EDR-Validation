#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlaciones_sparc_edr_175.py — Análisis de Correlaciones y Distribuciones

Analiza los resultados del ajuste EDR (edr_fit_results.csv) para buscar
correlaciones entre los parámetros de la EDR (A, R0) y las propiedades
físicas de las galaxias SPARC (Luminosidad, Masa de Gas, M/L).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de Matplotlib y Seaborn
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# -------------------------
# 1) CONFIGURACIÓN
# -------------------------
RESULTS_DIR = Path("./btfr_results")
RESULTS_FILENAME = RESULTS_DIR / "edr_fit_results.csv"
OUTPUT_DIR = RESULTS_DIR / "correlations"

# -------------------------
# 2) FUNCIONES DE PLOTEO
# -------------------------

def plot_scatter_with_correlation(df, x_col, y_col, x_label, y_label, save_path):
    """Genera un gráfico de dispersión y calcula la correlación de Pearson."""
    
    # Asegurarse de que no haya NaN en las columnas de interés
    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    
    if len(df_clean) < 2:
        print(f"Advertencia: No hay suficientes datos limpios para plotear {x_col} vs {y_col}.")
        return

    # Correlación de Pearson
    r_pearson, p_pearson = pearsonr(df_clean[x_col], df_clean[y_col])
    
    # Gráfico de dispersión
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.7, edgecolors='w', s=50)

    # Añadir coeficientes al gráfico
    corr_text = (f"Pearson R: {r_pearson:.3f}\n"
                 f"P-value: {p_pearson:.3e}")
    
    # Determinar la posición del texto (superior derecha o inferior izquierda)
    x_range = df_clean[x_col].max() - df_clean[x_col].min()
    y_range = df_clean[y_col].max() - df_clean[y_col].min()
    x_pos = df_clean[x_col].min() + 0.05 * x_range
    y_pos = df_clean[y_col].max() - 0.15 * y_range

    ax.text(x_pos, y_pos, corr_text, 
            transform=ax.transData, 
            fontsize=11, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Correlación: {y_label} vs {x_label}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de correlación guardado: {save_path.name}")

def plot_histograms(df, col, label, save_path):
    """Genera un histograma de la distribución de un parámetro."""
    
    df_clean = df.dropna(subset=[col]).copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histograma de densidad normalizada
    sns.histplot(df_clean[col], kde=True, bins=15, ax=ax, 
                 color='skyblue', edgecolor='black', stat="density")
    
    # Media y Desviación Estándar
    mean_val = df_clean[col].mean()
    std_val = df_clean[col].std()
    
    # Líneas para media y +/- 1 sigma
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='gray', linestyle=':', label=f'1 $\\sigma$: $\\pm${std_val:.2f}')
    ax.axvline(mean_val - std_val, color='gray', linestyle=':')

    ax.set_xlabel(label)
    ax.set_ylabel("Densidad de Galaxias")
    ax.set_title(f"Distribución de {label}", fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Histograma guardado: {save_path.name}")


# -------------------------
# 3) FUNCIÓN PRINCIPAL DE ANÁLISIS
# -------------------------

def main():
    """Carga los resultados y realiza los análisis de correlación."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        df = pd.read_csv(RESULTS_FILENAME)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de resultados en '{RESULTS_FILENAME}'.")
        print("Por favor, ejecute 'run_btfr_validation.py' primero.")
        return

    print(f"Analizando resultados de {len(df)} galaxias...")
    
    # ----------------------------------------------------
    # 3.1) Distribuciones de Parámetros EDR y Chi2
    # ----------------------------------------------------
    
    # Distribución del parámetro de amplitud de la EDR (A)
    plot_histograms(df, 'A_edr', r'Parámetro $A$ de la EDR ($\mathrm{km/s}$)', 
                    OUTPUT_DIR / "hist_A_edr.png")

    # Distribución del parámetro de radio de escala de la EDR (R0)
    plot_histograms(df, 'R0', r'Radio de Escala $R_0$ ($\mathrm{kpc}$)', 
                    OUTPUT_DIR / "hist_R0.png")
    
    # Distribución de Chi Cuadrado Reducido
    plot_histograms(df, 'chi2_red', r'Chi Cuadrado Reducido ($\chi^2_{\mathrm{red}}$)', 
                    OUTPUT_DIR / "hist_chi2_red.png")


    # ----------------------------------------------------
    # 3.2) Correlaciones EDR vs Propiedades de la Galaxia
    # ----------------------------------------------------
    
    # Definición de las correlaciones a investigar (X vs Y)
    correlations_list = [
        # EDR Parámetros vs Luminosidad/Masa (Proxies)
        ('L3.6_9', 'A_edr', r'Luminosidad $L_{3.6}$ ($10^9 L_{\odot}$)', r'Parámetro $A$ ($\mathrm{km/s}$)', "L_vs_A_edr.png"),
        ('MHI_9', 'A_edr', r'Masa de Gas $M_{\mathrm{HI}}$ ($10^9 M_{\odot}$)', r'Parámetro $A$ ($\mathrm{km/s}$)', "Mgas_vs_A_edr.png"),
        
        # EDR Parámetros vs Factores M/L
        ('Yd_fit', 'A_edr', r'Factor M/L Disco $Y_d$', r'Parámetro $A$ ($\mathrm{km/s}$)', "Yd_vs_A_edr.png"),
        ('Yd_fit', 'R0', r'Factor M/L Disco $Y_d$', r'Radio de Escala $R_0$ ($\mathrm{kpc}$)', "Yd_vs_R0.png"),
        
        # Parámetros EDR entre sí
        ('A_edr', 'R0', r'Parámetro $A$ ($\mathrm{km/s}$)', r'Radio de Escala $R_0$ ($\mathrm{kpc}$)', "A_vs_R0.png"),
        
        # Calidad del Ajuste vs Propiedades
        ('L3.6_9', 'chi2_red', r'Luminosidad $L_{3.6}$ ($10^9 L_{\odot}$)', r'Chi Cuadrado Reducido $\chi^2_{\mathrm{red}}$', "L_vs_chi2.png"),
        ('A_edr', 'chi2_red', r'Parámetro $A$ ($\mathrm{km/s}$)', r'Chi Cuadrado Reducido $\chi^2_{\mathrm{red}}$', "A_vs_chi2.png"),
    ]
    
    for x_col, y_col, x_label, y_label, filename in correlations_list:
        plot_scatter_with_correlation(df, x_col, y_col, x_label, y_label, OUTPUT_DIR / filename)

    print("\nAnálisis de correlaciones completado. Resultados guardados en './btfr_results/correlations'")


if __name__ == "__main__":
    main()
