#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
residuals_analysis_175.py — Análisis avanzado de residuales normalizados.

Este script carga los resultados del ajuste EDR de 'run_btfr_validation_175.py'
y realiza:
1. Pruebas de normalidad (Shapiro-Wilk) sobre los residuales.
2. Análisis de correlación entre los residuales y las propiedades físicas.
3. Gráficos Q-Q plot para evaluar la normalidad.
4. Histograma de los chi2 reducidos (chi2_red).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

# Configuración de Matplotlib y Seaborn
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# -------------------------
# 1) CONFIGURACIÓN Y CARGA DE DATOS
# -------------------------
RESULTS_DIR = Path("./btfr_results")
RESULTS_FILENAME = RESULTS_DIR / "edr_fit_results.csv"
OUTPUT_DIR = RESULTS_DIR / "residuals_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_fit_results():
    """Carga y limpia los resultados del ajuste EDR."""
    try:
        df = pd.read_csv(RESULTS_FILENAME)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de resultados en '{RESULTS_FILENAME}'.")
        print("Por favor, ejecute 'run_btfr_validation_175.py' primero.")
        return None

    # Columnas clave para el análisis de residuales
    features = ['log_Mbar', 'log_Vflat', 'L3.6_9']
    residuals_cols = ['chi2_red', 'sigma_extra']
    
    df_clean = df.dropna(subset=features + residuals_cols).reset_index(drop=True)
    
    print(f"Datos limpios para Análisis de Residuales: {len(df_clean)} galaxias.")
    
    return df_clean

# -------------------------
# 2) PRUEBAS ESTADÍSTICAS
# -------------------------

def statistical_tests(df):
    """Realiza pruebas de normalidad y correlación."""
    
    print("\n--- Pruebas Estadísticas ---")
    
    # a) Test de Normalidad de Shapiro-Wilk para chi2_red
    print("\n[A] Test de Normalidad de Shapiro-Wilk (para chi2_red):")
    # El chi2_red debería seguir una distribución chi-cuadrado si el modelo es correcto.
    # Aquí evaluamos su distribución empírica.
    stat_chi2, p_chi2 = stats.shapiro(df['chi2_red'])
    print(f" - Estadístico Shapiro-Wilk: {stat_chi2:.4f}")
    print(f" - Valor p: {p_chi2:.4e}")
    if p_chi2 < 0.05:
        print(" -> Se rechaza la hipótesis nula: La distribución NO es normal.")
    else:
        print(" -> No se rechaza la hipótesis nula: La distribución parece normal (o la muestra es pequeña).")
        
    # b) Test de Normalidad de Shapiro-Wilk para sigma_extra
    print("\n[B] Test de Normalidad de Shapiro-Wilk (para sigma_extra):")
    stat_sigma, p_sigma = stats.shapiro(df['sigma_extra'])
    print(f" - Estadístico Shapiro-Wilk: {stat_sigma:.4f}")
    print(f" - Valor p: {p_sigma:.4e}")
    if p_sigma < 0.05:
        print(" -> Se rechaza la hipótesis nula: La distribución NO es normal.")
    else:
        print(" -> No se rechaza la hipótesis nula: La distribución parece normal.")
        
    # c) Correlación entre residuales y propiedades físicas (Pearson)
    print("\n[C] Coeficientes de Correlación de Pearson:")
    
    properties = ['log_Mbar', 'L3.6_9', 'R0']
    
    for prop in properties:
        # Correlación chi2_red vs propiedad
        corr_chi2, pval_chi2 = stats.pearsonr(df['chi2_red'], df[prop])
        print(f" - chi2_red vs {prop}: r={corr_chi2:.3f}, p={pval_chi2:.3e}")
        
        # Correlación sigma_extra vs propiedad
        corr_sigma, pval_sigma = stats.pearsonr(df['sigma_extra'], df[prop])
        print(f" - sigma_extra vs {prop}: r={corr_sigma:.3f}, p={pval_sigma:.3e}")
    
    print("Nota: Un valor p bajo (<0.05) y un r alto (cercano a 1 o -1) sugieren un sesgo sistemático.")

# -------------------------
# 3) GRÁFICOS DE DIAGNÓSTICO
# -------------------------

def plot_diagnostics(df):
    """Genera gráficos de diagnóstico para los residuales."""
    print("\n--- Generando Gráficos de Diagnóstico ---")
    
    # a) Histograma de Chi2 Reducido
    plt.figure(figsize=(10, 6))
    sns.histplot(df['chi2_red'], bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(1.0, color='r', linestyle='--', label='$\chi^2_{\\mathrm{red}} = 1.0$ (Buen Ajuste)')
    plt.title('Distribución del $\chi^2_{\\mathrm{red}}$ (EDR)')
    plt.xlabel('$\chi^2_{\\mathrm{red}}$')
    plt.ylabel('Frecuencia')
    plt.legend()
    chi2_hist_path = OUTPUT_DIR / "chi2_red_histogram.png"
    plt.savefig(chi2_hist_path)
    plt.close()
    print(f"Gráfico de Histograma $\chi^2_{\\mathrm{red}}$ guardado: {chi2_hist_path.name}")
    
    # b) Histograma de Sigma Extra (Residuales Adicionales)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sigma_extra'], bins=20, kde=True, color='lightcoral', edgecolor='black')
    plt.title('Distribución de $\sigma_{\\mathrm{extra}}$ (Dispersión Intrínseca)')
    plt.xlabel('$\sigma_{\\mathrm{extra}}$ (km/s)')
    plt.ylabel('Frecuencia')
    sigma_hist_path = OUTPUT_DIR / "sigma_extra_histogram.png"
    plt.savefig(sigma_hist_path)
    plt.close()
    print(f"Gráfico de Histograma $\sigma_{\\mathrm{extra}}$ guardado: {sigma_hist_path.name}")

    # c) Gráfico Q-Q Plot para Sigma Extra (Evaluación visual de Normalidad)
    # Si los puntos caen en la línea recta, la distribución es normal.
    fig, ax = plt.subplots(figsize=(8, 8))
    qqplot(df['sigma_extra'], line='s', ax=ax)
    ax.set_title('Q-Q Plot de $\sigma_{\\mathrm{extra}}$ vs. Distribución Normal')
    ax.set_xlabel('Cuantiles Teóricos')
    ax.set_ylabel('Cuantiles de $\sigma_{\\mathrm{extra}}$')
    qq_plot_path = OUTPUT_DIR / "sigma_extra_qqplot.png"
    plt.savefig(qq_plot_path)
    plt.close()
    print(f"Gráfico Q-Q Plot guardado: {qq_plot_path.name}")
    
    # d) Diagramas de dispersión de residuales vs. propiedades físicas
    
    properties_to_plot = {
        'log_Mbar': 'Log Masa Bariónica Total ($\log_{10} M_{\\mathrm{bar}} / M_{\\odot}$)',
        'L3.6_9': 'Luminosidad 3.6 $\mu m$ ($\times 10^9 L_{\\odot}$)'
    }
    
    for prop, label in properties_to_plot.items():
        # Chi2_red vs Propiedad
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[prop], y=df['chi2_red'], alpha=0.7, color='navy')
        plt.title(f'$\chi^2_{\\mathrm{red}}$ vs {label}')
        plt.xlabel(label)
        plt.ylabel('$\chi^2_{\\mathrm{red}}$')
        plot_path = OUTPUT_DIR / f"chi2_vs_{prop}.png"
        plt.savefig(plot_path)
        plt.close()

        # Sigma_extra vs Propiedad
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[prop], y=df['sigma_extra'], alpha=0.7, color='green')
        plt.title(f'$\sigma_{\\mathrm{extra}}$ vs {label}')
        plt.xlabel(label)
        plt.ylabel('$\sigma_{\\mathrm{extra}}$ (km/s)')
        plot_path = OUTPUT_DIR / f"sigma_vs_{prop}.png"
        plt.savefig(plot_path)
        plt.close()
        
    print("Gráficos de dispersión de residuales guardados.")

# -------------------------
# 4) FUNCIÓN PRINCIPAL
# -------------------------

def main():
    """Ejecuta el análisis completo de residuales."""
    
    df_clean = load_fit_results()
    if df_clean is None:
        return
        
    # Paso 2: Pruebas Estadísticas
    statistical_tests(df_clean)
    
    # Paso 3: Gráficos de Diagnóstico
    plot_diagnostics(df_clean)

    print("\nAnálisis de Residuales completado.")

if __name__ == "__main__":
    main()
