#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edr_pca_clustering_175.py — Análisis de Componentes Principales (PCA)
y Clustering K-Means sobre los parámetros de la EDR.

Busca grupos intrínsecos de galaxias basados en sus parámetros EDR (A, R0, Yd)
y propiedades físicas (Luminosidad, Masa de Gas).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configuración de Matplotlib y Seaborn
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# -------------------------
# 1) CONFIGURACIÓN Y CARGA DE DATOS
# -------------------------
RESULTS_DIR = Path("./btfr_results")
RESULTS_FILENAME = RESULTS_DIR / "edr_fit_results.csv"
OUTPUT_DIR = RESULTS_DIR / "clustering"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data_for_pca():
    """Carga, limpia y selecciona las columnas para el análisis PCA."""
    try:
        df = pd.read_csv(RESULTS_FILENAME)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de resultados en '{RESULTS_FILENAME}'.")
        print("Por favor, ejecute 'run_btfr_validation_175.py' primero.")
        return None

    # Columnas seleccionadas para el análisis: Parámetros EDR y propiedades clave
    # Usamos los logaritmos de propiedades físicas para normalizar las escalas.
    # log_Mbar y log_Vflat son proxies de estas, pero usamos las originales
    # para evitar correlaciones implícitas ya incluidas en la BTFR.
    
    # 1. Parámetros EDR y M/L
    # 2. Propiedades Físicas (en escala logarítmica)
    df['log_L3.6'] = np.log10(df['L3.6_9'])
    df['log_MHI'] = np.log10(df['MHI_9'])
    
    # Lista final de características a usar
    features = ['A_edr', 'R0', 'Yd_fit', 'log_L3.6', 'log_MHI']

    # Filtrar NaN en las columnas de interés
    df_clean = df.dropna(subset=features).reset_index(drop=True)
    
    X = df_clean[features].values
    
    print(f"Datos limpios para PCA/Clustering: {len(df_clean)} galaxias.")
    print(f"Características utilizadas: {features}")
    
    return df_clean, X, features

# -------------------------
# 2) ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# -------------------------

def perform_pca(X, features):
    """Realiza el PCA, calcula la varianza explicada y plotea."""
    
    # 1. Estandarización de los datos (media=0, desviación estándar=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Aplicar PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # 3. Varianza Explicada Acumulada
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n--- Resultados PCA ---")
    for i, var in enumerate(explained_variance_ratio):
        print(f"Componente Principal {i+1}: Varianza Explicada = {var:.3f}")

    # Plotear Varianza Explicada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='--', color='b')
    plt.title('Varianza Explicada Acumulada por Componente Principal')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.axhline(0.90, color='r', linestyle=':', label='90% de Varianza')
    plt.legend()
    plt.grid(True)
    pca_var_path = OUTPUT_DIR / "pca_explained_variance.png"
    plt.savefig(pca_var_path)
    plt.close()
    print(f"Gráfico de Varianza PCA guardado: {pca_var_path.name}")
    
    # Reducción a 2 Componentes para visualización
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    return X_pca, explained_variance_ratio, scaler

# -------------------------
# 3) CLUSTERING K-MEANS
# -------------------------

def determine_optimal_k(X_pca):
    """Usa el método del Codo y el Coeficiente de Silueta para encontrar K óptimo."""
    
    max_k = 10
    inertias = []
    silhouette_scores = []
    
    print("\n--- Determinando K óptimo (1 a 9 clusters) ---")

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
        
        # Calcular Silueta solo para k > 1
        if k > 1:
            score = silhouette_score(X_pca, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"K={k}: Silueta Score = {score:.3f}")

    # Plot del Codo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), inertias, marker='o', linestyle='-')
    plt.title('Método del Codo para K-Means')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    elbow_path = OUTPUT_DIR / "kmeans_elbow_method.png"
    plt.savefig(elbow_path)
    plt.close()
    print(f"Gráfico del Codo guardado: {elbow_path.name}")
    
    # Plot del Coeficiente de Silueta
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k), silhouette_scores, marker='o', linestyle='-')
    plt.title('Coeficiente de Silueta por Número de Clusters')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Coeficiente de Silueta')
    silhouette_path = OUTPUT_DIR / "kmeans_silhouette_score.png"
    plt.savefig(silhouette_path)
    plt.close()
    print(f"Gráfico de Silueta guardado: {silhouette_path.name}")

    # Retorna el K con la Silueta máxima (a partir de K=2)
    if silhouette_scores:
        optimal_k = np.argmax(silhouette_scores) + 2 
        print(f"K Óptimo Sugerido por Silueta: {optimal_k}")
        return optimal_k
    else:
        return 3 # Valor por defecto si falla la Silueta

def perform_clustering(X_pca, optimal_k, df_clean):
    """Realiza el K-Means con el K óptimo y plotea los resultados."""
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    df_clean['Cluster'] = clusters
    
    # Plotear los clusters en el espacio PCA 2D
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_clean['Cluster'], 
                    palette=sns.color_palette("viridis", optimal_k), 
                    legend='full', s=70, alpha=0.8, edgecolors='w')
    
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='X', s=200, color='red', label='Centros')
    
    plt.title(f'Clustering K-Means de Galaxias (K={optimal_k}) en el Espacio PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    
    pca_cluster_path = OUTPUT_DIR / f"pca_clusters_k{optimal_k}.png"
    plt.savefig(pca_cluster_path)
    plt.close()
    print(f"Gráfico de Clusters guardado: {pca_cluster_path.name}")
    
    return df_clean

# -------------------------
# 4) INTERPRETACIÓN DE CLUSTERS
# -------------------------

def interpret_clusters(df_clustered, features):
    """Muestra las estadísticas de las características por cluster."""
    
    print("\n--- Características Medias por Cluster ---")
    
    # Calcular las medias de las características originales para cada cluster
    cluster_summary = df_clustered.groupby('Cluster')[features].agg(['mean', 'std'])
    print(cluster_summary)
    
    # Opcional: Graficar Boxplots de las características por cluster
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_clustered)
        plt.title(f'Distribución de {feature} por Cluster')
        plt.tight_layout()
        boxplot_path = OUTPUT_DIR / f"boxplot_{feature}_by_cluster.png"
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Boxplot guardado: {boxplot_path.name}")
        
    # Guardar los resultados con la asignación de cluster
    clustered_csv_path = OUTPUT_DIR / "edr_clustered_results.csv"
    df_clustered[['ID'] + features + ['Cluster']].to_csv(clustered_csv_path, index=False)
    print(f"\nResultados con Cluster ID guardados en: {clustered_csv_path.name}")


# -------------------------
# 5) FUNCIÓN PRINCIPAL
# -------------------------

def main():
    """Ejecuta el pipeline PCA y K-Means."""
    
    df_clean, X, features = load_data_for_pca()
    if df_clean is None:
        return
        
    X_pca, explained_variance_ratio, scaler = perform_pca(X, features)
    
    # Paso 3: Clustering K-Means
    optimal_k = determine_optimal_k(X_pca)
    
    # Si determinamos K=1, no tiene sentido hacer clustering, usamos un valor 
    # razonable o el óptimo si es > 1.
    if optimal_k < 2:
        print("El análisis de Silueta sugirió K=1, usando K=3 por defecto para mostrar el clustering.")
        optimal_k = 3

    df_clustered = perform_clustering(X_pca, optimal_k, df_clean)
    
    # Paso 4: Interpretación
    interpret_clusters(df_clustered, features)

    print("\nAnálisis PCA y Clustering completado.")

if __name__ == "__main__":
    main()
