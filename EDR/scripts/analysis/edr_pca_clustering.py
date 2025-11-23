#!/usr/bin/env python3
# edr_pca_clustering.py
# PCA + KMeans clustering for SPARC EDR results
# Usage: python edr_pca_clustering.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- CONFIG ----------
CSV_PATH = "EDR/data/sparc/results/sparc_results.csv"  # ajusta si necesario
OUTDIR = "edr_pca_output"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- LOAD ----------
if not os.path.isfile(CSV_PATH):
    print("CSV not found:", CSV_PATH)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH)
print("Columns:", df.columns.tolist())
print(df.head())

# ---------- SELECT FEATURES ----------
# Features we will use (must exist in your CSV)
features = ["A","R0","Yd","Yb","chi2_red","sigma_extra"]
available = [f for f in features if f in df.columns]
if len(available) < 3:
    raise SystemExit("Need at least 3 features from: " + ", ".join(features))
print("Using features:", available)

data = df[available].copy().apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
print("Rows used for analysis:", len(data))

# ---------- STANDARDIZE ----------
scaler = StandardScaler()
X = scaler.fit_transform(data.values)

# ---------- PCA ----------
n_comp = min(5, X.shape[1])
pca = PCA(n_components=n_comp, random_state=0)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_
expl_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained))],
    "explained_variance_ratio": explained
})
expl_df.to_csv(os.path.join(OUTDIR, "pca_explained_variance.csv"), index=False)
print("Saved PCA explained variance ->", os.path.join(OUTDIR, "pca_explained_variance.csv"))

# Scree plot
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
plt.xlabel("PC number")
plt.ylabel("Explained variance ratio")
plt.title("PCA Scree Plot")
plt.grid(True)
scree_path = os.path.join(OUTDIR, "pca_scree.png")
plt.savefig(scree_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved scree plot ->", scree_path)

# ---------- PREPARE TABLE ----------
pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_cols)
pca_df = pd.concat([pca_df, data.reset_index(drop=True)], axis=1)
# add galaxy names if exist
if "Galaxy" in df.columns:
    # align by index — if your original df had same row order for used rows, fine;
    # if not, consider creating data with Galaxy included earlier.
    galnames = df["Galaxy"].iloc[data.index].values if len(df)>=len(data) else ["G"+str(i) for i in range(len(data))]
    pca_df.insert(0, "Galaxy", galnames)

pca_df.to_csv(os.path.join(OUTDIR, "pca_components_plus_features.csv"), index=False)
print("Saved PCA components + features ->", os.path.join(OUTDIR, "pca_components_plus_features.csv"))

# ---------- KMEANS CLUSTERING  (k=2..4) ----------
cluster_summary = []
best = None
for k in [2,3,4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca[:,:2])  # cluster on first two PCs
    sil = silhouette_score(X_pca[:,:2], labels) if len(np.unique(labels))>1 else np.nan
    cluster_summary.append({"k":k, "silhouette": sil})
    # store best by silhouette
    if best is None or (not np.isnan(sil) and sil > best["sil"]):
        best = {"k":k, "sil": sil, "labels": labels, "centers": km.cluster_centers_}

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv(os.path.join(OUTDIR,"kmeans_summary.csv"), index=False)
print("Saved clustering summary ->", os.path.join(OUTDIR,"kmeans_summary.csv"))

# ---------- PLOT PC1 vs PC2 with best clusters ----------
best_k = best["k"]
best_labels = best["labels"]
best_centers = best["centers"]

plt.figure(figsize=(6,5))
for lab in np.unique(best_labels):
    mask = best_labels == lab
    plt.scatter(X_pca[mask,0], X_pca[mask,1], label=f"cluster {lab}")
# centroids
plt.scatter(best_centers[:,0], best_centers[:,1], marker='x', s=120)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA PC1 vs PC2 (k={best_k}, silhouette={best['sil']:.3f})")
plt.legend()
plt.grid(True)
pc_scatter_path = os.path.join(OUTDIR, "pca_pc1_pc2_clusters.png")
plt.savefig(pc_scatter_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved PC1 vs PC2 clusters ->", pc_scatter_path)

# ---------- Add cluster labels to pca_df and save ----------
pca_df["cluster"] = best_labels
pca_df.to_csv(os.path.join(OUTDIR, "pca_components_clusters.csv"), index=False)
print("Saved pca_components_clusters.csv")

# ---------- Save cluster membership summary ----------
cluster_members = pca_df[["Galaxy","cluster"] + pca_cols + available]
cluster_members.to_csv(os.path.join(OUTDIR, "cluster_members.csv"), index=False)
print("Saved cluster_members.csv")

print("ALL OUTPUTS in folder:", OUTDIR)
