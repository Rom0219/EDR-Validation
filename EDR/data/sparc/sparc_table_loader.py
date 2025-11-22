import pandas as pd
import re

def load_sparc_table(file_path):
    """
    Lee el archivo SPARC_Lelli2016c.txt.txt tal cual lo subiste,
    detecta automáticamente las columnas, limpia comentarios
    y devuelve un DataFrame listo para usar en los cálculos de Mbar.
    """

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            # Saltar comentarios y líneas vacías
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue

            # Separación por espacios múltiples
            parts = re.split(r"\s+", line.strip())
            rows.append(parts)

    # Detectar automáticamente cuántas columnas tiene
    max_cols = max(len(r) for r in rows)

    # Normalizar filas (rellenar)
    clean_rows = [r + [""]*(max_cols - len(r)) for r in rows]

    # Crear dataframe genérico
    df = pd.DataFrame(clean_rows)

    # La primera fila tiene los nombres reales
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # Convertir numéricas donde se pueda
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def extract_global_barionics(df):
    """
    Extrae (L_disk, L_bul, M_gas) usando los nombres estándar de SPARC.
    No todas las galaxias tienen bulbo → se maneja automáticamente.
    """

    results = {}

    # SPARC typical columns
    possible_Ldisk = ["Ldisk", "L_disk", "LDisk", "DiskLum"]
    possible_Lbul = ["Lbul", "L_bul", "LBulge", "BulLum"]
    possible_Mgas = ["Mgas", "M_gas", "GasMass", "MHI"]

    def find_column(possibles):
        for p in possibles:
            if p in df.columns:
                return p
        return None

    col_Ld = find_column(possible_Ldisk)
    col_Lb = find_column(possible_Lbul)
    col_Mg = find_column(possible_Mgas)

    results["L_disk"] = df[col_Ld].iloc[0] if col_Ld else 0
    results["L_bul"] = df[col_Lb].iloc[0] if col_Lb else 0
    results["M_gas"] = df[col_Mg].iloc[0] if col_Mg else 0

    return results


if __name__ == "__main__":
    file_path = "EDR/data/sparc/SPARC_Lelli2016c.txt.txt"

    print("\n=== CARGANDO TABLA SPARC ===\n")

    df = load_sparc_table(file_path)
    print("Columnas detectadas:\n", df.columns)

    bar = extract_global_barionics(df)
    print("\nValores extraídos:")
    print(bar)
