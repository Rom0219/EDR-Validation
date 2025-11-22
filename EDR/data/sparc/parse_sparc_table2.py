#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os

# Ruta correcta del archivo SPARC
BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "SPARC_Lelli2016c.txt.txt")

def parse_table2(file_path):
    """
    Parsea el archivo SPARC_Lelli2016c en formato texto/mrt
    y devuelve un DataFrame usable.
    """
    rows = []
    current = {}

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Object"):
                if current:
                    rows.append(current)
                current = {"Galaxy": line.split()[1]}

            elif "Distance" in line and "Mpc" in line:
                num = re.findall(r"([\d\.]+)", line)
                if num:
                    current["Distance_Mpc"] = float(num[0])

            elif "Inclination" in line:
                nums = re.findall(r"([\d\.]+)", line)
                if nums:
                    current["Incl_deg"] = float(nums[0])

            elif "Ldisk" in line:
                nums = re.findall(r"([\d\.Ee+-]+)", line)
                if nums:
                    current["Ldisk"] = float(nums[0])

            elif "Lbul" in line:
                nums = re.findall(r"([\d\.Ee+-]+)", line)
                if nums:
                    current["Lbul"] = float(nums[0])

            elif "Mgas" in line:
                nums = re.findall(r"([\d\.Ee+-]+)", line)
                if nums:
                    current["Mgas"] = float(nums[0])

    if current:
        rows.append(current)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Usando archivo:", INPUT_FILE)

    df = parse_table2(INPUT_FILE)

    print(df.head())
    df.to_csv(os.path.join(BASE_DIR, "SPARC_Lelli2016_Table2.csv"), index=False)
    print("\nArchivo generado en:")
    print(os.path.join(BASE_DIR, "SPARC_Lelli2016_Table2.csv"))
