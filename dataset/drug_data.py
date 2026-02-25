from pathlib import Path

import pandas as pd

URL = r"C:\Users\marga\OneDrive\Escritorio\Mis Documentos\Development\DATA_SCIENCE\Flask-Render\ML app\dataset\drug dataset\drug.csv"
TARGET_PATH = Path(__file__).resolve().parent / "drug.csv"

df = pd.read_csv(URL)
df.to_csv(TARGET_PATH, index=False)

print(f"CSV cargado desde: {URL}")
print(f"CSV guardado en: {TARGET_PATH}")


