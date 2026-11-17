"""Prediccion de precio de alquiler con los modelos entrenados (Ridge y Elastic Net).

Carga los coeficientes ya ajustados desde `data/final_fit.npz` y la media de
entrenamiento desde `data/X_full.csv` sin necesidad de re-ejecutar el notebook.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TASA_USD_PEN = 3.80
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOP_DISTRICTS = [
    "Miraflores",
    "San Isidro",
    "Santiago de Surco",
    "Barranco",
    "San Borja",
    "San Miguel",
    "La Molina",
    "Surquillo",
    "Chorrillos",
    "Comas",
    "Lince",
]


def load_artifacts():
    """Carga coeficientes guardados y reconstruye la media de estandarizacion."""
    fit = np.load(DATA_DIR / "final_fit.npz")
    feat_names = fit["feat_names"].tolist()
    sigma = fit["sigma"]

    x_full = pd.read_csv(DATA_DIR / "X_full.csv")
    x_train, _ = train_test_split(x_full.values, test_size=0.2, random_state=42)
    mu = x_train.mean(axis=0)
    return fit, feat_names, mu, sigma

#x_raw: los valores reales 
# (por ejemplo area_m2 = 120, bedrooms = 3, area_m2^2 = 14400).
#mu (μ): el promedio 
# de cada columna en los datos de entrenamiento (por ejemplo, en tu dataset de train bedrooms tiene media 2.60
# y area_m2 tiene media 131.78).
#sigmaσ): la desviación 
#estándar de cada columna en train (1.22 para bedrooms, 102.60 para area_m2)
def build_feature_vector(
    feat_names,
    area_m2: float,
    bedrooms: float,
    bathrooms: float,
    district: str = "Miraflores",
    housing_type: str = "Apartamento",
    age_years: float = 9.0,
    maintenance_pen: float = 0.0,
) -> np.ndarray:
    """Construye el vector de 34 features en el mismo orden de X_full.csv."""
    x = pd.Series(0.0, index=feat_names)

    x["bedrooms"] = bedrooms
    x["bathrooms"] = bathrooms
    x["area_m2"] = area_m2
    x["age_years"] = age_years
    x["maintenance_pen"] = maintenance_pen
    x["has_maintenance"] = 1.0 if maintenance_pen > 0 else 0.0

    # Categoria de referencia para tipo: Apartamento
    if housing_type.strip().lower() == "casa":
        x["tipo_Casa"] = 1.0

    # Categoria de referencia para distrito: Barranco; otros distritos -> 'Otro'
    dist_clean = district.strip()
    dist_grp = dist_clean if dist_clean in TOP_DISTRICTS else "Otro"
    dist_col = f"distrito_{dist_grp}"
    if dist_col in x.index:
        x[dist_col] = 1.0
        x[f"area*{dist_col}"] = area_m2

    x["area*bedrooms"] = area_m2 * bedrooms
    x["area*bathrooms"] = area_m2 * bathrooms
    x["bedrooms*bathrooms"] = bedrooms * bathrooms
    x["area_m2^2"] = area_m2 ** 2
    x["age_years^2"] = age_years ** 2

    return x.values


def predecir(
    area_m2: float,
    bedrooms: float,
    bathrooms: float,
    district: str = "Miraflores",
    housing_type: str = "Apartamento",
    age_years: float = 9.0,
    maintenance_pen: float = 0.0,
):
    """Retorna la prediccion de alquiler mensual"""
    fit, feat_names, mu, sigma = load_artifacts()
    x_raw = build_feature_vector(
        feat_names,
        area_m2=area_m2,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        district=district,
        housing_type=housing_type,
        age_years=age_years,
        maintenance_pen=maintenance_pen,
    )
    x_s = (x_raw - mu) / sigma

    price_ridge = float(np.exp(fit["ridge_intercept"] + x_s @ fit["ridge_coef"]))
    price_en = float(np.exp(fit["en_intercept"] + x_s @ fit["en_coef"]))
    return price_ridge, price_en


def parse_args():
    parser = argparse.ArgumentParser(description="Predecir precio de alquiler en Lima.")
    parser.add_argument("--area", type=float, help="Area en m2")
    parser.add_argument("--bedrooms", type=float, help="Numero de dormitorios")
    parser.add_argument("--bathrooms", type=float, help="Numero de banos")
    parser.add_argument("--district", type=str, default=None, help="Distrito (ej. Miraflores)")
    parser.add_argument("--tipo", type=str, default="Apartamento", choices=["Apartamento", "Casa"])
    parser.add_argument("--age", type=float, default=9.0, help="Antiguedad en anos (default: 9)")
    parser.add_argument("--maint", type=float, default=0.0, help="Mantenimiento en S/. (default: 0)")
    return parser.parse_args()


def ask_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Ingresa un numero valido.")


def main():
    args = parse_args()

    if args.area is not None and args.bedrooms is not None and args.bathrooms is not None:
        area = args.area
        bedrooms = args.bedrooms
        bathrooms = args.bathrooms
        district = args.district or "Miraflores"
        housing_type = args.tipo
        age = args.age
        maint = args.maint
    else:
        print("Distritos principales:", ", ".join(TOP_DISTRICTS), "(cualquier otro -> Otro)")
        area = ask_float("Area (m2)")
        bedrooms = ask_float("Dormitorios", 2)
        bathrooms = ask_float("Banos", 2)
        district = input("Distrito [Miraflores]: ").strip() or "Miraflores"
        housing_type = input("Tipo (Apartamento/Casa) [Apartamento]: ").strip() or "Apartamento"
        age = ask_float("Antiguedad en anos", 9.0)
        maint = ask_float("Mantenimiento mensual (S/.)", 0.0)

    price_ridge, price_en = predecir(
        area_m2=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        district=district,
        housing_type=housing_type,
        age_years=age,
        maintenance_pen=maint,
    )

    print("\n--- Estimacion de alquiler mensual ---")
    print(f"Ridge      : S/. {price_ridge:,.0f}  (USD {price_ridge / TASA_USD_PEN:,.0f})")
    print(f"Elastic Net: S/. {price_en:,.0f}  (USD {price_en / TASA_USD_PEN:,.0f})")

#ejemplitoo
#Area (m2): 120
#Dormitorios [2]: 3
#Banos [2]: 2
#Distrito [Miraflores]: Miraflores
#Tipo (Apartamento/Casa) [Apartamento]: Apartamento
#Antiguedad en anos [9.0]: 9
#Mantenimiento mensual (S/.) [0.0]: 120
#

if __name__ == "__main__":
    main()
