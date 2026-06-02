"""
scripts/04_simulate_ca.py
Simulación de expansión urbana con Autómata Celular (2025–2030).

Algoritmo:
  Para cada año t → t+1:
    1. Cargar estado urbano actual y features espaciales
    2. Calcular P_RF  = probabilidad del modelo Random Forest
    3. Calcular P_nbr = densidad de vecindad (regla CA)
    4. Calcular P_total = α*P_RF + β*P_nbr + γ*random
    5. Aplicar restricciones (agua, cenotes, reservas)
    6. Calcular cuántas celdas deben urbanizarse (growth_rate)
    7. Convertir las celdas de mayor probabilidad hasta llegar al cupo
    8. Guardar mapa de probabilidades y mapa binario

Salidas en results/maps/:
  - prediction_{year}.tif      → mapa de probabilidad (float32, 0-1)
  - urban_extent_{year}.tif    → mapa binario urbano/no-urbano (uint8)
  - urban_extent_{year}.shp    → shapefile del área urbana proyectada
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import uniform_filter
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CA_CONFIG, PREDICT_YEARS, BASE_YEAR, PATHS, PIXEL_RESOLUTION,
    STUDY_AREA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "dist_urban_edge", "dist_center", "dist_road",
    "ndvi_mean", "neighbor_3x3", "neighbor_5x5", "neighbor_9x9"
]

rng = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def load_raster(path: str) -> tuple[np.ndarray, dict]:
    """Carga un GeoTIFF y retorna (array, profile)."""
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
    return data, profile


def save_raster(array: np.ndarray, path: str, profile: dict, dtype=None):
    """Guarda un array como GeoTIFF."""
    p = profile.copy()
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    p.update(count=array.shape[0], compress="lzw")
    if dtype:
        p.update(dtype=dtype)
        array = array.astype(dtype)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array)


def raster_to_shapefile(urban_array: np.ndarray, profile: dict, out_path: str):
    """Vectoriza el mapa binario de área urbana a shapefile."""
    results = []
    data = urban_array.astype(np.uint8)
    for geom, val in shapes(data, mask=(data == 1), transform=profile["transform"]):
        if val == 1:
            results.append({"geometry": shape(geom), "urban": 1})

    if results:
        gdf = gpd.GeoDataFrame(results, crs=profile["crs"])
        # Disolver todos los polígonos en uno
        gdf_dissolved = gdf.dissolve()
        gdf_dissolved.to_file(out_path)
        log.info(f"    ✓ Shapefile: {out_path}")
    else:
        log.warning(f"    No se generaron polígonos urbanos para {out_path}")


# ─────────────────────────────────────────────────────────────
# ACTUALIZACIÓN DINÁMICA DE FEATURES
# ─────────────────────────────────────────────────────────────

def update_features_for_year(features_base: np.ndarray, urban: np.ndarray,
                              profile: dict) -> np.ndarray:
    """
    Actualiza las features que dependen del estado urbano actual:
      - dist_urban_edge (cambia con cada expansión)
      - neighbor_3x3, neighbor_5x5, neighbor_9x9
    Las otras features (dist_center, dist_road, ndvi_mean) son estáticas.
    """
    from scipy.ndimage import distance_transform_edt

    features = features_base.copy()
    non_urban = (urban == 0).astype(np.float32)

    # Actualizar distancia al borde urbano (banda 0)
    dist_edge_pixels = distance_transform_edt(non_urban)
    dist_edge_m = dist_edge_pixels * PIXEL_RESOLUTION
    max_dist = np.percentile(dist_edge_m[dist_edge_m > 0], 99) if dist_edge_m.max() > 0 else 1
    features[0] = np.clip(dist_edge_m / max_dist, 0, 1).astype(np.float32)

    # Actualizar densidades de vecindad (bandas 4, 5, 6)
    features[4] = uniform_filter(urban.astype(np.float32), size=3)   # 3x3
    features[5] = uniform_filter(urban.astype(np.float32), size=5)   # 5x5
    features[6] = uniform_filter(urban.astype(np.float32), size=9)   # 9x9

    return features


# ─────────────────────────────────────────────────────────────
# PROBABILIDADES CA
# ─────────────────────────────────────────────────────────────

def compute_rf_probability(model, features: np.ndarray, urban: np.ndarray) -> np.ndarray:
    """
    Aplica el modelo RF a todos los píxeles no-urbanos para obtener
    la probabilidad de transición.
    """
    rows, cols = urban.shape
    non_urban_mask = urban == 0

    # Vectorizar features para los píxeles no-urbanos
    X = features.reshape(features.shape[0], -1).T[non_urban_mask.ravel()]

    # Eliminar filas con NaN
    nan_mask = np.any(np.isnan(X), axis=1)
    X[nan_mask] = 0

    prob_urban = model.predict_proba(X)[:, 1]

    # Reconstruir mapa completo
    prob_map = np.zeros(rows * cols, dtype=np.float32)
    prob_map[non_urban_mask.ravel()] = prob_urban
    return prob_map.reshape(rows, cols)


def compute_neighborhood_probability(urban: np.ndarray,
                                     radius: int = None) -> np.ndarray:
    """
    Regla de vecindad CA: fracción de píxeles urbanos en la ventana.
    """
    radius = radius or CA_CONFIG["neighborhood_radius"]
    window = 2 * radius + 1
    return uniform_filter(urban.astype(np.float32), size=window)


def compute_total_probability(p_rf: np.ndarray, p_nbr: np.ndarray) -> np.ndarray:
    """
    Combina probabilidades:
    P_total = α*P_RF + β*P_nbr + γ*U(0,1)
    """
    alpha = CA_CONFIG["alpha"]
    beta  = CA_CONFIG["beta"]
    gamma = CA_CONFIG["gamma"]

    p_stochastic = rng.random(p_rf.shape, dtype=np.float32)
    p_total = alpha * p_rf + beta * p_nbr + gamma * p_stochastic

    # Normalizar a [0, 1]
    p_total = np.clip(p_total, 0, 1)
    return p_total.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# RESTRICCIONES DE CONVERSIÓN
# ─────────────────────────────────────────────────────────────

def build_exclusion_mask(shape: tuple, profile: dict) -> np.ndarray:
    """
    Construye máscara de píxeles que NO pueden urbanizarse
    (cuerpos de agua, cenotes, áreas protegidas).
    Por defecto: todos pueden urbanizarse (máscara de ceros).
    Si existen capas de restricción en data/raw/, se aplican.
    """
    mask = np.zeros(shape, dtype=bool)

    # Intentar cargar shapefiles de restricción si existen
    restriction_files = {
        "agua": os.path.join(PATHS["raw"], "cuerpos_agua.shp"),
        "reserva": os.path.join(PATHS["raw"], "areas_naturales_protegidas.shp"),
    }

    try:
        from rasterio.features import rasterize
        from shapely.geometry import mapping

        for name, path in restriction_files.items():
            if os.path.exists(path):
                gdf = gpd.read_file(path).to_crs(profile["crs"])
                burned = rasterize(
                    [(mapping(geom), 1) for geom in gdf.geometry if not geom.is_empty],
                    out_shape=shape,
                    transform=profile["transform"],
                    fill=0,
                    dtype=np.uint8,
                )
                mask = mask | burned.astype(bool)
                log.info(f"    Restricción '{name}': {burned.sum():,} píxeles excluidos")
    except Exception as e:
        log.debug(f"    No se pudieron cargar restricciones: {e}")

    return mask


# ─────────────────────────────────────────────────────────────
# SIMULACIÓN ANUAL
# ─────────────────────────────────────────────────────────────

def simulate_year(model, urban: np.ndarray, features: np.ndarray,
                  profile: dict, exclusion_mask: np.ndarray,
                  year: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula un año de expansión urbana.
    Retorna: (urban_new, prob_map)
    """
    log.info(f"  [{year}] Simulando...")

    # Actualizar features dinámicas
    features_updated = update_features_for_year(features, urban, profile)

    # Probabilidades RF para celdas no-urbanas
    p_rf  = compute_rf_probability(model, features_updated, urban)

    # Probabilidad de vecindad CA
    p_nbr = compute_neighborhood_probability(urban)

    # Probabilidad total
    p_total = compute_total_probability(p_rf, p_nbr)

    # Aplicar exclusiones (celdas que no pueden urbanizarse)
    p_total[exclusion_mask] = 0
    p_total[urban == 1] = 0  # ya es urbano

    # Calcular cuántas celdas deben convertirse este año
    n_urban_current = urban.sum()
    n_to_convert = int(n_urban_current * CA_CONFIG["annual_growth_rate"])
    n_non_urban = (urban == 0).sum() - exclusion_mask.sum()
    n_to_convert = min(n_to_convert, max(0, n_non_urban))

    log.info(f"    Urbano actual: {n_urban_current:,} px ({n_urban_current * PIXEL_RESOLUTION**2 / 1e6:.1f} km²)")
    log.info(f"    Celdas a convertir: {n_to_convert:,} px ({n_to_convert * PIXEL_RESOLUTION**2 / 1e6:.2f} km²)")

    # Seleccionar las celdas de mayor probabilidad
    urban_new = urban.copy()
    if n_to_convert > 0:
        flat_probs = p_total.ravel()
        threshold_idx = np.argsort(flat_probs)[-n_to_convert:]
        urban_flat = urban_new.ravel()
        urban_flat[threshold_idx] = 1
        urban_new = urban_flat.reshape(urban.shape)

    n_new = urban_new.sum() - urban.sum()
    log.info(f"    Nuevas celdas urbanas: {n_new:,}")

    return urban_new, p_total


# ─────────────────────────────────────────────────────────────
# ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────

def compute_area_statistics(urban_series: dict) -> pd.DataFrame:
    """Calcula estadísticas de área urbana por año."""
    stats = []
    years = sorted(urban_series.keys())

    for i, year in enumerate(years):
        urban = urban_series[year]
        area_km2 = urban.sum() * PIXEL_RESOLUTION**2 / 1e6
        area_ha  = urban.sum() * PIXEL_RESOLUTION**2 / 1e4

        if i > 0:
            prev_year = years[i-1]
            prev_area = urban_series[prev_year].sum() * PIXEL_RESOLUTION**2 / 1e6
            growth_km2 = area_km2 - prev_area
            growth_pct  = (area_km2 - prev_area) / prev_area * 100 if prev_area > 0 else 0
        else:
            growth_km2 = 0
            growth_pct  = 0

        stats.append({
            "year": year,
            "area_km2": round(area_km2, 2),
            "area_ha": round(area_ha, 1),
            "growth_km2": round(growth_km2, 2),
            "growth_pct": round(growth_pct, 2),
        })

    return pd.DataFrame(stats)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PASO 4: Simulación de Autómata Celular (2025–2030)")
    log.info("=" * 60)

    # Cargar modelo entrenado
    if not os.path.exists(PATHS["rf_model"]):
        log.error(f"Modelo no encontrado: {PATHS['rf_model']}")
        log.error("Ejecuta primero: python scripts/03_train_model.py")
        sys.exit(1)

    log.info(f"Cargando modelo RF: {PATHS['rf_model']}")
    model = joblib.load(PATHS["rf_model"])

    # Cargar estado urbano base (año más reciente disponible)
    base_lulc_path = PATHS["lulc_base"].format(year=BASE_YEAR)
    if not os.path.exists(base_lulc_path):
        log.error(f"LULC base no encontrado: {base_lulc_path}")
        sys.exit(1)

    urban_base, profile = load_raster(base_lulc_path)
    urban = urban_base[0].astype(np.uint8)
    log.info(f"Estado base {BASE_YEAR}: {urban.sum():,} píxeles urbanos "
             f"({urban.sum() * PIXEL_RESOLUTION**2 / 1e6:.1f} km²)")

    # Cargar features base
    feat_base_path = PATHS["features"].format(year=BASE_YEAR)
    if not os.path.exists(feat_base_path):
        log.error(f"Features base no encontradas: {feat_base_path}")
        sys.exit(1)

    features_base, _ = load_raster(feat_base_path)
    log.info(f"Features base cargadas: {features_base.shape[0]} bandas")

    # Construir máscara de exclusión
    log.info("Construyendo máscara de exclusión...")
    exclusion_mask = build_exclusion_mask(urban.shape, profile["meta"] if "meta" in profile else profile)

    # Simulación año por año
    urban_series = {BASE_YEAR: urban.copy()}
    current_urban = urban.copy()

    for year in PREDICT_YEARS:
        current_urban, prob_map = simulate_year(
            model, current_urban, features_base, profile, exclusion_mask, year
        )
        urban_series[year] = current_urban.copy()

        # Guardar mapa de probabilidades
        prob_path = PATHS["prediction"].format(year=year)
        save_raster(prob_map, prob_path, profile, dtype=rasterio.float32)
        log.info(f"  → Probabilidades guardadas: {prob_path}")

        # Guardar mapa binario
        extent_path = os.path.join(PATHS["results_maps"], f"urban_extent_{year}.tif")
        save_raster(current_urban, extent_path, profile, dtype=rasterio.uint8)

        # Vectorizar a shapefile
        shp_path = PATHS["urban_extent"].format(year=year)
        raster_to_shapefile(current_urban, profile, shp_path)

    # Estadísticas finales
    stats_df = compute_area_statistics(urban_series)
    stats_path = os.path.join(PATHS["results_reports"], "area_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    log.info("\n" + "=" * 60)
    log.info("ESTADÍSTICAS DE EXPANSIÓN URBANA:")
    log.info("=" * 60)
    log.info(stats_df.to_string(index=False))
    log.info(f"\n✓ Estadísticas guardadas: {stats_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
