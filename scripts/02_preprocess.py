"""
scripts/02_preprocess.py
Preprocesamiento de imágenes y extracción de variables espaciales (features).

Pasos:
  1. Clasificación LULC (urbano/no-urbano) para cada año histórico
  2. Cálculo de variables espaciales (distancias, índices, vecindad)
  3. Construcción del dataset de entrenamiento (X, y)

Salidas en data/processed/:
  - lulc_{year}.tif           → clasificación binaria (0=no-urbano, 1=urbano)
  - features_{year}.tif       → stack de todas las variables
  - training_dataset.parquet  → tabla X/y para Random Forest
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes
from scipy.ndimage import (
    binary_opening, label, uniform_filter,
    distance_transform_edt, generic_filter
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    STUDY_AREA, LULC_CONFIG, FEATURES_CONFIG, RF_CONFIG,
    TRAIN_YEARS, PATHS, PIXEL_RESOLUTION
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. CLASIFICACIÓN LULC
# ─────────────────────────────────────────────────────────────

def classify_lulc(landsat_path: str, year: int) -> np.ndarray:
    """
    Clasifica píxeles en urbano (1) / no-urbano (0) usando NDVI y NDBI.
    Aplica apertura morfológica para eliminar ruido.

    Estrategia:
    - Un píxel es 'urbano' si NDVI < umbral Y NDBI > umbral
    - Se eliminan parches aislados < min_patch_size píxeles
    """
    log.info(f"  Clasificando LULC para {year}...")

    with rasterio.open(landsat_path) as src:
        bands = src.read().astype(np.float32)
        profile = src.profile
        transform = src.transform

    # Asumir que el GeoTIFF tiene bandas en orden:
    # B2, B3, B4(Red), B5(NIR), B6(SWIR1), B7, NDVI, NDBI, EVI
    # Ajustar índices según el orden real de exportación de GEE
    red  = bands[2]   # SR_B4
    nir  = bands[3]   # SR_B5
    swir = bands[4]   # SR_B6

    # Calcular índices (por si el archivo no los incluye)
    ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red + 1e-9), 0)
    ndbi = np.where((swir + nir) != 0, (swir - nir) / (swir + nir + 1e-9), 0)

    # Clasificación binaria
    urban_mask = (
        (ndvi < LULC_CONFIG["ndvi_urban_max"]) &
        (ndbi > LULC_CONFIG["ndbi_urban_min"])
    ).astype(np.uint8)

    # Limpieza morfológica (elimina pequeños parches de ruido)
    struct = np.ones((3, 3), dtype=bool)
    urban_clean = binary_opening(urban_mask, structure=struct).astype(np.uint8)

    # Eliminar parches menores a min_patch_size
    labeled, n_features = label(urban_clean)
    sizes = np.bincount(labeled.ravel())
    small_patches = sizes < LULC_CONFIG["min_patch_size"]
    small_patches[0] = False
    urban_clean[small_patches[labeled]] = 0

    # Guardar GeoTIFF
    out_path = PATHS["lulc_base"].format(year=year)
    profile.update(count=1, dtype=rasterio.uint8, compress="lzw", nodata=255)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(urban_clean[np.newaxis, ...])

    urban_pct = urban_clean.mean() * 100
    log.info(f"  ✓ LULC {year}: {urban_pct:.1f}% urbano → {out_path}")
    return urban_clean, transform, profile


# ─────────────────────────────────────────────────────────────
# 2. VARIABLES ESPACIALES
# ─────────────────────────────────────────────────────────────

def compute_distance_to_urban_edge(urban: np.ndarray) -> np.ndarray:
    """Distancia euclidiana en píxeles al borde más cercano del área urbana."""
    non_urban = 1 - urban
    dist_pixels = distance_transform_edt(non_urban)
    return (dist_pixels * PIXEL_RESOLUTION).astype(np.float32)


def compute_distance_to_center(shape: tuple, transform) -> np.ndarray:
    """
    Distancia de cada píxel al centro histórico de Mérida (Plaza Grande).
    """
    import rasterio
    from rasterio.transform import rowcol

    rows, cols = shape
    center_lon, center_lat = STUDY_AREA["center"]

    # Crear grid de coordenadas
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]

    # Convertir centro histórico a píxeles
    center_row, center_col = rowcol(transform, center_lon, center_lat)

    dist_pixels = np.sqrt(
        (row_idx - center_row)**2 + (col_idx - center_col)**2
    )
    return (dist_pixels * PIXEL_RESOLUTION).astype(np.float32)


def compute_distance_to_roads(roads_path: str, shape: tuple, transform, crs: str) -> np.ndarray:
    """
    Rasteriza la red vial y calcula distancia euclidiana desde cada píxel.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    log.info("    Rasterizando red vial...")

    if not os.path.exists(roads_path):
        log.warning(f"    Red vial no encontrada: {roads_path}. Usando distancias nulas.")
        return np.zeros(shape, dtype=np.float32)

    roads = gpd.read_file(roads_path).to_crs(crs)
    road_raster = rasterize(
        [(mapping(geom), 1) for geom in roads.geometry if not geom.is_empty],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    dist_pixels = distance_transform_edt(1 - road_raster)
    return (dist_pixels * PIXEL_RESOLUTION).astype(np.float32)


def compute_neighborhood_density(urban: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Fracción de píxeles urbanos en una ventana de (2*radius+1) × (2*radius+1).
    Esta es la variable clave del Autómata Celular.
    """
    window = 2 * radius + 1
    density = uniform_filter(urban.astype(np.float32), size=window)
    return density.astype(np.float32)


def load_ndvi_mean(landsat_paths: list) -> np.ndarray:
    """Promedio de NDVI a lo largo de varios años como indicador de vegetación histórica."""
    ndvi_stack = []
    for path in landsat_paths:
        if not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            bands = src.read().astype(np.float32)
        red = bands[2]
        nir = bands[3]
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red + 1e-9), 0)
        ndvi_stack.append(ndvi)

    if not ndvi_stack:
        return np.zeros_like(red)
    return np.mean(ndvi_stack, axis=0).astype(np.float32)


def extract_features(year: int, urban: np.ndarray, transform, crs: str, landsat_paths: list) -> np.ndarray:
    """
    Construye el stack de features para un año dado.
    Retorna array (n_features, rows, cols).

    Features:
        0: dist_urban_edge   — distancia al borde urbano (m)
        1: dist_center       — distancia al centro histórico (m)
        2: dist_road         — distancia a carretera más cercana (m)
        3: ndvi_mean         — NDVI promedio histórico
        4: neighbor_3x3      — densidad vecindad 3x3
        5: neighbor_5x5      — densidad vecindad 5x5
        6: neighbor_9x9      — densidad vecindad 9x9
    """
    log.info(f"  Extrayendo features para {year}...")
    shape = urban.shape

    f0 = compute_distance_to_urban_edge(urban)
    log.info("    f0 dist_urban_edge ✓")

    f1 = compute_distance_to_center(shape, transform)
    log.info("    f1 dist_center ✓")

    f2 = compute_distance_to_roads(PATHS["roads"], shape, transform, crs)
    log.info("    f2 dist_road ✓")

    f3 = load_ndvi_mean(landsat_paths)
    log.info("    f3 ndvi_mean ✓")

    f4 = compute_neighborhood_density(urban, radius=1)   # 3x3
    f5 = compute_neighborhood_density(urban, radius=2)   # 5x5
    f6 = compute_neighborhood_density(urban, radius=4)   # 9x9
    log.info("    f4-f6 vecindad ✓")

    feature_stack = np.stack([f0, f1, f2, f3, f4, f5, f6], axis=0)

    # Normalizar distancias (0-1) para mejor convergencia del RF
    for i in [0, 1, 2]:
        max_val = np.percentile(feature_stack[i], 99)
        if max_val > 0:
            feature_stack[i] = np.clip(feature_stack[i] / max_val, 0, 1)

    return feature_stack.astype(np.float32)


def save_features(feature_stack: np.ndarray, year: int, profile: dict):
    """Guarda el stack de features como GeoTIFF multibanda."""
    out_path = PATHS["features"].format(year=year)
    n_bands = feature_stack.shape[0]
    profile.update(count=n_bands, dtype=rasterio.float32, compress="lzw", nodata=-9999)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(feature_stack)
        # Nombres de bandas para referencia
        feature_names = [
            "dist_urban_edge", "dist_center", "dist_road",
            "ndvi_mean", "neighbor_3x3", "neighbor_5x5", "neighbor_9x9"
        ]
        for i, name in enumerate(feature_names, start=1):
            dst.update_tags(i, name=name)

    log.info(f"  ✓ Features {year} guardadas: {n_bands} bandas → {out_path}")


# ─────────────────────────────────────────────────────────────
# 3. DATASET DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def build_training_dataset(year_pairs: list) -> tuple:
    """
    Construye el dataset X, y para entrenar el modelo de transición.

    Para cada par (t0, t1):
      - X = features en t0 para píxeles que eran no-urbanos en t0
      - y = 1 si el píxel se urbanizó en t1, 0 si siguió siendo no-urbano

    Aplica submuestreo para balancear clases y manejar memoria.
    """
    import pandas as pd

    log.info("  Construyendo dataset de entrenamiento...")
    all_X = []
    all_y = []

    for t0, t1 in year_pairs:
        lulc_t0_path = PATHS["lulc_base"].format(year=t0)
        lulc_t1_path = PATHS["lulc_base"].format(year=t1)
        feat_t0_path = PATHS["features"].format(year=t0)

        if not all(os.path.exists(p) for p in [lulc_t0_path, lulc_t1_path, feat_t0_path]):
            log.warning(f"  Archivos faltantes para par {t0}-{t1}, saltando.")
            continue

        log.info(f"  Par {t0}→{t1}:")

        with rasterio.open(lulc_t0_path) as src:
            urban_t0 = src.read(1).astype(np.uint8)
        with rasterio.open(lulc_t1_path) as src:
            urban_t1 = src.read(1).astype(np.uint8)
        with rasterio.open(feat_t0_path) as src:
            features = src.read().astype(np.float32)

        # Máscara: solo celdas no-urbanas en t0
        non_urban_mask = (urban_t0 == 0)

        # Variable objetivo: ¿se urbanizó en t1?
        y = urban_t1[non_urban_mask].astype(np.uint8)

        # Features para celdas no-urbanas en t0
        n_features = features.shape[0]
        X = features.reshape(n_features, -1).T[non_urban_mask.ravel()]

        # Submuestreo estratificado para manejar memoria
        max_samples = RF_CONFIG["max_train_samples"] // len(year_pairs)
        if len(y) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y), size=max_samples, replace=False)
            X, y = X[idx], y[idx]

        pos_rate = y.mean() * 100
        log.info(f"    {len(y):,} muestras, {pos_rate:.1f}% urbanización")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        log.error("No hay datos de entrenamiento disponibles.")
        sys.exit(1)

    X_final = np.vstack(all_X)
    y_final = np.concatenate(all_y)

    # Guardar como parquet (eficiente para datos geoespaciales)
    feature_names = [
        "dist_urban_edge", "dist_center", "dist_road",
        "ndvi_mean", "neighbor_3x3", "neighbor_5x5", "neighbor_9x9"
    ]
    df = pd.DataFrame(X_final, columns=feature_names)
    df["urban"] = y_final

    out_path = os.path.join(PATHS["processed"], "training_dataset.parquet")
    df.to_parquet(out_path, index=False)
    log.info(f"  ✓ Dataset guardado: {len(df):,} muestras → {out_path}")

    return X_final, y_final


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PASO 2: Preprocesamiento y extracción de features")
    log.info("=" * 60)

    crs = STUDY_AREA["crs"]
    landsat_paths = {
        year: os.path.join(PATHS["raw"], f"landsat_{year}.tif")
        for year in TRAIN_YEARS
    }

    # Verificar que existen los datos descargados
    missing = [p for p in landsat_paths.values() if not os.path.exists(p)]
    if missing:
        log.warning(f"Archivos LANDSAT faltantes: {missing}")
        log.warning("Ejecuta primero: python scripts/01_download_data.py")
        log.warning("Y descarga los archivos desde Google Drive a data/raw/")

    # Procesar cada año
    transforms = {}
    profiles = {}
    for year in TRAIN_YEARS:
        path = landsat_paths[year]
        if not os.path.exists(path):
            log.warning(f"  Saltando {year}: archivo no encontrado")
            continue

        log.info(f"\n[{year}] Procesando...")
        urban, transform, profile = classify_lulc(path, year)
        transforms[year] = transform
        profiles[year] = profile

        # Features con historial de NDVI (años anteriores disponibles)
        hist_paths = [landsat_paths[y] for y in TRAIN_YEARS if y <= year and os.path.exists(landsat_paths[y])]
        feat_stack = extract_features(year, urban, transform, crs, hist_paths)
        save_features(feat_stack, year, profile.copy())

    # Construir dataset de entrenamiento con pares de años
    year_pairs = list(zip(TRAIN_YEARS[:-1], TRAIN_YEARS[1:]))
    log.info(f"\n[DATASET] Pares de entrenamiento: {year_pairs}")
    build_training_dataset(year_pairs)

    log.info("\n" + "=" * 60)
    log.info("✓ Paso 2 completado.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
