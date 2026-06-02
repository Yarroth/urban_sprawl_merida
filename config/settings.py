"""
config/settings.py
Parámetros globales del proyecto de predicción de expansión urbana - Mérida, Yucatán.
Ajusta estos valores según tus necesidades y datos disponibles.
"""

# ─────────────────────────────────────────────
# ÁREA DE ESTUDIO
# ─────────────────────────────────────────────
STUDY_AREA = {
    "name": "Mérida, Yucatán, México",
    "bbox": {
        # Bounding box que cubre la ZMM (Zona Metropolitana de Mérida)
        # Formato: [min_lon, min_lat, max_lon, max_lat]
        "min_lon": -90.05,
        "min_lat":  20.75,
        "max_lon": -89.45,
        "max_lat":  21.25,
    },
    # Coordenadas del centro histórico (Plaza Grande)
    "center": (-89.6237, 20.9674),
    # CRS de salida (UTM zona 16N — ideal para Yucatán)
    "crs": "EPSG:32616",
}

# ─────────────────────────────────────────────
# RESOLUCIÓN Y AÑOS
# ─────────────────────────────────────────────
PIXEL_RESOLUTION = 30          # metros por píxel (LANDSAT nativo)
TRAIN_YEARS      = [2015, 2020, 2024]   # años históricos para entrenamiento
PREDICT_YEARS    = [2026, 2027, 2028, 2029, 2030]  # años a simular
BASE_YEAR        = 2024        # año de inicio de la simulación

# ─────────────────────────────────────────────
# DATOS SATELITALES (Google Earth Engine)
# ─────────────────────────────────────────────
GEE_CONFIG = {
    "collection": "LANDSAT/LC09/C02/T1_L2",   # LANDSAT 9, Collection 2
    "fallback":   "LANDSAT/LC08/C02/T1_L2",   # LANDSAT 8 como respaldo
    "cloud_cover_max": 15,     # % máximo de nubosidad permitida
    "months_dry_season": [1, 2, 3, 4, 11, 12],  # meses de temporada seca en Yucatán
    # Bandas necesarias (escala de reflectancia superficial)
    "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    "scale_factor": 0.0000275,
    "offset": -0.2,
}

# ─────────────────────────────────────────────
# CLASIFICACIÓN LULC
# ─────────────────────────────────────────────
LULC_CONFIG = {
    # Umbrales de índices espectrales para clasificación supervisada
    "ndvi_urban_max":  0.20,   # NDVI < 0.20 → probable urbano
    "ndbi_urban_min":  0.05,   # NDBI > 0.05 → probable urbano
    # Clases de salida
    "classes": {0: "no_urbano", 1: "urbano"},
    # Tamaño mínimo de parche para eliminar ruido (en píxeles)
    "min_patch_size": 9,
}

# ─────────────────────────────────────────────
# VARIABLES ESPACIALES (FEATURES)
# ─────────────────────────────────────────────
FEATURES_CONFIG = {
    # Distancias (en metros)
    "max_dist_road":        5000,  # distancia máxima a carretera
    "max_dist_urban_edge":  8000,  # distancia máxima al borde urbano
    "max_dist_center":     30000,  # distancia máxima al centro histórico
    # Ventanas de vecindad para regla CA (en píxeles)
    "neighbor_windows": [3, 5, 9],  # 3x3, 5x5, 9x9
    # Historial temporal de NDVI (años de promedio)
    "ndvi_history_years": 3,
}

# ─────────────────────────────────────────────
# MODELO RANDOM FOREST
# ─────────────────────────────────────────────
RF_CONFIG = {
    "n_estimators":     300,
    "max_depth":         20,
    "min_samples_leaf":   5,
    "class_weight":  "balanced",  # compensa desbalance urbano/no-urbano
    "n_jobs":            -1,       # usa todos los cores
    "random_state":      42,
    # Tamaño del set de validación
    "test_size":          0.25,
    # Número máximo de muestras de entrenamiento (por memoria)
    "max_train_samples": 500_000,
}

# ─────────────────────────────────────────────
# AUTÓMATA CELULAR
# ─────────────────────────────────────────────
CA_CONFIG = {
    # Pesos de la función de probabilidad total:
    # P_total = alpha*P_RF + beta*P_neighbors + gamma*P_stochastic
    "alpha": 0.60,   # peso del modelo RF
    "beta":  0.30,   # peso de la regla de vecindad
    "gamma": 0.10,   # componente estocástica

    # Umbral de probabilidad para convertir una celda a urbana
    "conversion_threshold": 0.50,

    # Tasa de crecimiento anual estimada (% del área urbana actual)
    # Basada en datos históricos ZMM: ~3.5% anual
    "annual_growth_rate": 0.035,

    # Radio de vecindad para regla CA (en píxeles)
    "neighborhood_radius": 5,

    # Restricciones de conversión (1 = no puede urbanizarse)
    "exclude_classes": ["agua", "cenote", "reserva_natural"],
}

# ─────────────────────────────────────────────
# RUTAS DE DATOS Y SALIDAS
# ─────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "raw":            os.path.join(BASE_DIR, "data", "raw"),
    "processed":      os.path.join(BASE_DIR, "data", "processed"),
    "output":         os.path.join(BASE_DIR, "data", "output"),
    "models":         os.path.join(BASE_DIR, "models"),
    "results_maps":   os.path.join(BASE_DIR, "results", "maps"),
    "results_reports":os.path.join(BASE_DIR, "results", "reports"),
    # Archivos específicos
    "lulc_base":      os.path.join(BASE_DIR, "data", "processed", "lulc_{year}.tif"),
    "features":       os.path.join(BASE_DIR, "data", "processed", "features_{year}.tif"),
    "rf_model":       os.path.join(BASE_DIR, "models", "rf_model.pkl"),
    "prediction":     os.path.join(BASE_DIR, "results", "maps", "prediction_{year}.tif"),
    "urban_extent":   os.path.join(BASE_DIR, "results", "maps", "urban_extent_{year}.shp"),
    "municipalities": os.path.join(BASE_DIR, "data", "raw", "merida_municipio.shp"),
    "roads":          os.path.join(BASE_DIR, "data", "raw", "red_vial_zmm.shp"),
}

# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────
VIZ_CONFIG = {
    "dpi": 200,
    "figsize": (14, 10),
    "colormap_probability": "YlOrRd",
    "colormap_lulc": {0: "#4CAF50", 1: "#E53935"},  # verde/rojo
    "output_format": "png",
}
