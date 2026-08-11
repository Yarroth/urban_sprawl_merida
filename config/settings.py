"""
config/settings.py  —  v2.0
Parámetros globales del proyecto. Versión 2.0 incorpora:
  - LightGBM en lugar de Random Forest
  - Variables kársticas específicas de Mérida (LST, cenotes, acuífero)
  - CA de reglas aprendidas (segundo LightGBM sobre vecindad)
  - Optimización multiobjetivo con Optuna
"""

# ── ÁREA DE ESTUDIO ──────────────────────────────────────────
STUDY_AREA = {
    "name": "Mérida, Yucatán, México",
    "bbox": {"min_lon": -90.05, "min_lat": 20.75, "max_lon": -89.45, "max_lat": 21.25},
    "center": (-89.6237, 20.9674),
    "crs": "EPSG:32616",
}

PIXEL_RESOLUTION = 30
TRAIN_YEARS      = [2015, 2020, 2024]
PREDICT_YEARS    = [2026, 2027, 2028, 2029, 2030]
BASE_YEAR        = 2024

# ── DATOS SATELITALES (GEE) ──────────────────────────────────
GEE_CONFIG = {
    "collection":        "LANDSAT/LC09/C02/T1_L2",
    "fallback":          "LANDSAT/LC08/C02/T1_L2",
    "cloud_cover_max":   15,
    "months_dry_season": [1, 2, 3, 4, 11, 12],
    "bands":             ["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7","ST_B10"],
    "scale_factor":      0.0000275,
    "offset":            -0.2,
    "lst_scale":         0.00341802,   # banda térmica ST_B10 → Kelvin
    "lst_offset":        149.0,
}

# ── CLASIFICACIÓN LULC ───────────────────────────────────────
LULC_CONFIG = {
    "ndvi_urban_max":  0.20,
    "ndbi_urban_min":  0.05,
    "classes":         {0: "no_urbano", 1: "urbano"},
    "min_patch_size":  9,
}

# ── VARIABLES KÁRSTICAS (contribución original) ──────────────
KARST_CONFIG = {
    # Temperatura superficial (LST) — isla de calor urbana
    "lst_weight":         0.35,
    "lst_threshold_hot":  35.0,       # °C — umbral de isla de calor

    # Cenotes — fuente del SEDUMA Yucatán
    "cenote_weight":      0.40,
    "cenote_exclusion_m": 200,        # radio de exclusión en metros
    "cenote_buffer_m":    500,        # radio de influencia kárstica

    # Vulnerabilidad acuífero kárstico
    "aquifer_weight":     0.25,
    # Índice de vulnerabilidad: 0=baja, 1=alta (basado en espesor del suelo)
    "aquifer_high_threshold": 0.6,
}

# ── FEATURES DEL MODELO ──────────────────────────────────────
# Versión 2.0: 10 features (7 base + 3 kársticas)
FEATURE_NAMES = [
    "dist_urban_edge",   # f0
    "dist_center",       # f1
    "dist_road",         # f2
    "ndvi_mean",         # f3
    "neighbor_3x3",      # f4
    "neighbor_5x5",      # f5
    "neighbor_9x9",      # f6
    "lst_mean",          # f7 ★ nuevo
    "dist_cenote",       # f8 ★ nuevo
    "karst_vuln",        # f9 ★ nuevo
]

# ── LIGHTGBM (reemplaza Random Forest) ───────────────────────
LGBM_CONFIG = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "n_jobs":          -1,
    "random_state":    42,
    "test_size":       0.25,
    "max_train_samples": 500_000,
    # Optuna — optimización de hiperparámetros
    "optuna_trials":   30,
    "optuna_timeout":  300,           # segundos
}

# ── AUTÓMATA CELULAR — REGLAS APRENDIDAS ★ ───────────────────
CA_CONFIG = {
    # Pesos de la función de probabilidad total
    "alpha":  0.55,   # P_LightGBM
    "beta":   0.25,   # P_vecindad aprendida
    "gamma":  0.15,   # P_kárstico
    "delta":  0.05,   # estocástico

    "annual_growth_rate":   0.035,
    "neighborhood_radius":  5,

    # CA de reglas aprendidas: features del estado de vecindad
    "ca_rule_features": [
        "nbr_density_3x3", "nbr_density_5x5", "nbr_density_9x9",
        "p_lgbm", "dist_edge", "lst_local", "karst_local"
    ],

    # Escenarios predefinidos
    "scenarios": {
        "no_plan":    {"alpha":0.60,"beta":0.30,"gamma":0.00,"delta":0.10,"exclusions":[]},
        "plan_trad":  {"alpha":0.55,"beta":0.30,"gamma":0.05,"delta":0.10,"exclusions":["cenotes","reservas"]},
        "ia_optimo":  {"alpha":0.55,"beta":0.25,"gamma":0.15,"delta":0.05,"exclusions":["cenotes","reservas","lst_hotspots","karst_alta"]},
    },
}

# ── RUTAS ────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "raw":             os.path.join(BASE_DIR, "data", "raw"),
    "processed":       os.path.join(BASE_DIR, "data", "processed"),
    "output":          os.path.join(BASE_DIR, "data", "output"),
    "models":          os.path.join(BASE_DIR, "models"),
    "results_maps":    os.path.join(BASE_DIR, "results", "maps"),
    "results_reports": os.path.join(BASE_DIR, "results", "reports"),
    "lulc_base":       os.path.join(BASE_DIR, "data", "processed", "lulc_{year}.tif"),
    "features":        os.path.join(BASE_DIR, "data", "processed", "features_{year}.tif"),
    "lgbm_model":      os.path.join(BASE_DIR, "models", "lgbm_model.pkl"),
    "ca_model":        os.path.join(BASE_DIR, "models", "ca_rules_model.pkl"),
    "prediction":      os.path.join(BASE_DIR, "results", "maps", "prediction_{year}_{scenario}.tif"),
    "urban_extent":    os.path.join(BASE_DIR, "results", "maps", "urban_extent_{year}_{scenario}.tif"),
    "municipalities":  os.path.join(BASE_DIR, "data", "raw", "merida_municipio.shp"),
    "roads":           os.path.join(BASE_DIR, "data", "raw", "red_vial_zmm.shp"),
    "cenotes":         os.path.join(BASE_DIR, "data", "raw", "cenotes_seduma.shp"),
    "lst":             os.path.join(BASE_DIR, "data", "processed", "lst_{year}.tif"),
    "karst_vuln":      os.path.join(BASE_DIR, "data", "raw", "vulnerabilidad_karstica.tif"),
}

VIZ_CONFIG = {
    "dpi": 200,
    "figsize": (18, 8),
    "colormap_probability": "YlOrRd",
    "scenario_colors": {"no_plan":"#E53935","plan_trad":"#43A047","ia_optimo":"#7C4DFF"},
}
