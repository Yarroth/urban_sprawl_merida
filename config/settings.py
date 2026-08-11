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

    # CA de reglas aprendidas: columnas reales del dataset X_ca, en orden.
    # X_ca = features disponibles (FEATURE_NAMES ∩ dataset) + P_LightGBM
    # out-of-fold al final (ver build_ca_training_data en 03_train_model.py).
    # Se deriva de FEATURE_NAMES para evitar que la lista se desincronice.
    "ca_rule_features": FEATURE_NAMES + ["p_lgbm"],

    # Escenarios predefinidos
    # growth_rate: tasa de conversión anual por escenario, calibrada contra la
    # evolución histórica de la ZMM 2000-2020 (ver scripts/06_calibrar_tasas.py):
    #   no_plan    3.5%/año = TCMA de superficie construida ZMM 2000→2020
    #                         (21 103 → 42 186 ha, IMEPLAN vía Novedades Yucatán)
    #   plan_trad  3.1%/año = TCMA de expansión urbana ZMM 2000→2020
    #                         (+84.6% total, Diagnóstico CitiesAdapt GIZ)
    #   ia_optimo  2.3%/año = convergencia a la TCMA poblacional de la ZMM
    #                         (2.24%/año 2010-2020, INEGI; núcleo Mérida 2.2%)
    #                         = densificación: el suelo crece al ritmo de la
    #                           población, no más rápido (dispersión ≈ +1.3 pp).
    # La gestión desacelera la expansión frente a no_plan, de modo que el área
    # proyectada diverge además de la ubicación.
    "scenarios": {
        "no_plan":    {"alpha":0.60,"beta":0.30,"gamma":0.00,"delta":0.10,
                       "exclusions":[], "growth_rate": 0.035},
        "plan_trad":  {"alpha":0.55,"beta":0.30,"gamma":0.05,"delta":0.10,
                       "exclusions":["cenotes","reservas"], "growth_rate": 0.031},
        "ia_optimo":  {"alpha":0.55,"beta":0.25,"gamma":0.15,"delta":0.05,
                       "exclusions":["cenotes","reservas","lst_hotspots","karst_alta"],
                       "growth_rate": 0.023},
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

# ── SEGURIDAD PEATONAL — ANILLO PERIFÉRICO DE MÉRIDA ★ ───────
# Módulo 07: valida la relación incidentes peatonales vs densidad vehicular y
# simula escenarios de política (pirámide de movilidad, semaforización,
# paradas accesibles). Fuentes usadas para la calibración:
#   - Aforo del Periférico: ~150,000 veh/día (+20% en temporada) — Gob. de
#     Yucatán (2025), SIPSE (2015).
#   - Muertes en el Periférico: 20–25/año históricas; 17 en 2025 — Diario de
#     Yucatán; 2022 (ene–abr): 186 accidentes, 147 lesionados, 5 muertes —
#     Azteca Yucatán (vía Top 5 nacional de vías peligrosas).
#   - Peatones entre las víctimas: ~68% (34 de 50 fallecidos por atropellamiento
#     en el anillo).
#   - Parque vehicular ZMM: ~838,726 vehículos (2023); 1.8 personas/veh (2020).
SAFETY_CONFIG = {
    "ring_length_km": 48.0,          # longitud aproximada del anillo
    "aadt_total": 150_000,           # veh/día en toda la sección
    "baseline_deaths_year": 17,      # muertes totales 2025 (ancla de calibración)
    "pedestrian_share": 0.68,        # fracción de víctimas que son peatones
    "speed_limit_kmh": 80,           # límite nominal actual
    "crossing_share": 0.60,          # fracción de muertes en cruces con semáforo
    "volume_exponent": 0.6,          # elasticidad incidentes vs volumen (0.5–1.0 en literatura)
    "speed_fatality_exponent": 3.0,  # severidad ∝ (V/80)^3 (curva WHO/ETSC)
    # Infraestructura real del anillo (Gob. de Yucatán / prensa):
    #   - 26 semáforos vehiculares y 16 peatonales (Saidén Ojeda, Gob. Yucatán)
    #   - Programa de seguridad: 8 puentes nuevos + 7 rehabilitados = 15;
    #     18 cruces peatonales seguros; 9 bahías de ascenso/descenso
    "infra_real": {
        "semaforos_vehiculares": 26,
        "semaforos_peatonales": 16,
        "puentes_peatonales": 15,
        "cruces_seguros": 18,
        "bahias_bus": 9,
    },
    # 12 sectores del anillo (~4 km c/u; total ~48 km) según el atlas
    # "Análisis del Anillo Periférico de Mérida" (corredor 500 m, red vial y
    # movilidad; 27 láminas de hexágonos N→WSW). Los pesos = densidad de celdas
    # de alta concentración vehicular extraída de la lámina "12 sectores"
    # (fracción de área roja del mapa de TDPA, escala compartida), normalizada:
    #   S 0.126 > SW 0.109 > SSW 0.108 > WSW 0.095 > NNE 0.094 > ESE 0.085
    #   > E 0.071 > ENE 0.069 > N 0.067 > SSE 0.066 > SE 0.064 > NE 0.045
    # Consistente con congestión reportada (Sur/City Center, Suroeste/Caucel,
    # Norte/Las Américas; Noreste el más ligero). Semáforos, puentes y cruces
    # seguros por sector suman los totales de infra_real; E/ESE/SE = tramos
    # menos urbanos del anillo.
    "segments": [
        {"name": "N",   "weight": 0.068, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 2, "urban": True},
        {"name": "NNE", "weight": 0.094, "length_km": 4, "crossings": 2, "ped_signals": 2,
         "bridges": 1, "safe_crossings": 2, "bus_stops": 2, "urban": True},
        {"name": "NE",  "weight": 0.045, "length_km": 4, "crossings": 1, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 1, "urban": True},
        {"name": "ENE", "weight": 0.069, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 2, "urban": True},
        {"name": "E",   "weight": 0.071, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 2, "urban": False},
        {"name": "ESE", "weight": 0.085, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 2, "bus_stops": 2, "urban": False},
        {"name": "SE",  "weight": 0.064, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 2, "urban": False},
        {"name": "SSE", "weight": 0.066, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 1, "safe_crossings": 1, "bus_stops": 2, "urban": True},
        {"name": "S",   "weight": 0.126, "length_km": 4, "crossings": 3, "ped_signals": 2,
         "bridges": 2, "safe_crossings": 2, "bus_stops": 3, "urban": True},
        {"name": "SSW", "weight": 0.108, "length_km": 4, "crossings": 3, "ped_signals": 2,
         "bridges": 1, "safe_crossings": 2, "bus_stops": 3, "urban": True},
        {"name": "SW",  "weight": 0.109, "length_km": 4, "crossings": 3, "ped_signals": 2,
         "bridges": 2, "safe_crossings": 2, "bus_stops": 3, "urban": True},
        {"name": "WSW", "weight": 0.095, "length_km": 4, "crossings": 2, "ped_signals": 1,
         "bridges": 2, "safe_crossings": 2, "bus_stops": 2, "urban": True},
    ],
    # ── Serie temporal 2020–2025 ──
    # Muertes reportadas por año en el Periférico (prensa local):
    #   2022 = 19 fallecimientos (Canal Doce / Reporteros Hoy)
    #   2024 ≈ 14 (Por Esto: "3 meses de 2025 iguala todo 2024")
    #   2025 = 17 (Diario de Yucatán); 2026 = 18 a julio (Yucatán.com.mx)
    # 2020/2021/2023 se interpolaron dentro de la banda histórica 18–21
    # (20–25 anuales según conteos históricos).
    "deaths_observed": {2020: 20, 2021: 21, 2022: 19, 2023: 18, 2024: 14, 2025: 17},
    # Crecimiento del parque vehicular de Yucatán (INEGI): +77% en una década
    # (~5.9%/año); >1.1M unidades en 2024–25 (crecimiento anual >6%).
    "fleet_growth_annual": 0.059,
    # Levers por escenario (factores multiplicativos sobre la tasa de muertes):
    #   vol  = cambio de volumen vehicular (1.0 = sin cambio)
    #   speed= límite efectivo en tramos urbanos (km/h)
    #   cross= factor de riesgo en cruces semaforizados (semáforos coordinados,
    #          fases peatonales, LPI, cruces protegidos)
    #   stops= factor de riesgo en cruces asociados a paradas de autobús
    "scenarios": {
        "base":               {"vol": 1.00, "speed": 80, "cross": 1.00, "stops": 1.00},
        "semaforizacion":     {"vol": 1.00, "speed": 80, "cross": 0.55, "stops": 1.00},
        "piramide_movilidad": {"vol": 0.95, "speed": 60, "cross": 0.50, "stops": 1.00},
        "transito_accesible": {"vol": 0.85, "speed": 80, "cross": 1.00, "stops": 0.70},
        "vision_cero":        {"vol": 0.80, "speed": 60, "cross": 0.40, "stops": 0.60},
    },
}
