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
    # ── Validación con atropellamientos reales (prensa 2026) ──
    # Corpus de reportes localizados por tramo (prensa local: Diario de
    # Yucatán, Reporteros Hoy, InfoLliteras, Yucatán.com.mx, Sol Yucatán,
    # Novedades). Muestra pequeña y sesgada hacia eventos fatales: útil como
    # validación exploratoria, no como censo.
    "incidentes_prensa": [
        {"fecha": "2026-05-06", "sector": "N",   "fuente": "InfoLliteras", "nota": "periférico norte, 2ª muerte en <1 semana"},
        {"fecha": "2026-03-30", "sector": "N",   "fuente": "prensa local", "nota": "muere cruzando el norte sin usar puente"},
        {"fecha": "2026-07-01", "sector": "N",   "fuente": "Reporteros Hoy", "nota": "fallece al cruzar el tramo norte"},
        {"fecha": "2026-06-19", "sector": "S",   "fuente": "Yucatán.com.mx", "nota": "Salvador Alvarado Sur / Kanasín"},
        {"fecha": "2026-06-24", "sector": "S",   "fuente": "Yucatán.com.mx", "nota": "cerca de puente peatonal; 2º en la semana"},
        {"fecha": "2026-07-02", "sector": "S",   "fuente": "Instagram/prensa", "nota": "km 7, atropellan a estudiante"},
        {"fecha": "2026-07-05", "sector": "S",   "fuente": "Reporteros Hoy", "nota": "peatón grave en el periférico Sur"},
        {"fecha": "2026-05-05", "sector": "S",   "fuente": "Reporteros Hoy", "nota": "km 5, cerca del puente de la Av. 86"},
        {"fecha": "2026-06-19", "sector": "SE",  "fuente": "InfoLliteras", "nota": "periférico Sur Oriente, Mazda 2"},
        {"fecha": "2026-05-01", "sector": "SW",  "fuente": "prensa local", "nota": "km 48, dos personas fallecen"},
        {"fecha": "2026-07-08", "sector": "WSW", "fuente": "Diario de Yucatán", "nota": "grave atropellado en el Periférico Poniente"},
        {"fecha": "2026-04-11", "sector": "E",   "fuente": "Novedades Yucatán", "nota": "incidente fatal en el periférico oriente"},
        {"fecha": "2026-02-05", "sector": "E",   "fuente": "Sol Yucatán", "nota": "peatón muere en el Periférico Oriente"},
        # ── Históricos 2024–2025 (fuera del año de calibración) ──
        {"fecha": "2024-04-18", "sector": "SE",  "fuente": "Diario de Yucatán vía Yucatán al Mano",
         "nota": "peatón cruza en el puente vehicular de Kanasín; provoca volcadura (racha mar-abr 2024)"},
        {"fecha": "2024-12-01", "sector": "SE",  "fuente": "Yucatán.com.mx",
         "nota": "peatón atropellada a la altura de calle 42, Reparto Granjas de Kanasín"},
        {"fecha": "2025-07-01", "sector": "SE",  "fuente": "Reporteros Hoy",
         "nota": "km 13 del anillo, peatón fallece (fecha aprox.; km 13 ≈ SE entre Sur y Oriente)"},
        {"fecha": "2026-07-18", "sector": "E",   "fuente": "Reporteros Hoy", "nota": "accidente en intersección, entrada Col. Cactus"},
        {"fecha": "2026-04-15", "sector": "E",   "fuente": "Instagram/prensa", "nota": "fallece atropellado en el oriente"},
    ],
    # ── Demanda peatonal por sector (0–1) ──
    # Capa de puntos de deseo de cruce. Mezcla de: (a) intensidad peatonal del
    # atlas (lámina II.5 concentración de peatones: rojo = caminata intensa,
    # extraída por cuña de 30°), (b) demanda revelada por los atropellamientos
    # de prensa (prensa_share) y (c) cruces documentados de flujo alto
    # (puentes de Cholul, Chichí Suárez — auditor vial R. Flores Ayora, Diario
    # de Yucatán 04/2024 —, Kanasín, Xmatkuil, Dzununcán). Mínimo 0.25 para no
    # anular sectores con poca actividad medida pero cruces conflictivos.
    "ped_demand": {
        "N": 0.60,   # Cholul y acceso norte (flujo alto + prensa)
        "NNE": 0.71, # Cholul / área norte
        "NE": 0.87,  # Chichí Suárez — puente de mayor flujo (experto)
        "ENE": 0.25, # actividad medida baja
        "E": 0.80,   # demanda revelada por prensa (4 eventos) pese a menor actividad
        "ESE": 0.25, # actividad medida baja
        "SE": 0.80,  # Kanasín — cúmulo prensa 2024–2025 (puente vehicular, Reparto Granjas)
        "SSE": 0.83, # intensidad atlas alta
        "S": 1.00,   # Xmatkuil, Dzununcán + prensa
        "SSW": 0.50,
        "SW": 0.58,
        "WSW": 0.80, # poniente/Caucel (intensidad atlas alta)
    },
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
