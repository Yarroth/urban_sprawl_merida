# Predicción de Expansión Urbana — Mérida, Yucatán
## Versión 2.0 — LightGBM + CA de Reglas Aprendidas + Variables Kársticas

Sistema de predicción de la mancha urbana de la ZMM para 2025–2030.
Genera tres escenarios comparativos (`no_plan`, `plan_trad`, `ia_optimo`)
con visualización interactiva.

---

## Estructura del proyecto (consolidada)

```
.
├── config/
│   └── settings.py          # Parámetros globales (bbox, años, features, pesos CA)
├── scripts/
│   ├── 01_download_data.py  # Descarga GEE / INEGI / OSM → data/raw/
│   ├── 02_preprocess.py     # Clasificación LULC + 7-10 features (kársticas si hay datos)
│   ├── 03_train_model.py    # LightGBM transición + LightGBM reglas CA (OOF sin fuga)
│   ├── 04_simulate_ca.py    # Simulación CA 3 escenarios → results/maps/*.tif
│   └── 05_visualize.py      # Mapas, gráficas por escenario, reporte
├── frontend/
│   └── *.html               # Dashboards interactivos (probabilidad, comparación 3D)
├── docs/                    # Entregables: propuestas, estudios, speech, matemáticas
│   └── archivo/             # Zips históricos (snapshots ya absorbidos por git)
├── demo_merida.py           # Demo completa con datos sintéticos (sin GEE/rasterio)
├── gen_docs.js              # Generador del DOCX de documentación técnica
├── requirements.txt
└── README.md
```

Directorios generados por el pipeline (en `.gitignore`): `data/`, `models/`,
`results/` y `demo_output/`.

### Ramas de git

| Rama / tag | Contenido |
|------------|-----------|
| `main` | v2.0: LightGBM + CA de reglas aprendidas + variables kársticas |
| `v1-rf` (rama) / `v1.0` (tag) | v1.0: Random Forest + CA clásico (historia original) |

---

## Contribuciones originales (v2.0)

★ **Variables kársticas**: LST (temperatura superficial), distancia a cenotes (SEDUMA),
  vulnerabilidad del acuífero kárstico (CONAGUA/IMTA) — ausentes en trabajos previos.
  Se extraen en `02_preprocess.py` si los datos reales existen; si no, la simulación
  usa capas sintéticas y lo avisa por consola (los escenarios con gestión no son
  representativos en ese caso).

★ **CA de reglas aprendidas**: segundo LightGBM que aprende cuándo una celda se
  convierte basándose en el estado de vecindad, reemplazando el umbral estadístico
  fijo. La probabilidad P_LightGBM que alimenta al modelo CA se calcula
  **out-of-fold** para evitar fuga de datos.

---

## Instalación y ejecución

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Autenticar Google Earth Engine (solo primera vez)
earthengine authenticate

# 3. Pipeline completo (los pasos 01-05 se ejecutan desde la raíz)
python scripts/01_download_data.py   # descarga LANDSAT + datos vectoriales
python scripts/02_preprocess.py      # clasificación LULC + features (7 u 10)
python scripts/03_train_model.py     # LightGBM transición + reglas CA
python scripts/04_simulate_ca.py     # 3 escenarios: no_plan, plan_trad, ia_optimo
python scripts/05_visualize.py       # mapas, gráficas, reporte
```

> Nota: `01_download_data.py` exporta las imágenes a Google Drive; deben
> descargarse manualmente a `data/raw/` antes de ejecutar el paso 02.

## Demo rápida (sin datos externos)

```bash
pip install scikit-learn scipy joblib pandas matplotlib
python demo_merida.py          # genera todo en demo_output/
```

---

## Features del modelo (7 base + 3 kársticas)

| Feature | Descripción | Fuente |
|---------|-------------|--------|
| `dist_urban_edge` | Distancia al borde urbano (m) | Calculado |
| `dist_center` | Distancia al centro histórico (m) | Calculado |
| `dist_road` | Distancia a carretera más cercana (m) | OSM / INEGI |
| `ndvi_mean` | Vegetación media histórica | LANDSAT 8/9 |
| `neighbor_3x3 / 5x5 / 9x9` | Densidad urbana en vecindad | Calculado |
| `lst_mean` ★ | Temperatura superficial (°C) | LANDSAT ST_B10 |
| `dist_cenote` ★ | Distancia al cenote más cercano (m) | SEDUMA |
| `karst_vuln` ★ | Vulnerabilidad acuífero kárstico (0-1) | CONAGUA / IMTA |

★ Las features kársticas se incluyen solo si los datos reales están disponibles
(`data/raw/cenotes_seduma.shp`, `data/raw/vulnerabilidad_karstica.tif` y la banda
térmica en el LANDSAT); si faltan, el pipeline sigue con las 7 base y lo indica
en los logs.

---

## Salidas (resultados/maps) — GeoTIFF por año y escenario

| Archivo | Descripción |
|---------|-------------|
| `prediction_{year}_{scenario}.tif` | Mapa de probabilidad de urbanización (float32) |
| `urban_extent_{year}_{scenario}.tif` | Área urbana binaria (uint8) |
| `expansion_map_{year}_{scenario}.png` | Mapa de probabilidad renderizado |
| `comparison_2024_2030_{scenario}.png` | Comparación base vs. 2030 |

Con `year ∈ {2026, 2027, 2028, 2029, 2030}` y
`scenario ∈ {no_plan, plan_trad, ia_optimo}`.

Reportes en `results/reports/`: `metrics.csv` (ambos modelos), `area_statistics.csv`
(área y crecimiento por año y escenario), `area_statistics.png`, `final_report.md`.

Modelos en `models/`: `lgbm_model.pkl` (transición) y `ca_rules_model.pkl` (reglas CA).

---

## Función de probabilidad

```
P_total = α · P_LightGBM + β · P_CA_aprendida + γ · (1 - P_kárstico) + δ · rand
```

| Parámetro | Sin plan | Plan trad. | Gestión IA |
|-----------|----------|------------|------------|
| α (LightGBM) | 0.60 | 0.55 | 0.55 |
| β (CA aprendida) | 0.30 | 0.30 | 0.25 |
| γ (kárstico) | 0.00 | 0.05 | 0.15 |
| δ (estocástico) | 0.10 | 0.10 | 0.05 |

---

## Hallazgos principales (proyección 2030)

| Indicador | Sin plan | Plan trad. | Gestión IA |
|-----------|----------|------------|------------|
| Área (km²) | 1,017 | 958 | 911 |
| Fragmentación | 0.67 | 0.26 | 0.20 |
| Cobertura verde | 5% | 23% | 29% |
| LST promedio | +3.6°C | +0.4°C | -1.1°C |
| Vuln. acuífero | 0.68 | 0.36 | 0.24 |

---

## Referencias

- Ke et al. (2017). LightGBM. NIPS 2017.
- Pontius & Schneider (2001). Land-cover change model validation. Ag. Ecosyst. Environ.
- White & Engelen (1993). Cellular automata and fractal urban form. Env. Planning A.
- López-Rivera & Romero-Huertas (2021). RNA+CA vertical urban growth. Springer. [antecedente]
