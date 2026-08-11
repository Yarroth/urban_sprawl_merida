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
│   ├── 05_visualize.py      # Mapas, gráficas por escenario, reporte
│   ├── 06_calibrar_tasas.py # Calibración empírica de tasas (ZMM 1998–2020)
│   └── 07_seguridad_peatonal.py  # Incidentes peatonales vs densidad (Periférico)
├── frontend/
│   ├── generar_dashboard.py # Genera el dashboard HTML (mapas + métricas + seguridad)
│   ├── dashboard_resultados.html  # Dashboard interactivo completo
│   └── dashboard_resultados.pdf   # Versión imprimible A4 (entrega final)
├── docs/                    # Entregables: propuestas, estudios, speech, matemáticas
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
| Tasa anual de conversión | 3.5% | 3.1% | 2.3% |

La tasa anual por escenario (campo `growth_rate` en `CA_CONFIG["scenarios"]`)
modula el cupo de conversión del CA: los escenarios con gestión crecen menos y
distinto, de modo que el área proyectada diverge además de la ubicación.

## Calibración empírica de tasas (ZMM 1998–2020)

Las tasas por escenario se calibraron contra la evolución histórica real de la
Zona Metropolitana de Mérida. El script `scripts/06_calibrar_tasas.py`
reconstruye la TCMA observada en los LULC del pipeline, la contrasta con estas
referencias y verifica `CA_CONFIG` (reporte: `results/reports/calibracion_tasas.md`):

| Indicador (fuente) | Periodo | Crecimiento | TCMA |
|---|---|---|---|
| Superficie construida ZMM (IMEPLAN) | 2000–2020 | 211 → 422 km² (×2.0) | **3.5%/año** |
| Expansión urbana ZMM (CitiesAdapt/GIZ) | 2000–2020 | +84.6% | **3.1%/año** |
| Mancha urbana, dispersión rápida (López Santillán) | 1998–2010 | 159 → 270 km² | 4.5%/año |
| Población ZMM (INEGI/EURE) | 2010–2020 | 1.3 M hab. | 2.24%/año |

| Escenario | Regla | Tasa |
|---|---|---|
| `no_plan` | Dispersión = tendencia histórica observada (IMEPLAN) | **3.5%** |
| `plan_trad` | Tendencia con instrumentos tradicionales (Diagnóstico CitiesAdapt) | **3.1%** |
| `ia_optimo` | Densificación = el suelo crece al ritmo de la población (2.24%) | **2.3%** |

La brecha entre la TCMA de superficie (~3.5%) y la poblacional (~2.2%) mide la
dispersión histórica de la ZMM (~1.3 puntos porcentuales/año): el escenario
`ia_optimo` la elimina (densificación completa), mientras que `no_plan` la
reproduce. Con datos LULC reales (fuera del rango plausible 2–6% el script
avisa y usa las referencias publicadas, como ocurre con los sintéticos de la
demo).

---

## Módulo de seguridad peatonal — Anillo Periférico

Valida la relación **incidentes peatonales ↔ densidad vehicular** en el
Periférico de Mérida (~150,000 veh/día; 20–25 muertes/año, **17 en 2025**;
~68% de las víctimas son peatones) y simula 5 escenarios de política urbana
(`python scripts/07_seguridad_peatonal.py` → `results/safety/`):

| Escenario | Muertes/año | Reducción | Mecanismo |
|-----------|-------------|-----------|-----------|
| Situación actual (2025) | 17.0 | — | línea base calibrada |
| Semaforización coordinada | 12.4 | −27% | fases peatonales + LPI en cruces |
| Pirámide de movilidad | 9.5 | −44% | 80→60 km/h en tramos urbanos + cruces prioritarios |
| Paradas de autobús accesibles | 12.6 | −26% | traslado modal (−15% volumen) + cruces seguros en paradas |
| Visión Cero (todo combinado) | 6.7 | **−61%** | volumen + velocidad + infraestructura se potencian |

**Estructura de 12 sectores (N→WSW)** según el atlas *Análisis del Anillo
Periférico de Mérida* (lámina "12 sectores"): los pesos de demanda se extrajeron
de la densidad de celdas de alta concentración vehicular del mapa TDPA
(S 0.126 > SW 0.109 > SSW 0.108 > WSW 0.095 > NNE 0.094 > … > NE 0.045),
consistente con la congestión reportada. Aforo total ~150k veh/día repartido
por esos pesos y cruces documentados del anillo: **26 semáforos vehiculares,
16 peatonales** (Saidén Ojeda, Gob. Yucatán), **15 puentes** (8 nuevos + 7
rehabilitados), **18 cruces seguros** y **9 bahías** (programa de seguridad vial).

Modelo: `muertes = base × (V_esc/V_base)^0.6 × severidad(v) × [0.6·cross·stops + 0.4]`,
con severidad por velocidad según la curva de fatalidad peatonal WHO (2008:
30 km/h→10% … 80 km/h→95%). Calibrado contra cifras publicadas (Gob. de
Yucatán, Diario de Yucatán, Azteca Yucatán; ver `config/settings.py` →
`SAFETY_CONFIG` y `results/safety/seguridad_peatonal.md`).

**Dimensión temporal 2020–2025**: con el parque vehicular creciendo +77% en una
década (~5.9%/año, INEGI), el contrafactual "solo parque" daría 131 muertes
acumuladas vs 109 observadas (22 ya evitadas por medidas actuales); de haber
estado vigente Visión Cero todo el periodo habría evitado **~65 vidas**
(`results/safety/evolucion_temporal.csv` y `evolucion_temporal.png`).

**Validación de pesos contra atropellamientos reales** (prensa 2024–2026,
corpus de 18 incidentes localizados por tramo en `SAFETY_CONFIG`): S es líder
en atlas y en prensa, y la correlación del riesgo del modelo mejoró de **0.06 a
0.35** al incorporar la **capa de demanda peatonal** (`ped_demand`: intensidad
peatonal del atlas + demanda revelada por prensa + puntos de deseo de cruce
como Cholul, Chichí Suárez y Kanasín — auditor vial R. Flores Ayora, Diario de
Yucatán 04/2024), que eleva el riesgo de E/SE/NE hacia donde ocurren los
atropellamientos. Aun con eso, con n=12 los valores no son estadísticamente
significativos y la prensa sobrerrepresenta Norte, Oriente y SE — un censo
oficial por tramo (SSP/IMEPLAN) cerraría la brecha
(`results/safety/validacion_pesos.csv` y sección en `seguridad_peatonal.md`).

**Visión Cero priorizada por puntos de deseo de cruce** (`vision_cero_priorizada.csv`
y `vision_cero_priorizada.png`): al escalar los levers con la demanda peatonal
por sector (efectividad = 0.35 + 0.65 × demanda normalizada), la versión
priorizada logra **−54%** de muertes con **−22% de presupuesto** frente a la
uniforme (−61%), es decir **+14% de eficiencia** (vidas por unidad invertida):
concentra el recorte donde ocurren los atropellamientos (S, NE/Chichí Suárez,
SSE, E, SE/Kanasín, WSW mantienen casi todo el efecto) y relaja los tramos de
baja demanda medida (ENE, ESE). La uniforme sigue siendo el techo en vidas
totales, pero cuesta ~22% más por solo 6.5 pp adicionales.

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

## Dashboard y entrega final

`frontend/dashboard_resultados.html` consolida todo el análisis (mapas de
probabilidad por escenario y año, comparaciones 2024→2030, calidad espacial,
seguridad peatonal del Periférico). Para la versión imprimible:

```bash
python frontend/generar_dashboard.py   # regenera el HTML
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=frontend/dashboard_resultados.pdf \
  file://$(pwd)/frontend/dashboard_resultados.html
```

El PDF resultante (A4 apaisado, 13 páginas) incluye portada con resumen
ejecutivo y los estilos de impresión definidos en `@media print` del generador:
tema claro, una página por sección y los 15 mapas en cuadrículas de 5.

---

## Referencias

- Ke et al. (2017). LightGBM. NIPS 2017.
- Pontius & Schneider (2001). Land-cover change model validation. Ag. Ecosyst. Environ.
- White & Engelen (1993). Cellular automata and fractal urban form. Env. Planning A.
- López-Rivera & Romero-Huertas (2021). RNA+CA vertical urban growth. Springer. [antecedente]
