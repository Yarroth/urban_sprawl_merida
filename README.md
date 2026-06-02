# Predicción de Expansión Urbana — Mérida, Yucatán
## Urban Sprawl Prediction (2025–2030)

Proyecto de modelado espacial que combina **Autómatas Celulares (CA)** con **Random Forest** para predecir la expansión de la mancha urbana de Mérida a 5 años.

---

## Arquitectura del modelo

```
Datos LANDSAT/Sentinel (GEE)
        ↓
Clasificación LULC (Land Use/Land Cover)
        ↓
Extracción de variables espaciales (features)
        ↓
Entrenamiento Random Forest (transición urbano/no-urbano)
        ↓
Simulación Autómata Celular (5 iteraciones anuales)
        ↓
Mapas de probabilidad 2026–2030 (.tif + .shp)
```

---

## Estructura del proyecto

```
urban_sprawl_merida/
├── config/
│   └── settings.py          # Parámetros globales (bbox, años, resolución)
├── data/
│   ├── raw/                 # Imágenes descargadas (LANDSAT, límites municipales)
│   ├── processed/           # Rasters clasificados, features calculados
│   └── output/              # Predicciones finales (.tif)
├── models/
│   └── rf_model.pkl         # Modelo entrenado serializado
├── scripts/
│   ├── 01_download_data.py  # Descarga GEE / USGS / INEGI
│   ├── 02_preprocess.py     # Clasificación LULC y extracción de features
│   ├── 03_train_model.py    # Entrenamiento Random Forest
│   ├── 04_simulate_ca.py    # Simulación Autómata Celular
│   └── 05_visualize.py      # Generación de mapas y reportes
├── notebooks/
│   └── exploratory.ipynb    # Análisis exploratorio
├── results/
│   ├── maps/                # Mapas GeoTIFF de predicción por año
│   └── reports/             # Métricas, figuras, estadísticas
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Clonar y crear entorno virtual
cd urban_sprawl_merida
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Autenticar Google Earth Engine (solo primera vez)
earthengine authenticate
```

---

## Pipeline de ejecución

```bash
# Paso 1: Descargar datos satelitales (LANDSAT 8/9 + límites INEGI)
python scripts/01_download_data.py

# Paso 2: Preprocesar imágenes y calcular variables espaciales
python scripts/02_preprocess.py

# Paso 3: Entrenar modelo Random Forest
python scripts/03_train_model.py

# Paso 4: Correr simulación de autómata celular 2025-2030
python scripts/04_simulate_ca.py

# Paso 5: Generar mapas y reporte final
python scripts/05_visualize.py
```

---

## Variables (features) del modelo

| Variable | Descripción | Fuente |
|----------|-------------|--------|
| `dist_road` | Distancia a carretera más cercana | INEGI Marco Geoestadístico |
| `dist_urban_edge` | Distancia al borde urbano actual | Calculado |
| `dist_center` | Distancia al centro histórico (Plaza Grande) | Calculado |
| `slope` | Pendiente del terreno (°) | DEM SRTM 30m |
| `ndvi_mean` | Vegetación media (últimos 3 años) | LANDSAT 8/9 |
| `ndbi_mean` | Índice de built-up | LANDSAT 8/9 |
| `pop_density` | Densidad poblacional AGEB | INEGI Censo 2020 |
| `neighbors_8` | Fracción de celdas urbanas en vecindad 8x8 | Calculado |

---

## Metodología

### 1. Clasificación LULC
Se usa LANDSAT 8/9 Collection 2 (2015, 2020, 2024) para clasificar cada pixel en:
- `1` = Urbano
- `0` = No urbano (vegetación, suelo desnudo, cuerpos de agua)

Usando índices NDVI, NDBI y EVI + clasificador supervisado.

### 2. Random Forest para transición
El modelo aprende **qué celdas no-urbanas pasaron a urbanas** entre pares de años (2015→2020, 2020→2024). La variable objetivo es binaria: ¿se urbanizó esta celda?

### 3. Autómata Celular
La probabilidad de RF se combina con reglas de vecindad espacial:
```
P_total(t) = α × P_RF + β × P_neighbors + γ × P_stochastic
```
Parámetros calibrables en `config/settings.py`.

### 4. Validación
- **Figura de mérito (FOM)**: métrica estándar para modelos de cambio de uso de suelo
- **Kappa de Cohen**: acuerdo espacial
- **Curva ROC / AUC**

---

## Datos requeridos y fuentes

| Dataset | Fuente | Acceso |
|---------|--------|--------|
| LANDSAT 8/9 Collection 2 | USGS / Google Earth Engine | Gratuito (cuenta GEE) |
| Límites municipales Mérida | INEGI Marco Geoestadístico 2023 | Gratuito |
| Red vial nacional | INEGI | Gratuito |
| DEM (elevación) | SRTM 30m via GEE | Gratuito |
| Densidad de población | INEGI Censo 2020 | Gratuito |

---

## Resultados esperados

- `results/maps/prediction_2026.tif` — Mapa de probabilidad de urbanización
- `results/maps/prediction_2027.tif` ... hasta 2030
- `results/maps/urban_extent_2030.shp` — Shapefile del área urbana proyectada
- `results/reports/metrics.csv` — Métricas de validación
- `results/reports/area_statistics.csv` — Estadísticas de crecimiento por año
