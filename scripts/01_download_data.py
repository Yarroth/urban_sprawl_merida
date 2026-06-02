"""
scripts/01_download_data.py
Descarga datos satelitales desde Google Earth Engine y datos vectoriales de INEGI.

Requiere:
- Cuenta de Google Earth Engine autenticada (`earthengine authenticate`)
- earthengine-api, geopandas, requests

Salidas en data/raw/:
- landsat_{year}.tif       → imagen LANDSAT multibanda por año
- merida_municipio.shp     → límite municipal
- red_vial_zmm.shp         → red vial de la ZMM
- dem_merida.tif           → modelo digital de elevación SRTM
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path

import numpy as np

# Agregar el directorio raíz al path para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import STUDY_AREA, GEE_CONFIG, TRAIN_YEARS, PATHS, PIXEL_RESOLUTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SECCIÓN 1: Google Earth Engine — Imágenes LANDSAT
# ─────────────────────────────────────────────────────────────

def init_gee():
    """Inicializa la API de Google Earth Engine."""
    try:
        import ee
        try:
            ee.Initialize(opt_url="https://earthengine.googleapis.com")
        except ee.EEException:
            ee.Authenticate()
            ee.Initialize()
        log.info("Google Earth Engine inicializado correctamente.")
        return ee
    except ImportError:
        log.error("earthengine-api no instalado. Ejecuta: pip install earthengine-api")
        sys.exit(1)


def get_landsat_composite(ee, year: int, bbox: dict, months: list) -> "ee.Image":
    """
    Genera un compuesto mediana de imágenes LANDSAT para un año dado,
    filtrado a temporada seca para minimizar nubosidad en Yucatán.
    """
    aoi = ee.Geometry.Rectangle([
        bbox["min_lon"], bbox["min_lat"],
        bbox["max_lon"], bbox["max_lat"]
    ])

    def apply_scale(image):
        """Aplica factores de escala de LANDSAT Collection 2."""
        optical = image.select("SR_B.").multiply(GEE_CONFIG["scale_factor"]).add(GEE_CONFIG["offset"])
        return image.addBands(optical, None, True).copyProperties(image, ["system:time_start"])

    def mask_clouds(image):
        """Enmascara nubes y sombras usando la banda QA_PIXEL."""
        qa = image.select("QA_PIXEL")
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)   # nubes
        shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # sombras de nube
        return image.updateMask(cloud_mask.And(shadow_mask))

    # Filtrar por año y meses de temporada seca
    date_filters = ee.Filter.Or(*[
        ee.Filter.date(f"{year}-{m:02d}-01", f"{year}-{m:02d}-28")
        for m in months
    ])

    collection = (
        ee.ImageCollection(GEE_CONFIG["collection"])
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUD_COVER", GEE_CONFIG["cloud_cover_max"]))
        .filter(date_filters)
        .map(mask_clouds)
        .map(apply_scale)
        .select(GEE_CONFIG["bands"])
        .median()
        .clip(aoi)
    )

    return collection


def compute_spectral_indices(ee, image) -> "ee.Image":
    """
    Calcula NDVI, NDBI y EVI como bandas adicionales.
    LANDSAT 9: B5=NIR, B4=Red, B6=SWIR1, B2=Blue
    """
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    evi  = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {"NIR": image.select("SR_B5"), "RED": image.select("SR_B4"),
         "BLUE": image.select("SR_B2")}
    ).rename("EVI")
    return image.addBands([ndvi, ndbi, evi])


def download_landsat(ee, year: int) -> str:
    """
    Descarga imagen LANDSAT de un año como GeoTIFF a data/raw/.
    Usa ee.batch.Export para imágenes grandes.
    """
    out_path = os.path.join(PATHS["raw"], f"landsat_{year}.tif")
    if os.path.exists(out_path):
        log.info(f"  landsat_{year}.tif ya existe, saltando descarga.")
        return out_path

    log.info(f"  Generando compuesto LANDSAT para {year}...")
    image = get_landsat_composite(
        ee, year,
        STUDY_AREA["bbox"],
        GEE_CONFIG["months_dry_season"]
    )
    image = compute_spectral_indices(ee, image)

    log.info(f"  Exportando a Drive (puede tardar varios minutos)...")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"landsat_merida_{year}",
        folder="urban_sprawl_merida",
        fileNamePrefix=f"landsat_{year}",
        region=ee.Geometry.Rectangle([
            STUDY_AREA["bbox"]["min_lon"], STUDY_AREA["bbox"]["min_lat"],
            STUDY_AREA["bbox"]["max_lon"], STUDY_AREA["bbox"]["max_lat"]
        ]),
        scale=PIXEL_RESOLUTION,
        crs=STUDY_AREA["crs"],
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()

    # Esperar a que termine (máx. 30 minutos)
    log.info(f"  Tarea GEE iniciada (ID: {task.id}). Esperando...")
    for _ in range(180):
        status = task.status()
        state = status["state"]
        if state == "COMPLETED":
            log.info(f"  ✓ Exportación de {year} completada.")
            break
        elif state in ("FAILED", "CANCELLED"):
            log.error(f"  ✗ Tarea fallida: {status.get('error_message')}")
            sys.exit(1)
        time.sleep(10)

    log.warning(f"  NOTA: Descarga manual requerida desde Google Drive → data/raw/landsat_{year}.tif")
    return out_path


# ─────────────────────────────────────────────────────────────
# SECCIÓN 2: DEM (SRTM via GEE)
# ─────────────────────────────────────────────────────────────

def download_dem(ee) -> str:
    """Descarga DEM SRTM 30m para el área de estudio."""
    out_path = os.path.join(PATHS["raw"], "dem_merida.tif")
    if os.path.exists(out_path):
        log.info("  dem_merida.tif ya existe, saltando.")
        return out_path

    log.info("  Descargando DEM SRTM 30m...")
    dem = ee.Image("USGS/SRTMGL1_003").clip(
        ee.Geometry.Rectangle([
            STUDY_AREA["bbox"]["min_lon"], STUDY_AREA["bbox"]["min_lat"],
            STUDY_AREA["bbox"]["max_lon"], STUDY_AREA["bbox"]["max_lat"]
        ])
    )
    slope = ee.Terrain.slope(dem)
    dem_slope = dem.rename("elevation").addBands(slope.rename("slope"))

    task = ee.batch.Export.image.toDrive(
        image=dem_slope,
        description="dem_merida",
        folder="urban_sprawl_merida",
        fileNamePrefix="dem_merida",
        region=ee.Geometry.Rectangle([
            STUDY_AREA["bbox"]["min_lon"], STUDY_AREA["bbox"]["min_lat"],
            STUDY_AREA["bbox"]["max_lon"], STUDY_AREA["bbox"]["max_lat"]
        ]),
        scale=30,
        crs=STUDY_AREA["crs"],
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info(f"  DEM exportado a Drive (ID: {task.id})")
    return out_path


# ─────────────────────────────────────────────────────────────
# SECCIÓN 3: Datos vectoriales INEGI
# ─────────────────────────────────────────────────────────────

def download_inegi_boundaries():
    """
    Descarga el límite municipal de Mérida desde el Marco Geoestadístico de INEGI.
    URL oficial del Marco Geoestadístico 2023.
    """
    import zipfile
    import geopandas as gpd

    out_path = os.path.join(PATHS["raw"], "merida_municipio.shp")
    if os.path.exists(out_path):
        log.info("  merida_municipio.shp ya existe, saltando.")
        return out_path

    log.info("  Descargando Marco Geoestadístico INEGI 2023 (Yucatán)...")

    # INEGI ofrece descarga por estado — clave Yucatán = 31
    url = "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469_s.zip"

    zip_path = os.path.join(PATHS["raw"], "marco_geo_yuc.zip")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info("  Descomprimiendo Marco Geoestadístico...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(PATHS["raw"])

        # Filtrar solo el municipio de Mérida (CVE_MUN = 050)
        shp_files = list(Path(PATHS["raw"]).rglob("*mun*.shp"))
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            merida = gdf[(gdf["CVE_ENT"] == "31") & (gdf["CVE_MUN"] == "050")]
            merida = merida.to_crs(STUDY_AREA["crs"])
            merida.to_file(out_path)
            log.info(f"  ✓ Límite municipal guardado: {out_path}")
        else:
            log.warning("  No se encontraron shapefiles. Descarga manual requerida.")
            _save_manual_instructions_boundary()
    except Exception as e:
        log.warning(f"  Descarga automática falló: {e}")
        _save_manual_instructions_boundary()

    return out_path


def _save_manual_instructions_boundary():
    """Guarda instrucciones para descarga manual si la automática falla."""
    instructions = """
DESCARGA MANUAL DE DATOS INEGI:
================================

1. Límite municipal de Mérida:
   URL: https://www.inegi.org.mx/app/mapas/
   - Seleccionar: Marco Geoestadístico → Estado: Yucatán → Municipio: Mérida
   - Descargar como shapefile y guardar en: data/raw/merida_municipio.shp

2. Red vial:
   URL: https://www.inegi.org.mx/temas/mapas/vial/
   - Descargar la capa de carreteras para Yucatán
   - Guardar en: data/raw/red_vial_zmm.shp

3. Datos censales (densidad de población por AGEB):
   URL: https://www.inegi.org.mx/programas/ccpv/2020/
   - Descargar: Resultados por localidad, Yucatán, municipio 050
   - Guardar CSV en: data/raw/densidad_poblacional_2020.csv
"""
    with open(os.path.join(PATHS["raw"], "instrucciones_descarga_manual.txt"), "w") as f:
        f.write(instructions)
    log.info("  Instrucciones de descarga manual guardadas en data/raw/instrucciones_descarga_manual.txt")


def download_roads(ee) -> str:
    """
    Descarga red vial desde OpenStreetMap via GEE (FeatureCollection) o INEGI.
    Alternativa: usar datos OSM con overpy.
    """
    out_path = os.path.join(PATHS["raw"], "red_vial_zmm.shp")
    if os.path.exists(out_path):
        log.info("  red_vial_zmm.shp ya existe, saltando.")
        return out_path

    log.info("  Descargando red vial (OSM via overpy)...")
    try:
        import overpy
        api = overpy.Overpass()
        bbox = STUDY_AREA["bbox"]

        # Query Overpass para carreteras principales
        query = f"""
        [out:json][timeout:120];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary"]
          ({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        );
        out geom;
        """
        result = api.query(query)

        import geopandas as gpd
        from shapely.geometry import LineString

        roads = []
        for way in result.ways:
            coords = [(node.lon, node.lat) for node in way.nodes]
            if len(coords) >= 2:
                roads.append({
                    "geometry": LineString(coords),
                    "highway": way.tags.get("highway", "unknown"),
                    "name": way.tags.get("name", ""),
                })

        gdf = gpd.GeoDataFrame(roads, crs="EPSG:4326").to_crs(STUDY_AREA["crs"])
        gdf.to_file(out_path)
        log.info(f"  ✓ Red vial guardada: {len(gdf)} segmentos → {out_path}")

    except ImportError:
        log.warning("  overpy no instalado. Ejecuta: pip install overpy")
        _save_manual_instructions_boundary()
    except Exception as e:
        log.warning(f"  Descarga de red vial falló: {e}. Descarga manual requerida.")

    return out_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PASO 1: Descarga de datos")
    log.info("=" * 60)

    # Crear directorios si no existen
    for path in PATHS.values():
        if not path.endswith((".tif", ".shp", ".pkl", ".csv")):
            os.makedirs(path, exist_ok=True)

    # Inicializar GEE
    log.info("\n[GEE] Inicializando Google Earth Engine...")
    ee = init_gee()

    # Descargar imágenes LANDSAT para cada año de entrenamiento
    log.info(f"\n[LANDSAT] Descargando imágenes para años: {TRAIN_YEARS}")
    for year in TRAIN_YEARS:
        log.info(f"  → Año {year}")
        download_landsat(ee, year)

    # Descargar DEM
    log.info("\n[DEM] Descargando modelo digital de elevación...")
    download_dem(ee)

    # Descargar datos INEGI
    log.info("\n[INEGI] Descargando datos vectoriales...")
    download_inegi_boundaries()
    download_roads(ee)

    log.info("\n" + "=" * 60)
    log.info("✓ Paso 1 completado.")
    log.info("NOTA: Los archivos exportados a Google Drive deben descargarse")
    log.info("      manualmente y copiarse a data/raw/ antes de continuar.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
