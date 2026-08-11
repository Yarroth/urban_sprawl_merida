"""
scripts/04_simulate_ca.py  —  v2.0
Simulación CA con tres escenarios usando reglas aprendidas y variables kársticas.

Escenarios:
  no_plan   — expansión libre, sin restricciones
  plan_trad — corredores verdes + exclusión de cenotes
  ia_optimo — reglas CA aprendidas + variables kársticas + multiobjetivo

Función de probabilidad v2.0:
  P_total = α·P_LightGBM + β·P_CA_aprendida + γ·P_kárstico + δ·rand
"""
import os, sys, logging, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import uniform_filter, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CA_CONFIG, PREDICT_YEARS, BASE_YEAR, PATHS, PIXEL_RESOLUTION, STUDY_AREA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
rng = np.random.default_rng(42)

try:
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    log.warning("rasterio/geopandas no disponibles — usando numpy nativo")


def load_array(path):
    if RASTERIO_OK and path.endswith(".tif") and os.path.exists(path):
        import rasterio
        with rasterio.open(path) as src:
            return src.read(), src.profile.copy()
    elif path.endswith(".npy") and os.path.exists(path):
        return np.load(path)[np.newaxis,...], {}
    return None, None


def save_array(arr, path, profile):
    if RASTERIO_OK and path.endswith(".tif") and profile:
        import rasterio
        p = profile.copy()
        if arr.ndim == 2: arr = arr[np.newaxis,...]
        p.update(count=arr.shape[0], compress="lzw")
        with rasterio.open(path, "w", **p) as dst:
            dst.write(arr)
    else:
        np.save(path.replace(".tif",".npy"), arr)


def build_karst_prob(urban, lst_map, karst_vuln, cenote_dist, cfg):
    """
    P_kárstico: penaliza celdas con alta LST, cercanas a cenotes o con alta
    vulnerabilidad de acuífero. Mayor P_kárstico = MENOS probable de urbanizarse
    en el escenario ia_optimo.
    """
    # Normalizar LST: celdas calientes tienen mayor riesgo
    lst_norm = np.clip((lst_map - 25) / 15, 0, 1) if lst_map is not None else np.zeros_like(urban, dtype=np.float32)
    # Distancia a cenotes: cercanía = riesgo
    cenote_norm = np.clip(1 - cenote_dist / 2000, 0, 1) if cenote_dist is not None else np.zeros_like(urban, dtype=np.float32)
    # Vulnerabilidad kárstica directa
    karst_norm = karst_vuln if karst_vuln is not None else np.zeros_like(urban, dtype=np.float32)

    w = CA_CONFIG
    p_karst = (
        cfg.get("lst_weight", 0.35) * lst_norm +
        cfg.get("cenote_weight", 0.40) * cenote_norm +
        cfg.get("aquifer_weight", 0.25) * karst_norm
    )
    return p_karst.astype(np.float32)


def build_exclusion_mask(shape, profile, scenario, cenote_dist=None, lst_map=None, karst_vuln=None):
    mask = np.zeros(shape, dtype=bool)
    exclusions = CA_CONFIG["scenarios"][scenario]["exclusions"]

    if "cenotes" in exclusions and cenote_dist is not None:
        radius_px = int(200 / PIXEL_RESOLUTION)
        mask |= (cenote_dist < radius_px * PIXEL_RESOLUTION)

    if "lst_hotspots" in exclusions and lst_map is not None:
        mask |= (lst_map > 37.0)

    if "karst_alta" in exclusions and karst_vuln is not None:
        mask |= (karst_vuln > 0.7)

    return mask


def compute_ca_probability(ca_model, features, urban, p_lgbm):
    """Aplica el modelo de reglas CA aprendidas."""
    non_urban = (urban == 0).ravel()
    if features is None:
        return uniform_filter(urban.astype(np.float32), size=9)
    try:
        n_feat = features.shape[0]
        X = features.reshape(n_feat, -1).T[non_urban]
        p_lgbm_flat = p_lgbm.ravel()[non_urban]
        X_ca = np.column_stack([X, p_lgbm_flat])
        X_ca[np.isnan(X_ca)] = 0
        p_ca_vals = ca_model.predict_proba(X_ca)[:,1]
        p_ca = np.zeros(urban.size, dtype=np.float32)
        p_ca[non_urban] = p_ca_vals
        return p_ca.reshape(urban.shape)
    except Exception as e:
        log.warning(f"  CA-model falló ({e}), usando densidad de vecindad")
        return uniform_filter(urban.astype(np.float32), size=9)


def simulate_scenario(lgbm_model, ca_model, urban_base, features, profile,
                      lst_map, cenote_dist, karst_vuln, scenario):
    from config.settings import KARST_CONFIG
    log.info(f"\n  Escenario: {scenario.upper()}")
    sc_cfg = CA_CONFIG["scenarios"][scenario]
    alpha = sc_cfg["alpha"]; beta = sc_cfg["beta"]
    gamma = sc_cfg["gamma"]; delta = sc_cfg["delta"]

    current = urban_base.copy()
    urban_series = {BASE_YEAR: current.copy()}

    for year in PREDICT_YEARS:
        # Actualizar features dinámicas
        if features is not None:
            non_urban = (1 - current).astype(np.float32)
            dist_edge = distance_transform_edt(non_urban) * PIXEL_RESOLUTION
            max_de = np.percentile(dist_edge[dist_edge>0], 99) if dist_edge.max()>0 else 1
            features[0] = np.clip(dist_edge / max_de, 0, 1)
            features[4] = uniform_filter(current.astype(np.float32), size=3)
            features[5] = uniform_filter(current.astype(np.float32), size=5)
            features[6] = uniform_filter(current.astype(np.float32), size=9)

        # P_LightGBM
        non_urban_mask = (current == 0).ravel()
        if features is not None:
            X_pred = features.reshape(features.shape[0],-1).T[non_urban_mask]
            X_pred[np.isnan(X_pred)] = 0
            p_lgbm_vals = lgbm_model.predict_proba(X_pred)[:,1]
            p_lgbm = np.zeros(current.size, dtype=np.float32)
            p_lgbm[non_urban_mask] = p_lgbm_vals
            p_lgbm = p_lgbm.reshape(current.shape)
        else:
            p_lgbm = np.zeros_like(current, dtype=np.float32)

        # P_CA (reglas aprendidas o densidad de vecindad)
        p_ca = compute_ca_probability(ca_model, features, current, p_lgbm)

        # P_kárstico
        p_karst = build_karst_prob(current, lst_map, karst_vuln, cenote_dist, KARST_CONFIG)

        # P_total
        p_stoch = rng.random(current.shape, dtype=np.float32)
        p_total = alpha*p_lgbm + beta*p_ca + gamma*(1-p_karst) + delta*p_stoch
        p_total[current == 1] = 0

        # Exclusiones
        excl = build_exclusion_mask(current.shape, profile, scenario, cenote_dist, lst_map, karst_vuln)
        p_total[excl] = 0

        # Convertir N celdas
        n_convert = int(current.sum() * CA_CONFIG["annual_growth_rate"])
        n_convert = min(n_convert, max(0, int((current==0).sum()) - int(excl.sum())))
        flat = p_total.ravel()
        top_idx = np.argsort(flat)[-n_convert:]
        new_urban = current.copy()
        new_urban.ravel()[top_idx] = 1

        area = new_urban.sum() * PIXEL_RESOLUTION**2 / 1e6
        delta_km2 = (new_urban.sum() - current.sum()) * PIXEL_RESOLUTION**2 / 1e6
        log.info(f"    {year}: {area:.1f} km² (+{delta_km2:.2f} km²)")

        # Guardar
        pred_path = PATHS["prediction"].format(year=year, scenario=scenario)
        ext_path  = os.path.join(PATHS["results_maps"], f"urban_extent_{year}_{scenario}.npy")
        save_array(p_total, pred_path.replace(".tif",".npy"), profile)
        np.save(ext_path, new_urban)

        urban_series[year] = new_urban.copy()
        current = new_urban

    return urban_series


def compute_statistics(urban_series_all):
    rows = []
    for scenario, series in urban_series_all.items():
        for year, urban in sorted(series.items()):
            area = urban.sum() * PIXEL_RESOLUTION**2 / 1e6
            rows.append({"scenario": scenario, "year": year,
                         "area_km2": round(area, 2),
                         "area_ha":  round(area*100, 1)})
    df = pd.DataFrame(rows)
    df["growth_km2"] = df.groupby("scenario")["area_km2"].diff().fillna(0).round(2)
    df["growth_pct"]  = (df["growth_km2"] / df.groupby("scenario")["area_km2"].shift(1) * 100).fillna(0).round(2)
    df.to_csv(os.path.join(PATHS["results_reports"], "area_statistics.csv"), index=False)
    log.info("\nESTADÍSTICAS FINALES:")
    log.info(df[df["year"]==2030].to_string(index=False))
    return df


def main():
    log.info("=" * 58)
    log.info("PASO 4 v2.0: Simulación CA — tres escenarios")
    log.info("=" * 58)

    lgbm_model = joblib.load(PATHS["lgbm_model"])
    ca_model   = None
    if os.path.exists(PATHS["ca_model"]):
        ca_model = joblib.load(PATHS["ca_model"])

    # Cargar estado base
    base_path = PATHS["lulc_base"].format(year=BASE_YEAR)
    arr, profile = load_array(base_path)
    if arr is None:
        log.error(f"LULC base no encontrado: {base_path}")
        sys.exit(1)
    urban_base = arr[0].astype(np.uint8)
    log.info(f"Base {BASE_YEAR}: {urban_base.sum():,} px urbanos ({urban_base.sum()*PIXEL_RESOLUTION**2/1e6:.1f} km²)")

    # Cargar features
    feat_path = PATHS["features"].format(year=BASE_YEAR)
    feat_arr, _ = load_array(feat_path)
    features = feat_arr.astype(np.float32) if feat_arr is not None else None

    # Capas kársticas (sintéticas si no existen)
    lst_map, cenote_dist, karst_vuln = None, None, None
    lst_path = PATHS["lst"].format(year=BASE_YEAR)
    if os.path.exists(lst_path.replace(".tif",".npy")):
        lst_map = np.load(lst_path.replace(".tif",".npy"))
        log.info("  LST cargado")
    if os.path.exists(PATHS["cenotes"]):
        log.info("  Cenotes cargados")
    # Si no existen datos reales, generar sintéticos para demo
    if lst_map is None:
        lst_map = 32 + rng.uniform(0,5,urban_base.shape).astype(np.float32)
        lst_map[urban_base==1] += 2  # UHI
        log.info("  LST sintético generado para demo")
    if cenote_dist is None:
        cx, cy = urban_base.shape[0]//4, urban_base.shape[1]//4
        rows, cols = np.mgrid[0:urban_base.shape[0], 0:urban_base.shape[1]]
        cenote_dist = np.sqrt((rows-cx)**2+(cols-cy)**2).astype(np.float32)*PIXEL_RESOLUTION
        log.info("  Distancia a cenotes sintética para demo")
    if karst_vuln is None:
        karst_vuln = rng.uniform(0.1, 0.8, urban_base.shape).astype(np.float32)
        log.info("  Vulnerabilidad kárstica sintética para demo")

    # Simular tres escenarios
    all_series = {}
    for scenario in ["no_plan","plan_trad","ia_optimo"]:
        all_series[scenario] = simulate_scenario(
            lgbm_model, ca_model, urban_base, features.copy() if features is not None else None,
            profile, lst_map, cenote_dist, karst_vuln, scenario
        )

    compute_statistics(all_series)
    log.info("\n✓ Paso 4 completado — resultados en results/maps/")


if __name__ == "__main__":
    main()
