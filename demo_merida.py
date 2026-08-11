"""
demo_merida.py
Demo completo del pipeline de predicción de expansión urbana.
Genera datos sintéticos que simulan la ZMM de Mérida y corre los 5 pasos.

No requiere Google Earth Engine, rasterio, ni archivos externos.
Solo necesita: numpy, scipy, scikit-learn, pandas, matplotlib, joblib

Ejecución:
    python demo_merida.py

Salidas en demo_output/:
    lulc_2015.npy, lulc_2020.npy, lulc_2024.npy  → clasificaciones históricas
    rf_model.pkl                                   → modelo entrenado
    prediction_{year}.npy                          → mapas de probabilidad
    urban_extent_{year}.npy                        → área urbana binaria
    area_statistics.csv                            → estadísticas de crecimiento
    feature_importance.png, roc_curve.png          → figuras de evaluación
    expansion_maps.png                             → visualización final
    final_report.txt                               → reporte ejecutivo
"""

import os
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import (
    distance_transform_edt, uniform_filter, binary_opening, label
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, RocCurveDisplay
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
GRID_SIZE       = 200          # píxeles por lado (200×200 = 40,000 celdas)
PIXEL_METERS    = 30           # 30m/píxel como LANDSAT
OUTPUT_DIR      = "demo_output"
TRAIN_YEARS     = [2015, 2020, 2024]
PREDICT_YEARS   = [2026, 2027, 2028, 2029, 2030]
BASE_YEAR       = 2024
GROWTH_RATE     = 0.035        # 3.5% anual (histórico ZMM)
rng             = np.random.default_rng(42)

FEATURE_NAMES = [
    "dist_urban_edge", "dist_center", "dist_road",
    "ndvi_mean", "neighbor_3x3", "neighbor_5x5", "neighbor_9x9"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PASO 1: GENERAR DATOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────

def make_urban_core(size, center_frac=0.5, radius_frac=0.12, seed_rng=None):
    """Crea un núcleo urbano circular (simula el centro histórico de Mérida)."""
    r = seed_rng or rng
    cx = int(size * center_frac)
    cy = int(size * center_frac)
    rows, cols = np.mgrid[0:size, 0:size]
    dist = np.sqrt((rows - cx)**2 + (cols - cy)**2)
    radius = size * radius_frac
    return (dist < radius).astype(np.uint8)


def grow_urban(urban, growth_rate, noise_scale=0.15, r=None):
    """
    Expande el área urbana para un año posterior.
    La probabilidad de expansión decae con la distancia al borde.
    """
    r = r or rng
    non_urban = (urban == 0).astype(np.float32)
    dist = distance_transform_edt(non_urban)
    # Probabilidad más alta cerca del borde urbano
    prob = np.exp(-dist / (dist.max() * 0.25 + 1e-9))
    prob *= non_urban
    # Añadir variabilidad espacial con ruido Perlin-like
    noise = r.random((urban.shape)) * noise_scale
    prob = prob * (1 - noise_scale) + noise * prob
    # Cuántas celdas convertir
    n_urban = urban.sum()
    n_convert = int(n_urban * growth_rate)
    # Seleccionar las de mayor probabilidad
    flat = prob.ravel()
    top_idx = np.argsort(flat)[-n_convert:]
    new_urban = urban.copy()
    new_urban.ravel()[top_idx] = 1
    return new_urban


def generate_road_network(size):
    """Simula la red vial principal de la ZMM (carreteras radiales + periférico)."""
    road = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    # Carreteras radiales
    for angle in [0, 45, 90, 135]:
        rad = np.deg2rad(angle)
        for d in range(size):
            r = int(c + d * np.sin(rad))
            col = int(c + d * np.cos(rad))
            r2 = int(c - d * np.sin(rad))
            col2 = int(c - d * np.cos(rad))
            for rr, cc in [(r, col), (r2, col2)]:
                if 0 <= rr < size and 0 <= cc < size:
                    road[rr, cc] = 1
                    # Dar grosor de 2px
                    if 0 <= rr+1 < size:
                        road[rr+1, cc] = 1
    # Periférico (anillo a ~35% del radio)
    radius_ring = int(size * 0.35)
    theta = np.linspace(0, 2*np.pi, 800)
    for t in theta:
        rr = int(c + radius_ring * np.sin(t))
        cc = int(c + radius_ring * np.cos(t))
        if 0 <= rr < size and 0 <= cc < size:
            road[rr, cc] = 1
    return road


def generate_synthetic_data():
    """Genera los mapas LULC históricos y la red vial sintéticos."""
    log.info("=" * 58)
    log.info("PASO 1: Generando datos sintéticos de Mérida")
    log.info("=" * 58)

    size = GRID_SIZE

    # Año base 2015 — núcleo pequeño
    urban_2015 = make_urban_core(size, radius_frac=0.12, seed_rng=rng)
    # Agregar colonias satélite (norte y sur)
    urban_2015 = grow_urban(urban_2015, growth_rate=0.08, r=rng)

    # 2020 — 5 años de crecimiento
    urban_2020 = urban_2015.copy()
    for _ in range(5):
        urban_2020 = grow_urban(urban_2020, growth_rate=GROWTH_RATE * 1.1, r=rng)

    # 2024 — 4 años adicionales
    urban_2024 = urban_2020.copy()
    for _ in range(4):
        urban_2024 = grow_urban(urban_2024, growth_rate=GROWTH_RATE, r=rng)

    # Red vial
    roads = generate_road_network(size)

    # Guardar
    for year, u in [(2015, urban_2015), (2020, urban_2020), (2024, urban_2024)]:
        np.save(os.path.join(OUTPUT_DIR, f"lulc_{year}.npy"), u)
        area = u.sum() * PIXEL_METERS**2 / 1e6
        log.info(f"  LULC {year}: {u.sum():,} px urbanos → {area:.1f} km²")

    np.save(os.path.join(OUTPUT_DIR, "roads.npy"), roads)
    log.info(f"  Red vial: {roads.sum():,} px de carretera\n")

    return {"2015": urban_2015, "2020": urban_2020, "2024": urban_2024, "roads": roads}


# ─────────────────────────────────────────────────────────────
# PASO 2: EXTRAER FEATURES
# ─────────────────────────────────────────────────────────────

def extract_features(urban, roads, center=None):
    """Calcula el stack de 7 features para una imagen dada."""
    size = urban.shape[0]
    if center is None:
        center = (size // 2, size // 2)

    # f0: distancia al borde urbano (normalizada)
    non_urban = (1 - urban).astype(np.float32)
    dist_edge = distance_transform_edt(non_urban) * PIXEL_METERS
    max_de = np.percentile(dist_edge[dist_edge > 0], 99) if dist_edge.max() > 0 else 1
    f0 = np.clip(dist_edge / max_de, 0, 1).astype(np.float32)

    # f1: distancia al centro histórico (normalizada)
    rows, cols = np.mgrid[0:size, 0:size]
    dist_center = np.sqrt((rows - center[0])**2 + (cols - center[1])**2) * PIXEL_METERS
    max_dc = dist_center.max()
    f1 = (dist_center / max_dc).astype(np.float32)

    # f2: distancia a carreteras (normalizada)
    dist_road = distance_transform_edt(1 - roads) * PIXEL_METERS
    max_dr = np.percentile(dist_road[dist_road > 0], 99) if dist_road.max() > 0 else 1
    f2 = np.clip(dist_road / max_dr, 0, 1).astype(np.float32)

    # f3: NDVI sintético (zonas no urbanas tienen más vegetación)
    ndvi_base = rng.uniform(0.1, 0.5, size=(size, size)).astype(np.float32)
    ndvi_base[urban == 1] *= 0.3   # zonas urbanas menos vegetadas
    f3 = ndvi_base

    # f4-f6: densidad de vecindad
    f4 = uniform_filter(urban.astype(np.float32), size=3)
    f5 = uniform_filter(urban.astype(np.float32), size=5)
    f6 = uniform_filter(urban.astype(np.float32), size=9)

    return np.stack([f0, f1, f2, f3, f4, f5, f6], axis=0)


def build_training_dataset(lulc_maps, roads):
    """Construye X, y para entrenamiento a partir de pares de años."""
    log.info("=" * 58)
    log.info("PASO 2: Extracción de features y dataset de entrenamiento")
    log.info("=" * 58)

    all_X, all_y = [], []
    year_pairs = [(2015, 2020), (2020, 2024)]

    for t0, t1 in year_pairs:
        urban_t0 = lulc_maps[str(t0)]
        urban_t1 = lulc_maps[str(t1)]
        features  = extract_features(urban_t0, roads)

        non_urban_mask = (urban_t0 == 0).ravel()
        y = urban_t1.ravel()[non_urban_mask].astype(np.uint8)
        X = features.reshape(7, -1).T[non_urban_mask]

        # Submuestreo para velocidad
        max_n = 30_000
        if len(y) > max_n:
            idx = rng.choice(len(y), size=max_n, replace=False)
            X, y = X[idx], y[idx]

        log.info(f"  Par {t0}→{t1}: {len(y):,} muestras | {y.mean()*100:.1f}% urbanización")
        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    log.info(f"  Dataset total: {len(y_all):,} muestras\n")
    return X_all, y_all


# ─────────────────────────────────────────────────────────────
# PASO 3: ENTRENAR RANDOM FOREST
# ─────────────────────────────────────────────────────────────

def figure_of_merit(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def train_and_evaluate(X, y):
    log.info("=" * 58)
    log.info("PASO 3: Entrenamiento y validación del modelo RF")
    log.info("=" * 58)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Validación cruzada rápida
    log.info("  Validación cruzada 5-fold...")
    model_cv = RandomForestClassifier(
        n_estimators=50, max_depth=15, class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    cv_scores = cross_val_score(
        model_cv, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1
    )
    log.info(f"  CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Modelo final
    log.info("  Entrenando modelo final (300 árboles)...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", oob_score=True, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    log.info(f"  ✓ Entrenado en {time.time()-t0:.1f}s | OOB: {model.oob_score_:.4f}")

    # Métricas
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc_roc":    round(roc_auc_score(y_test, y_proba), 4),
        "f1_score":   round(f1_score(y_test, y_pred), 4),
        "kappa":      round(cohen_kappa_score(y_test, y_pred), 4),
        "fom":        round(figure_of_merit(y_test, y_pred), 4),
        "oob_score":  round(model.oob_score_, 4),
        "cv_auc_mean":round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
    }

    log.info(f"  AUC-ROC:  {metrics['auc_roc']}")
    log.info(f"  F1 Score: {metrics['f1_score']}")
    log.info(f"  Kappa:    {metrics['kappa']}")
    log.info(f"  FOM:      {metrics['fom']}\n")

    # Guardar modelo
    model_path = os.path.join(OUTPUT_DIR, "rf_model.pkl")
    joblib.dump(model, model_path)

    # Importancia de features
    _plot_feature_importance(model)

    # Curva ROC
    _plot_roc_curve(model, X_test, y_test)

    # CSV métricas
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "metrics.csv"), index=False
    )

    return model, metrics


def _plot_feature_importance(model):
    importances = model.feature_importances_
    std = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E53935" if v > 0.18 else "#FF7043" if v > 0.12 else "#78909C"
              for v in importances[order]]
    ax.barh([FEATURE_NAMES[i] for i in order], importances[order],
            xerr=std[order], color=colors, capsize=3, edgecolor="white")
    ax.set_xlabel("Importancia (Gini impurity)", fontsize=10)
    ax.set_title("Importancia de Variables — Random Forest", fontsize=11, fontweight="bold")
    ax.axvline(1/len(FEATURE_NAMES), color="gray", linestyle="--", alpha=0.6,
               label="Importancia uniforme")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    log.info("  → feature_importance.png guardado")


def _plot_roc_curve(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                    color="#E53935", name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Curva ROC — Transición Urbana", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    log.info("  → roc_curve.png guardado\n")


# ─────────────────────────────────────────────────────────────
# PASO 4: SIMULACIÓN AUTÓMATA CELULAR
# ─────────────────────────────────────────────────────────────

def simulate_ca(model, urban_base, roads):
    log.info("=" * 58)
    log.info("PASO 4: Simulación Autómata Celular 2025–2030")
    log.info("=" * 58)

    current = urban_base.copy()
    urban_series = {BASE_YEAR: current.copy()}
    center = (GRID_SIZE // 2, GRID_SIZE // 2)

    for year in PREDICT_YEARS:
        # Actualizar features con estado urbano actual
        features = extract_features(current, roads, center)

        # P_RF — probabilidad del modelo para celdas no-urbanas
        non_urban_mask = (current == 0).ravel()
        X_pred = features.reshape(7, -1).T[non_urban_mask]
        p_rf_vals = model.predict_proba(X_pred)[:, 1]
        p_rf = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
        p_rf[non_urban_mask] = p_rf_vals
        p_rf = p_rf.reshape(GRID_SIZE, GRID_SIZE)

        # P_vecindad — densidad de vecindad
        p_nbr = uniform_filter(current.astype(np.float32), size=9)

        # P_total
        p_stoch = rng.random((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        p_total = 0.60 * p_rf + 0.30 * p_nbr + 0.10 * p_stoch
        p_total[current == 1] = 0  # ya es urbano

        # Convertir las N celdas de mayor probabilidad
        n_convert = int(current.sum() * GROWTH_RATE)
        flat = p_total.ravel()
        top_idx = np.argsort(flat)[-n_convert:]
        new_urban = current.copy()
        new_urban.ravel()[top_idx] = 1

        area_km2 = new_urban.sum() * PIXEL_METERS**2 / 1e6
        delta_km2 = (new_urban.sum() - current.sum()) * PIXEL_METERS**2 / 1e6
        log.info(f"  {year}: {area_km2:.1f} km² (+{delta_km2:.2f} km²)")

        # Guardar
        np.save(os.path.join(OUTPUT_DIR, f"prediction_{year}.npy"), p_total)
        np.save(os.path.join(OUTPUT_DIR, f"urban_extent_{year}.npy"), new_urban)

        urban_series[year] = new_urban.copy()
        current = new_urban

    log.info("")
    return urban_series


# ─────────────────────────────────────────────────────────────
# PASO 5: VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def visualize_results(lulc_maps, urban_series):
    log.info("=" * 58)
    log.info("PASO 5: Generando visualizaciones")
    log.info("=" * 58)

    _plot_expansion_maps(lulc_maps, urban_series)
    _plot_area_statistics(urban_series)
    generate_report(urban_series)


def _plot_expansion_maps(lulc_maps, urban_series):
    """Panel 3×3: mapas históricos + predicciones."""
    years_hist = [2015, 2020, 2024]
    years_pred = PREDICT_YEARS

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Expansión Urbana — Mérida, Yucatán\nHistórico y Proyección 2026–2030",
                 color="white", fontsize=14, fontweight="bold", y=1.02)

    # Fila 1: histórico + base
    for i, year in enumerate(years_hist):
        ax = axes[0, i]
        u = lulc_maps[str(year)]
        cmap = mcolors.ListedColormap(["#1A2634", "#E53935"])
        ax.imshow(u, cmap=cmap, vmin=0, vmax=1)
        area = u.sum() * PIXEL_METERS**2 / 1e6
        ax.set_title(f"{year}\n{area:.1f} km²", color="white", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#0D1117")

    # Cuarto panel fila 1: comparativa 2024 vs 2030
    ax = axes[0, 3]
    u24 = lulc_maps["2024"]
    u30 = urban_series.get(2030, u24)
    rgb = np.zeros((*u24.shape, 3))
    rgb[u24 == 1] = [0.8, 0.2, 0.2]      # rojo: urbano 2024
    rgb[(u30 == 1) & (u24 == 0)] = [1.0, 0.85, 0.0]  # amarillo: nuevo 2030
    ax.imshow(rgb)
    new_km2 = ((u30 - u24) > 0).sum() * PIXEL_METERS**2 / 1e6
    ax.set_title(f"Delta 2024→2030\n+{new_km2:.1f} km²", color="white", fontsize=10)
    legend_elems = [
        Patch(color=[0.8, 0.2, 0.2], label="Urbano 2024"),
        Patch(color=[1.0, 0.85, 0.0], label="Nueva expansión"),
    ]
    ax.legend(handles=legend_elems, loc="lower left",
              facecolor="#1C2833", edgecolor="#444", labelcolor="white", fontsize=7)
    ax.axis("off")

    # Fila 2: predicciones con mapa de probabilidad
    for i, year in enumerate(PREDICT_YEARS[:4]):
        ax = axes[1, i]
        prob = np.load(os.path.join(OUTPUT_DIR, f"prediction_{year}.npy"))
        u_base = lulc_maps["2024"]

        # Urbano base gris
        base_rgb = np.zeros((*u_base.shape, 4))
        base_rgb[u_base == 1] = [0.2, 0.2, 0.3, 0.8]
        ax.imshow(base_rgb)

        # Probabilidad
        cmap_prob = plt.cm.YlOrRd
        prob_masked = np.ma.masked_where(prob < 0.25, prob)
        im = ax.imshow(prob_masked, cmap=cmap_prob, vmin=0.25, vmax=1.0, alpha=0.85)

        area = urban_series[year].sum() * PIXEL_METERS**2 / 1e6
        ax.set_title(f"Prob. {year}\n{area:.1f} km²", color="white", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#0D1117")

    for ax in axes.ravel():
        ax.set_facecolor("#0D1117")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "expansion_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  → expansion_maps.png guardado")


def _plot_area_statistics(urban_series):
    years  = sorted(urban_series.keys())
    areas  = [urban_series[y].sum() * PIXEL_METERS**2 / 1e6 for y in years]
    growth = [0] + [areas[i] - areas[i-1] for i in range(1, len(areas))]

    df = pd.DataFrame({"year": years, "area_km2": areas, "growth_km2": growth})
    df.to_csv(os.path.join(OUTPUT_DIR, "area_statistics.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    fig.patch.set_facecolor("#0D1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#AAA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ax.spines.values(): s.set_edgecolor("#333")

    hist_y = [y for y in years if y <= BASE_YEAR]
    pred_y = [y for y in years if y > BASE_YEAR]
    hist_a = [areas[years.index(y)] for y in hist_y]
    pred_a = [areas[years.index(y)] for y in pred_y]

    ax1.fill_between(hist_y, hist_a, alpha=0.2, color="#42A5F5")
    ax1.fill_between(pred_y, pred_a, alpha=0.15, color="#FFA726")
    ax1.plot(hist_y, hist_a, "o-", color="#42A5F5", lw=2, ms=8, label="Histórico (simulado)")
    ax1.plot(pred_y, pred_a, "s--", color="#FFA726", lw=2, ms=8, label="Proyectado")
    ax1.axvline(BASE_YEAR, color="#555", ls=":", lw=1.5)
    ax1.set_ylabel("Área urbana (km²)", color="#CCC")
    ax1.set_title("Evolución del Área Urbana — Mérida ZMM (Demo)", color="white", fontsize=12, fontweight="bold")
    ax1.legend(facecolor="#1C2833", edgecolor="#444", labelcolor="white")
    ax1.yaxis.set_tick_params(labelcolor="#CCC")
    ax1.xaxis.set_tick_params(labelcolor="#CCC")

    colors = ["#42A5F5" if y <= BASE_YEAR else "#FFA726" for y in years[1:]]
    ax2.bar(years[1:], growth[1:], color=colors, edgecolor="none", alpha=0.85)
    ax2.set_ylabel("Crecimiento anual (km²)", color="#CCC")
    ax2.set_xlabel("Año", color="#CCC")
    ax2.set_title("Crecimiento Anual", color="white", fontsize=11)
    ax2.yaxis.set_tick_params(labelcolor="#CCC")
    ax2.xaxis.set_tick_params(labelcolor="#CCC")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "area_statistics.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  → area_statistics.png guardado")
    return df


def generate_report(urban_series):
    stats_path   = os.path.join(OUTPUT_DIR, "area_statistics.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    df_s = pd.read_csv(stats_path)
    df_m = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    area_base = df_s[df_s["year"] == BASE_YEAR]["area_km2"].values[0]
    area_2030 = df_s[df_s["year"] == 2030]["area_km2"].values[0]
    delta = area_2030 - area_base
    pct   = delta / area_base * 100

    report = f"""
╔══════════════════════════════════════════════════════════╗
║   PREDICCIÓN DE EXPANSIÓN URBANA — MÉRIDA, YUCATÁN      ║
║   Reporte de Demo (datos sintéticos)                     ║
╚══════════════════════════════════════════════════════════╝

RESUMEN
  Área urbana {BASE_YEAR}:  {area_base:.1f} km²
  Área urbana 2030:   {area_2030:.1f} km²
  Crecimiento total:  +{delta:.1f} km²  (+{pct:.1f}%)

MODELO RANDOM FOREST
"""
    if df_m is not None:
        m = df_m.iloc[0]
        report += f"""  AUC-ROC:      {m.get('auc_roc','N/A')}
  F1 Score:     {m.get('f1_score','N/A')}
  Kappa:        {m.get('kappa','N/A')}
  FOM:          {m.get('fom','N/A')}
  OOB Score:    {m.get('oob_score','N/A')}
  CV AUC (5k):  {m.get('cv_auc_mean','N/A')} ± {m.get('cv_auc_std','N/A')}
"""

    report += "\nESTADÍSTICAS DE ÁREA POR AÑO\n"
    report += f"  {'Año':<8}{'Área (km²)':<14}{'Crecimiento (km²)'}\n"
    report += "  " + "-"*40 + "\n"
    for r in df_s.itertuples():
        sign = "+" if r.growth_km2 > 0 else ""
        report += f"  {r.year:<8}{r.area_km2:<14.1f}{sign}{r.growth_km2:.2f}\n"

    report += """
ARCHIVOS GENERADOS EN demo_output/
  lulc_{2015,2020,2024}.npy     Mapas LULC históricos
  rf_model.pkl                  Modelo entrenado
  prediction_{year}.npy         Probabilidades (2026-2030)
  urban_extent_{year}.npy       Área urbana binaria
  feature_importance.png        Importancia de variables
  roc_curve.png                 Curva ROC
  expansion_maps.png            Mapa resumen del crecimiento
  area_statistics.png           Gráfica de evolución
  area_statistics.csv           Tabla de estadísticas
  metrics.csv                   Métricas de validación

NOTA: Este es un demo con datos sintéticos.
Con datos LANDSAT reales, ejecuta los scripts numerados (01-05).
"""

    out = os.path.join(OUTPUT_DIR, "final_report.txt")
    with open(out, "w") as f:
        f.write(report)
    print(report)
    log.info(f"  → final_report.txt guardado")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    log.info("▶ Demo: Predicción de Expansión Urbana — Mérida, Yucatán")
    log.info(f"  Grid: {GRID_SIZE}×{GRID_SIZE} px | Resolución: {PIXEL_METERS}m | "
             f"Output: {OUTPUT_DIR}/\n")

    # Pipeline completo
    data       = generate_synthetic_data()
    X, y       = build_training_dataset(data, data["roads"])
    model, _   = train_and_evaluate(X, y)
    urban_s    = simulate_ca(model, data["2024"], data["roads"])
    visualize_results(data, urban_s)

    log.info(f"\n✓ Demo completado en {time.time()-t_start:.1f}s")
    log.info(f"  Revisa los resultados en: {os.path.abspath(OUTPUT_DIR)}/")
