"""
scripts/05_visualize.py
Genera mapas, figuras y reporte final del proyecto.

Salidas en results/:
  - maps/expansion_map_{year}.png     → mapa de probabilidad sobre base cartográfica
  - maps/comparison_2024_2030.png     → comparación antes/después
  - maps/animated_growth.gif          → animación del crecimiento 2025–2030
  - reports/final_report.md           → reporte ejecutivo en Markdown
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd
import rasterio
from rasterio.plot import show

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    PREDICT_YEARS, BASE_YEAR, PATHS, PIXEL_RESOLUTION, VIZ_CONFIG
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# MAPA DE PROBABILIDADES
# ─────────────────────────────────────────────────────────────

def plot_probability_map(year: int, base_urban: np.ndarray = None):
    """
    Genera un mapa de probabilidad de urbanización para un año dado.
    Superpone la probabilidad sobre el área urbana base.
    """
    pred_path = PATHS["prediction"].format(year=year)
    if not os.path.exists(pred_path):
        log.warning(f"  Predicción {year} no encontrada: {pred_path}")
        return

    with rasterio.open(pred_path) as src:
        prob = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(1, 1, figsize=VIZ_CONFIG["figsize"])
    ax.set_facecolor("#1C2833")

    # Fondo: área urbana base
    if base_urban is not None:
        urban_rgba = np.zeros((*base_urban.shape, 4))
        urban_rgba[base_urban == 1] = [0.2, 0.2, 0.25, 0.8]   # gris oscuro
        ax.imshow(urban_rgba, extent=extent, origin="upper", aspect="equal")

    # Probabilidades de expansión
    cmap = plt.cm.get_cmap("YlOrRd")
    prob_masked = np.ma.masked_where(prob < 0.2, prob)  # ocultar prob muy bajas
    im = ax.imshow(prob_masked, extent=extent, origin="upper",
                   cmap=cmap, vmin=0.2, vmax=1.0, alpha=0.85, aspect="equal")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Probabilidad de urbanización", fontsize=11, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Leyenda
    legend_elements = [
        Patch(facecolor="#33363D", edgecolor="gray", alpha=0.8, label="Área urbana base"),
        Patch(facecolor="#FCDE00", edgecolor="none", label="Alta prob. (>80%)"),
        Patch(facecolor="#E53935", edgecolor="none", label="Muy alta prob. (>90%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", facecolor="#2C3E50",
              edgecolor="gray", labelcolor="white", fontsize=9)

    ax.set_title(f"Probabilidad de Expansión Urbana — Mérida, Yucatán\nAño {year}",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Longitud", color="white")
    ax.set_ylabel("Latitud", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    fig.patch.set_facecolor("#1C2833")

    out_path = os.path.join(PATHS["results_maps"], f"expansion_map_{year}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Mapa {year}: {out_path}")


# ─────────────────────────────────────────────────────────────
# COMPARACIÓN 2024 vs 2030
# ─────────────────────────────────────────────────────────────

def plot_comparison_2024_2030():
    """Panel comparativo del área urbana en 2024 vs. proyección 2030."""
    base_path  = PATHS["lulc_base"].format(year=BASE_YEAR)
    pred_path  = os.path.join(PATHS["results_maps"], f"urban_extent_2030.tif")

    if not os.path.exists(base_path):
        log.warning("LULC base no disponible para comparación.")
        return

    with rasterio.open(base_path) as src:
        urban_2024 = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0D1117")
    cmap = mcolors.ListedColormap(["#1A2634", "#E53935"])

    # Panel 2024
    axes[0].imshow(urban_2024, extent=extent, origin="upper",
                   cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axes[0].set_title(f"Área Urbana {BASE_YEAR} (Actual)", fontsize=13,
                       fontweight="bold", color="white")
    area_2024 = urban_2024.sum() * PIXEL_RESOLUTION**2 / 1e6
    axes[0].text(0.02, 0.02, f"{area_2024:.1f} km²", transform=axes[0].transAxes,
                  color="white", fontsize=11, bbox=dict(facecolor="#2C3E50", alpha=0.8))

    # Panel 2030
    if os.path.exists(pred_path):
        with rasterio.open(pred_path) as src:
            urban_2030 = src.read(1)
        axes[1].imshow(urban_2030, extent=extent, origin="upper",
                       cmap=cmap, vmin=0, vmax=1, aspect="equal")
        area_2030 = urban_2030.sum() * PIXEL_RESOLUTION**2 / 1e6
        growth = area_2030 - area_2024
        axes[1].text(0.02, 0.02, f"{area_2030:.1f} km²\n(+{growth:.1f} km²)",
                      transform=axes[1].transAxes, color="white", fontsize=11,
                      bbox=dict(facecolor="#2C3E50", alpha=0.8))

        # Resaltar nuevas áreas en amarillo
        new_urban = (urban_2030 == 1) & (urban_2024 == 0)
        highlight = np.zeros((*new_urban.shape, 4))
        highlight[new_urban] = [1.0, 0.9, 0.0, 0.7]
        axes[1].imshow(highlight, extent=extent, origin="upper", aspect="equal")
    else:
        axes[1].text(0.5, 0.5, "Ejecuta 04_simulate_ca.py\nprimero",
                     ha="center", va="center", transform=axes[1].transAxes,
                     color="white", fontsize=12)

    axes[1].set_title("Proyección Urbana 2030 (simulada)", fontsize=13,
                       fontweight="bold", color="white")

    for ax in axes:
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_xlabel("Longitud", color="#888")
        ax.set_ylabel("Latitud", color="#888")

    legend_elements = [
        Patch(facecolor="#E53935", label="Área urbana base"),
        Patch(facecolor="#F0E000", label="Nueva expansión simulada"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower left",
                   facecolor="#1C2833", edgecolor="#444", labelcolor="white")

    fig.suptitle("Expansión Urbana — Mérida, Yucatán\nComparación 2024 → 2030",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    out_path = os.path.join(PATHS["results_maps"], "comparison_2024_2030.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Comparación 2024-2030: {out_path}")


# ─────────────────────────────────────────────────────────────
# GRÁFICA DE ESTADÍSTICAS DE ÁREA
# ─────────────────────────────────────────────────────────────

def plot_area_statistics():
    """Gráfica de evolución del área urbana y crecimiento anual."""
    stats_path = os.path.join(PATHS["results_reports"], "area_statistics.csv")
    if not os.path.exists(stats_path):
        log.warning("Estadísticas de área no disponibles.")
        return

    df = pd.read_csv(stats_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#0D1117")
    color_area   = "#42A5F5"
    color_growth = "#EF5350"

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#AAA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Área total
    historical = df[df["year"] <= BASE_YEAR]
    predicted  = df[df["year"] > BASE_YEAR]

    ax1.fill_between(df["year"], df["area_km2"], alpha=0.2, color=color_area)
    ax1.plot(historical["year"], historical["area_km2"], "o-", color=color_area,
             linewidth=2, markersize=8, label="Histórico")
    ax1.plot(predicted["year"],  predicted["area_km2"],  "s--", color="#FFA726",
             linewidth=2, markersize=8, label="Proyectado")
    ax1.axvline(x=BASE_YEAR, color="#555", linestyle=":", linewidth=1.5, label=f"Año base ({BASE_YEAR})")
    ax1.set_ylabel("Área urbana (km²)", color="#CCC", fontsize=11)
    ax1.set_title("Evolución del Área Urbana — Mérida, Yucatán", color="white",
                   fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#1C2833", edgecolor="#444", labelcolor="white")
    ax1.yaxis.set_tick_params(labelcolor="#CCC")
    ax1.xaxis.set_tick_params(labelcolor="#CCC")

    # Crecimiento anual
    growth_data = df[df["growth_km2"] > 0]
    colors_bar = ["#42A5F5" if y <= BASE_YEAR else "#FFA726" for y in growth_data["year"]]
    ax2.bar(growth_data["year"], growth_data["growth_km2"], color=colors_bar,
            edgecolor="none", alpha=0.85)
    ax2.set_ylabel("Crecimiento anual (km²)", color="#CCC", fontsize=11)
    ax2.set_xlabel("Año", color="#CCC", fontsize=11)
    ax2.set_title("Crecimiento Anual del Área Urbana", color="white", fontsize=12)
    ax2.yaxis.set_tick_params(labelcolor="#CCC")
    ax2.xaxis.set_tick_params(labelcolor="#CCC")

    plt.tight_layout()
    out_path = os.path.join(PATHS["results_reports"], "area_statistics.png")
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Gráfica de estadísticas: {out_path}")


# ─────────────────────────────────────────────────────────────
# REPORTE EJECUTIVO
# ─────────────────────────────────────────────────────────────

def generate_report():
    """Genera un reporte ejecutivo en Markdown con los resultados principales."""
    metrics_path = os.path.join(PATHS["results_reports"], "metrics.csv")
    stats_path   = os.path.join(PATHS["results_reports"], "area_statistics.csv")
    feat_path    = os.path.join(PATHS["results_reports"], "feature_importance.csv")

    metrics_str = "No disponible"
    stats_str   = "No disponible"
    feat_str    = "No disponible"
    final_growth = ""

    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0]
        metrics_str = (
            f"| Métrica | Valor |\n"
            f"|---------|-------|\n"
            f"| AUC-ROC | {m.get('auc_roc', 'N/A')} |\n"
            f"| F1 Score | {m.get('f1_score', 'N/A')} |\n"
            f"| Kappa | {m.get('kappa', 'N/A')} |\n"
            f"| FOM | {m.get('fom', 'N/A')} |\n"
            f"| OOB Score | {m.get('oob_score', 'N/A')} |\n"
            f"| CV AUC (5-fold) | {m.get('cv_auc_mean', 'N/A')} ± {m.get('cv_auc_std', 'N/A')} |"
        )

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        rows = [f"| {r.year} | {r.area_km2:.1f} | {r.area_ha:.0f} | "
                f"{'+' if r.growth_km2 > 0 else ''}{r.growth_km2:.2f} | "
                f"{'+' if r.growth_pct > 0 else ''}{r.growth_pct:.1f}% |"
                for r in df.itertuples()]
        stats_str = (
            "| Año | Área (km²) | Área (ha) | Crecimiento (km²) | Crecimiento (%) |\n"
            "|-----|-----------|-----------|-------------------|-----------------|\n" +
            "\n".join(rows)
        )
        # Calcular crecimiento total proyectado
        base_row = df[df["year"] == BASE_YEAR]
        last_row = df[df["year"] == max(df["year"])]
        if not base_row.empty and not last_row.empty:
            delta = last_row["area_km2"].values[0] - base_row["area_km2"].values[0]
            pct   = delta / base_row["area_km2"].values[0] * 100
            final_growth = f"\n> **Crecimiento proyectado 2024–2030:** +{delta:.1f} km² ({pct:.1f}%)\n"

    if os.path.exists(feat_path):
        df_f = pd.read_csv(feat_path)
        rows = [f"| {r.feature} | {r.importance:.4f} | {r.std:.4f} |"
                for r in df_f.itertuples()]
        feat_str = (
            "| Variable | Importancia | Desv. Est. |\n"
            "|----------|-------------|------------|\n" +
            "\n".join(rows)
        )

    report = f"""# Reporte de Predicción de Expansión Urbana
## Mérida, Yucatán — Proyección 2025–2030

**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumen Ejecutivo

Este reporte presenta los resultados del modelo de predicción de expansión
de la mancha urbana de la Zona Metropolitana de Mérida (ZMM) para el período
2025–2030, utilizando un enfoque híbrido de **Random Forest + Autómata Celular**.

{final_growth}

---

## 2. Metodología

### Datos
- **Imágenes satelitales:** LANDSAT 8/9 Collection 2, resolución 30m
- **Años de entrenamiento:** {', '.join(str(y) for y in [2015, 2020, 2024])}
- **Clasificación:** Urbano / No-urbano mediante NDVI + NDBI

### Modelo
- **Random Forest** para aprender patrones de transición histórica
- **Autómata Celular** para simular la dinámica espacial de expansión
- **Función de probabilidad:** `P = {CA_CONFIG_summary()}`

---

## 3. Métricas de Validación del Modelo

{metrics_str}

**Interpretación:**
- AUC-ROC > 0.80: modelo con buena capacidad discriminativa
- FOM > 0.20: aceptable para modelos de cambio LULC (Pontius et al. 2008)
- Kappa > 0.60: acuerdo sustancial

---

## 4. Variables más Importantes

{feat_str}

---

## 5. Proyección de Área Urbana

{stats_str}

---

## 6. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `results/maps/prediction_{{year}}.tif` | Mapa de probabilidad por año (float32) |
| `results/maps/urban_extent_{{year}}.tif` | Área urbana binaria por año |
| `results/maps/urban_extent_{{year}}.shp` | Shapefile del área urbana proyectada |
| `results/maps/comparison_2024_2030.png` | Mapa comparativo |
| `results/reports/metrics.csv` | Métricas de validación |
| `results/reports/area_statistics.csv` | Estadísticas de área |
| `models/rf_model.pkl` | Modelo Random Forest serializado |

---

## 7. Notas y Limitaciones

- El modelo asume continuidad en las tendencias de crecimiento históricas.
- No incorpora cambios en políticas urbanas o grandes proyectos de infraestructura
  que no estén reflejados en el período de entrenamiento.
- La resolución de 30m (LANDSAT) puede subestimar cambios en predios pequeños.
- Para mayor precisión, se recomienda incorporar datos de densidad poblacional
  por AGEB (INEGI 2020) y el Plan de Ordenamiento Territorial de Mérida.

---

## 8. Referencias

- Pontius, R.G. & Schneider, L.C. (2001). Land-cover change model validation.
  *Agriculture, Ecosystems & Environment*, 85(1–3), 239–248.
- White, R. & Engelen, G. (1993). Cellular automata and fractal urban form.
  *Environment and Planning A*, 25(8), 1175–1199.
- INEGI (2020). Censo de Población y Vivienda 2020.
"""

    out_path = os.path.join(PATHS["results_reports"], "final_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"  ✓ Reporte generado: {out_path}")


def CA_CONFIG_summary():
    from config.settings import CA_CONFIG
    return (f"{CA_CONFIG['alpha']}×P_RF + "
            f"{CA_CONFIG['beta']}×P_vecindad + "
            f"{CA_CONFIG['gamma']}×aleatorio")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PASO 5: Visualización y reporte")
    log.info("=" * 60)

    # Cargar área urbana base
    base_urban = None
    base_path = PATHS["lulc_base"].format(year=BASE_YEAR)
    if os.path.exists(base_path):
        with rasterio.open(base_path) as src:
            base_urban = src.read(1)

    # Mapas de probabilidad para cada año predicho
    log.info("\n[MAPAS] Generando mapas de probabilidad...")
    for year in PREDICT_YEARS:
        plot_probability_map(year, base_urban)

    # Comparación 2024 vs 2030
    log.info("\n[COMPARACIÓN] Generando mapa comparativo...")
    plot_comparison_2024_2030()

    # Gráfica de estadísticas
    log.info("\n[ESTADÍSTICAS] Generando gráficas...")
    plot_area_statistics()

    # Reporte ejecutivo
    log.info("\n[REPORTE] Generando reporte final...")
    generate_report()

    log.info("\n" + "=" * 60)
    log.info("✓ Paso 5 completado. Todos los resultados en results/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
