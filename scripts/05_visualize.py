"""
scripts/05_visualize.py
Genera mapas, figuras y reporte final del proyecto.

Salidas en results/:
  - maps/expansion_map_{year}_{scenario}.png     → mapa de probabilidad por año y escenario
  - maps/comparison_2024_2030_{scenario}.png     → comparación antes/después por escenario
  - reports/area_statistics.png                  → evolución del área por escenario
  - reports/final_report.md                      → reporte ejecutivo en Markdown
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
    PREDICT_YEARS, BASE_YEAR, PATHS, PIXEL_RESOLUTION, VIZ_CONFIG, CA_CONFIG
)

SCENARIOS = list(CA_CONFIG["scenarios"].keys())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# MAPA DE PROBABILIDADES
# ─────────────────────────────────────────────────────────────

def plot_probability_map(year: int, scenario: str, base_urban: np.ndarray = None):
    """
    Genera un mapa de probabilidad de urbanización para un año y escenario.
    Superpone la probabilidad sobre el área urbana base.
    """
    pred_path = PATHS["prediction"].format(year=year, scenario=scenario)
    if not os.path.exists(pred_path):
        log.warning(f"  Predicción {year}/{scenario} no encontrada: {pred_path}")
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

    ax.set_title(f"Probabilidad de Expansión Urbana — Mérida, Yucatán\nAño {year} · Escenario: {scenario}",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Longitud", color="white")
    ax.set_ylabel("Latitud", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    fig.patch.set_facecolor("#1C2833")

    out_path = os.path.join(PATHS["results_maps"], f"expansion_map_{year}_{scenario}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Mapa {year}/{scenario}: {out_path}")


# ─────────────────────────────────────────────────────────────
# COMPARACIÓN 2024 vs 2030
# ─────────────────────────────────────────────────────────────

def plot_comparison_2024_2030(scenario: str):
    """Panel comparativo del área urbana en 2024 vs. proyección 2030 (por escenario)."""
    base_path  = PATHS["lulc_base"].format(year=BASE_YEAR)
    pred_path  = PATHS["urban_extent"].format(year=2030, scenario=scenario)

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

    axes[1].set_title(f"Proyección Urbana 2030 — {scenario}", fontsize=13,
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

    fig.suptitle(f"Expansión Urbana — Mérida, Yucatán\nComparación 2024 → 2030 · Escenario: {scenario}",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    out_path = os.path.join(PATHS["results_maps"], f"comparison_2024_2030_{scenario}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Comparación 2024-2030 ({scenario}): {out_path}")


# ─────────────────────────────────────────────────────────────
# GRÁFICA DE ESTADÍSTICAS DE ÁREA
# ─────────────────────────────────────────────────────────────

def plot_area_statistics():
    """Gráfica de evolución del área urbana y crecimiento anual por escenario."""
    stats_path = os.path.join(PATHS["results_reports"], "area_statistics.csv")
    if not os.path.exists(stats_path):
        log.warning("Estadísticas de área no disponibles.")
        return

    df = pd.read_csv(stats_path)
    scenarios = list(df["scenario"].unique())
    colors = VIZ_CONFIG["scenario_colors"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    fig.patch.set_facecolor("#0D1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#AAA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.xaxis.set_tick_params(labelcolor="#CCC")
        ax.yaxis.set_tick_params(labelcolor="#CCC")

    # Área total por escenario
    for scenario in scenarios:
        grp = df[df["scenario"] == scenario].sort_values("year")
        color = colors.get(scenario, "#42A5F5")
        hist = grp[grp["year"] <= BASE_YEAR]
        pred = grp[grp["year"] > BASE_YEAR]
        ax1.fill_between(grp["year"], grp["area_km2"], alpha=0.12, color=color)
        ax1.plot(hist["year"], hist["area_km2"], "o-", color=color,
                 linewidth=2, markersize=7, label=f"{scenario} (histórico)")
        ax1.plot(pred["year"], pred["area_km2"], "s--", color=color,
                 linewidth=2, markersize=7, label=f"{scenario} (proyectado)")
    ax1.axvline(x=BASE_YEAR, color="#555", linestyle=":", linewidth=1.5,
                label=f"Año base ({BASE_YEAR})")
    ax1.set_ylabel("Área urbana (km²)", color="#CCC", fontsize=11)
    ax1.set_title("Evolución del Área Urbana por Escenario — Mérida, Yucatán",
                  color="white", fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#1C2833", edgecolor="#444", labelcolor="white", fontsize=9)

    # Crecimiento anual por escenario (barras agrupadas)
    years = sorted(df["year"].unique())
    n_sc = len(scenarios)
    width = 0.8 / max(n_sc, 1)
    x = np.arange(len(years))
    for i, scenario in enumerate(scenarios):
        grp = df[df["scenario"] == scenario].sort_values("year")
        color = colors.get(scenario, "#888")
        ax2.bar(x + (i - (n_sc - 1) / 2) * width, grp["growth_km2"],
                width=width, label=scenario, color=color, alpha=0.85, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, color="#CCC")
    ax2.set_ylabel("Crecimiento anual (km²)", color="#CCC", fontsize=11)
    ax2.set_xlabel("Año", color="#CCC", fontsize=11)
    ax2.set_title("Crecimiento Anual por Escenario", color="white", fontsize=12)
    ax2.legend(facecolor="#1C2833", edgecolor="#444", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(PATHS["results_reports"], "area_statistics.png")
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"], facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  ✓ Gráfica de estadísticas: {out_path}")


# ─────────────────────────────────────────────────────────────
# REPORTE EJECUTIVO
# ─────────────────────────────────────────────────────────────

def generate_report():
    """Genera un reporte ejecutivo en Markdown con los resultados v2.0."""
    metrics_path = os.path.join(PATHS["results_reports"], "metrics.csv")
    stats_path   = os.path.join(PATHS["results_reports"], "area_statistics.csv")

    metrics_str = "No disponible"
    stats_str   = "No disponible"
    growth_summary = ""

    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0]
        metrics_str = (
            "| Métrica | Valor |\n"
            "|---------|-------|\n"
            f"| AUC-ROC (LightGBM transición) | {m.get('auc_roc', 'N/A')} |\n"
            f"| F1 Score (transición) | {m.get('f1_score', 'N/A')} |\n"
            f"| Kappa (transición) | {m.get('kappa', 'N/A')} |\n"
            f"| FOM (transición) | {m.get('fom', 'N/A')} |\n"
            f"| CV AUC 5-fold | {m.get('cv_auc_mean', 'N/A')} ± {m.get('cv_auc_std', 'N/A')} |\n"
            f"| AUC-ROC (reglas CA aprendidas) | {m.get('ca_auc_roc', 'N/A')} |\n"
            f"| FOM (reglas CA aprendidas) | {m.get('ca_fom', 'N/A')} |"
        )

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        blocks = []
        growth_rows = []
        for scenario in SCENARIOS:
            grp = df[df["scenario"] == scenario].sort_values("year")
            if grp.empty:
                continue
            rows = [
                f"| {int(r.year)} | {r.area_km2:.1f} | {r.area_ha:.0f} | "
                f"{'+' if r.growth_km2 > 0 else ''}{r.growth_km2:.2f} | "
                f"{'+' if r.growth_pct > 0 else ''}{r.growth_pct:.1f}% |"
                for r in grp.itertuples()
            ]
            base_row = grp[grp["year"] == BASE_YEAR]
            last_row = grp[grp["year"] == grp["year"].max()]
            if not base_row.empty and not last_row.empty:
                delta = last_row["area_km2"].values[0] - base_row["area_km2"].values[0]
                pct = delta / base_row["area_km2"].values[0] * 100
                growth_rows.append(
                    f"- **{scenario}:** +{delta:.1f} km² ({pct:.1f}%) "
                    f"entre {BASE_YEAR} y {int(grp['year'].max())}"
                )
            blocks.append(
                f"### Escenario: {scenario}\n\n"
                "| Año | Área (km²) | Área (ha) | Crecimiento (km²) | Crecimiento (%) |\n"
                "|-----|-----------|-----------|-------------------|-----------------|\n"
                + "\n".join(rows)
            )
        stats_str = "\n\n".join(blocks)
        if growth_rows:
            growth_summary = (
                "\n> **Crecimiento proyectado por escenario:**\n"
                + "\n".join(growth_rows)
                + "\n"
            )

    report = f"""# Reporte de Predicción de Expansión Urbana
## Mérida, Yucatán — Proyección 2025–2030 (v2.0)

**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumen Ejecutivo

Este reporte presenta los resultados del sistema de predicción de expansión
de la mancha urbana de la Zona Metropolitana de Mérida (ZMM) para 2025–2030,
combinando **LightGBM + Autómata Celular de reglas aprendidas + variables
kársticas** (LST, cenotes, vulnerabilidad del acuífero). Se simulan tres
escenarios: `no_plan`, `plan_trad` e `ia_optimo`.

{growth_summary}

---

## 2. Metodología

### Datos
- **Imágenes satelitales:** LANDSAT 8/9 Collection 2, resolución 30m
- **Años de entrenamiento:** 2015, 2020, 2024
- **Clasificación:** Urbano / No-urbano mediante NDVI + NDBI

### Modelo
- **LightGBM** de transición urbana (gradient boosting con escala balanceada)
- **LightGBM de reglas CA aprendidas** sobre el estado de vecindad
  (P_LightGBM calculada out-of-fold para evitar fuga de datos)
- **Variables kársticas** como features y restricciones (LST, cenotes, acuífero)
- **Función de probabilidad:** {CA_CONFIG_summary()}

---

## 3. Métricas de Validación del Modelo

{metrics_str}

**Interpretación:**
- AUC-ROC > 0.80: modelo con buena capacidad discriminativa
- FOM > 0.20: aceptable para modelos de cambio LULC (Pontius et al. 2008)
- Kappa > 0.60: acuerdo sustancial

---

## 4. Proyección de Área Urbana por Escenario

{stats_str}

---

## 5. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `results/maps/prediction_{{year}}_{{scenario}}.tif` | Mapa de probabilidad (float32) por año y escenario |
| `results/maps/urban_extent_{{year}}_{{scenario}}.tif` | Área urbana binaria por año y escenario |
| `results/maps/expansion_map_{{year}}_{{scenario}}.png` | Mapa de probabilidad renderizado |
| `results/maps/comparison_2024_2030_{{scenario}}.png` | Comparación base vs. 2030 por escenario |
| `results/reports/metrics.csv` | Métricas de validación de ambos modelos |
| `results/reports/area_statistics.csv` | Estadísticas de área por año y escenario |
| `models/lgbm_model.pkl` | LightGBM de transición serializado |
| `models/ca_rules_model.pkl` | LightGBM de reglas CA serializado |

---

## 6. Notas y Limitaciones

- El modelo asume continuidad en las tendencias de crecimiento históricas.
- Las capas kársticas dependen de datos reales (SEDUMA, CONAGUA/IMTA); si no
  están disponibles, el pipeline usa capas sintéticas y los escenarios con
  gestión no son representativos.
- La resolución de 30m (LANDSAT) puede subestimar cambios en predios pequeños.
- Para mayor precisión, se recomienda incorporar densidad poblacional por AGEB
  (INEGI 2020) y el Plan de Ordenamiento Territorial de Mérida.

---

## 7. Referencias

- Ke, G. et al. (2017). LightGBM. NIPS 2017.
- Pontius, R.G. & Schneider, L.C. (2001). Land-cover change model validation.
  *Agriculture, Ecosystems & Environment*, 85(1–3), 239–248.
- White, R. & Engelen, G. (1993). Cellular automata and fractal urban form.
  *Environment and Planning A*, 25(8), 1175–1199.
- INEGI (2020). Censo de Población y Vivienda 2020.
- SEDUMA (2023). Registro de cenotes y sistemas kársticos de Yucatán.
"""

    out_path = os.path.join(PATHS["results_reports"], "final_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"  ✓ Reporte generado: {out_path}")


def CA_CONFIG_summary():
    from config.settings import CA_CONFIG
    parts = []
    for name, cfg in CA_CONFIG["scenarios"].items():
        parts.append(
            f"`{name}`: α={cfg['alpha']}·P_LightGBM + β={cfg['beta']}·P_CA + "
            f"γ={cfg['gamma']}·P_kárstico + δ={cfg['delta']}·aleatorio"
        )
    return "; ".join(parts)


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

    # Mapas de probabilidad para cada año y escenario
    log.info("\n[MAPAS] Generando mapas de probabilidad...")
    for scenario in SCENARIOS:
        for year in PREDICT_YEARS:
            plot_probability_map(year, scenario, base_urban)

    # Comparación 2024 vs 2030 por escenario
    log.info("\n[COMPARACIÓN] Generando mapas comparativos...")
    for scenario in SCENARIOS:
        plot_comparison_2024_2030(scenario)

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
