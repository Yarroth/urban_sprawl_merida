"""
scripts/08_indicadores_ambientales.py
Indicadores ambientales de la expansión por escenario (desde rasters reales).

Cruza los extents urbanos simulados (results/maps/urban_extent_*_{scenario}.tif)
con las features espaciales reales del pipeline (data/processed/features_2024.tif,
LST · NDVI · dist_cenote · vulnerabilidad kárstica) para medir DÓNDE aterriza la
expansión 2024→2030 en cada escenario:

  - lst_nuevas:      temperatura superficial (°C) media de las celdas nuevas
  - ndvi_nuevas:     NDVI medio de las celdas nuevas (conservación de vegetación)
  - dist_cenote:     distancia media (m) a cenotes de las celdas nuevas
  - karst_nuevas:    vulnerabilidad kárstica media de las celdas nuevas
  - verde_2030:      cobertura no urbana (%) del área de estudio en 2030

Se reportan las celdas NUEVAS (extent 2030 − urbano 2024) porque la LST media
del conjunto total mezcla el núcleo histórico y oculta el efecto de la ubicación
de la expansión (efecto de composición).

Salidas en results/reports/:
  - indicadores_ambientales.csv  → tabla por escenario
  - indicadores_ambientales.png  → gráfica comparativa
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PATHS  # noqa: E402

SCENARIOS = ["no_plan", "plan_trad", "ia_optimo"]
SC_LABEL = {"no_plan": "Sin planificación", "plan_trad": "Plan tradicional", "ia_optimo": "Gestión IA"}
SC_COLOR = {"no_plan": "#E53935", "plan_trad": "#43A047", "ia_optimo": "#7C4DFF"}

OUT_CSV = ROOT / "results" / "reports" / "indicadores_ambientales.csv"
OUT_PNG = ROOT / "results" / "reports" / "indicadores_ambientales.png"


def main():
    feat_path = ROOT / "data" / "processed" / "features_2024.tif"
    lulc_path = ROOT / "data" / "processed" / "lulc_2024.tif"
    if not feat_path.exists():
        print(f"[08] No existe {feat_path}; se omite (ejecuta primero 01→02).")
        return

    with rasterio.open(feat_path) as f:
        feat = f.read()  # (10, H, W): orden según training_dataset.parquet
    # Bandas según scripts/02_preprocess.extract_features (orden del parquet):
    # ndvi_mean=4º, lst_mean=8º, dist_cenote=9º, karst_vuln=10º (1-indexed)
    lst = feat[7]
    ndvi = feat[3]
    cenote = feat[8]
    karst = feat[9]
    with rasterio.open(lulc_path) as l:
        base = l.read(1) == 1  # urbano 2024

    rows = []
    for sc in SCENARIOS:
        ext = ROOT / "results" / "maps" / f"urban_extent_2030_{sc}.tif"
        if not ext.exists():
            print(f"[08] Falta {ext.name}; se omite {sc}.")
            continue
        with rasterio.open(ext) as e:
            urb = e.read(1) == 1
        nuevas = urb & ~base
        verde = 100.0 * (1.0 - urb.sum() / urb.size)
        rows.append({
            "escenario": sc,
            "celdas_nuevas": int(nuevas.sum()),
            "lst_nuevas": float(np.nanmean(lst[nuevas])) if nuevas.any() else np.nan,
            "ndvi_nuevas": float(np.nanmean(ndvi[nuevas])) if nuevas.any() else np.nan,
            "dist_cenote_m": float(np.nanmean(cenote[nuevas])) if nuevas.any() else np.nan,
            "karst_nuevas": float(np.nanmean(karst[nuevas])) if nuevas.any() else np.nan,
            "verde_2030_pct": float(verde),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[08] Sin datos; se omite.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[08] CSV guardado: {OUT_CSV}")
    print(df.round(2).to_string(index=False))

    # ── Gráfica comparativa (2 paneles) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = [SC_COLOR.get(s, "#888") for s in df["escenario"]]
    labels = [SC_LABEL.get(s, s) for s in df["escenario"]]

    # Panel 1: dónde aterriza la expansión (LST y karst de celdas nuevas)
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["lst_nuevas"], w, color=colors, label="LST nuevas (°C)", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Temperatura superficial media (°C)")
    ax.set_title("Expansión 2024→2030: celdas nuevas (ubicación real)", fontsize=11)
    for xi, v in zip(x, df["lst_nuevas"]):
        ax.text(xi - w / 2, v + 0.15, f"{v:.1f}°", ha="center", fontsize=9, color="#333")
    for xi, v in zip(x, df["karst_nuevas"]):
        ax.text(xi + w / 2, v * 24 + 23.5, f"kárst {v:.2f}", ha="center", fontsize=8, color="#7C4DFF")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, df["karst_nuevas"] * 24, w, color="#7C4DFF", alpha=0.35, label="Vulnerab. kárstica")
    ax2.set_ylim(20, 34)
    ax2.set_yticks([])
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Panel 2: cobertura verde y NDVI de celdas nuevas
    ax = axes[1]
    ax.bar(x, df["verde_2030_pct"], 0.45, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Cobertura no urbana 2030 (%)")
    ax.set_ylim(70, 85)
    for xi, v in zip(x, df["verde_2030_pct"]):
        ax.text(xi, v + 0.25, f"{v:.1f}%", ha="center", fontsize=9, color="#333")
    for xi, v in zip(x, df["ndvi_nuevas"]):
        ax.text(xi, 71.5, f"NDVI {v:.2f}", ha="center", fontsize=8, color="#1a6e3a")
    ax.set_title("Conservación de suelo no urbano 2030", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle("Indicadores ambientales de la expansión por escenario (rasters reales)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    print(f"[08] Gráfica guardada: {OUT_PNG}")


if __name__ == "__main__":
    main()
