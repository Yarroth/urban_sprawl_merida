"""
scripts/06_calibrar_tasas.py  —  v2.0
Calibración empírica de las tasas de crecimiento por escenario del CA contra la
evolución histórica de la Zona Metropolitana de Mérida (ZMM) 1998–2020.

Qué hace:
  1. Reconstruye la TCMA (tasa de crecimiento media anual) de superficie urbana
     observada en los LULC del propio pipeline (data/processed/lulc_{year}.tif,
     clase 1 = urbano) si están disponibles.
  2. La contrasta con las referencias publicadas de la ZMM (IMEPLAN, Diagnóstico
     CitiesAdapt-GIZ, INEGI/EURE).
  3. Deriva las tasas por escenario con reglas documentadas:
       no_plan   = TCMA observada de dispersión (default 3.5%/año, IMEPLAN)
       plan_trad = TCMA observada con instrumentos tradicionales
                   (≈ 3.1%/año, Diagnóstico CitiesAdapt: 84.6% en 20 años)
       ia_optimo = convergencia a la TCMA poblacional de la ZMM
                   (≈ 2.3%/año, INEGI 2010-2020: 2.24%) → densificación
  4. Verifica los valores en CA_CONFIG y escribe results/reports/calibracion_tasas.md

Uso:
  python scripts/06_calibrar_tasas.py
"""
import os, sys, logging, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CA_CONFIG, TRAIN_YEARS, PATHS, PIXEL_RESOLUTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import numpy as np
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False


# ── Referencias históricas publicadas de la ZMM ─────────────────────────────
# Cada entrada: (indicador, t0, t1, área/ha o crecimiento, TCMA implícita, fuente)
REFERENCIAS = [
    {
        "indicador": "Superficie construida ZMM",
        "t0": 2000, "t1": 2020, "a0_ha": 21103, "a1_ha": 42186,
        "fuente": "IMEPLAN, vía Novedades Yucatán (2024)",
    },
    {
        "indicador": "Expansión urbana ZMM (Diagnóstico)",
        "t0": 2000, "t1": 2020, "a0_ha": None, "a1_ha": None,
        "crecimiento_total": 0.846,
        "fuente": "Diagnóstico Zona de Transición Mérida-Cuxtal, CitiesAdapt/GIZ (2025)",
    },
    {
        "indicador": "Mancha urbana (periodo previo, dispersión rápida)",
        "t0": 1998, "t1": 2010, "a0_ha": 15944, "a1_ha": 27027,
        "fuente": "López Santillán (2011) / Grupo R4 (2020)",
    },
    {
        "indicador": "Población ZMM (ancla de densificación)",
        "t0": 2010, "t1": 2020, "tcma_directa": 0.0224,
        "fuente": "INEGI Censo 2020, vía EURE 51(153) — Aguilar (2025)",
    },
]


def tcma(a0, a1, dt):
    """Tasa de crecimiento media anual: (a1/a0)^(1/dt) - 1."""
    if a0 is None or a1 is None or a0 <= 0 or dt <= 0:
        return None
    return (a1 / a0) ** (1.0 / dt) - 1.0


def area_urbana_km2(year):
    """Área urbana (clase 1) en km² del LULC del año dado, o None si no existe."""
    path = PATHS["lulc_base"].format(year=year)
    if not (RASTERIO_OK and os.path.exists(path)):
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    px_km2 = (PIXEL_RESOLUTION / 1000.0) ** 2
    return float((arr == 1).sum()) * px_km2


def main():
    # 1) TCMA observada en los LULC del pipeline (datos reales en producción)
    observado = {}
    areas = {}
    for y in TRAIN_YEARS:
        a = area_urbana_km2(y)
        areas[y] = a
        if a is not None:
            log.info("LULC %d: %.2f km² urbanos", y, a)
    if any(v is not None for v in areas.values()):
        anos = [y for y in TRAIN_YEARS if areas[y] is not None]
        for i in range(len(anos) - 1):
            t0, t1 = anos[i], anos[i + 1]
            observado[f"{t0}–{t1}"] = tcma(areas[t0], areas[t1], t1 - t0)
        t0, t1 = anos[0], anos[-1]
        observado[f"{t0}–{t1}"] = tcma(areas[t0], areas[t1], t1 - t0)

    # Rango plausible para la ZMM (2%–6% anual); fuera de él, se ignora el dato
    # (p. ej. con datos sintéticos la señal da ~11% y no debe calibrar nada).
    RANGO_PLAUSIBLE = (0.02, 0.06)
    obs_total = observado.get(f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}") if observado else None
    if obs_total is not None and not (RANGO_PLAUSIBLE[0] <= obs_total <= RANGO_PLAUSIBLE[1]):
        log.warning("TCMA observada %.1f%%/año fuera del rango plausible %.0f–%.0f%% — "
                    "se usan las referencias publicadas (datos sintéticos/no calibrables).",
                    obs_total * 100, RANGO_PLAUSIBLE[0] * 100, RANGO_PLAUSIBLE[1] * 100)
        obs_total = None

    # 2) Referencias publicadas → TCMA implícitas
    for r in REFERENCIAS:
        if "tcma_directa" in r:
            r["tcma"] = r["tcma_directa"]
        else:
            r["tcma"] = tcma(r["a0_ha"], r["a1_ha"], r["t1"] - r["t0"]) or \
                ((1 + r.get("crecimiento_total", 0)) ** (1.0 / (r["t1"] - r["t0"])) - 1.0)

    # 3) Derivación por escenario
    base = obs_total if obs_total is not None else 0.035   # dispersión observada
    derivadas = {
        "no_plan":   base,                                    # expansión libre = tendencia histórica
        "plan_trad": base * (0.0311 / 0.0353),                # instrumentos tradicionales (Diagnóstico)
        "ia_optimo": 0.023,                                   # densificación ≈ TCMA poblacional (2.24%)
    }

    # 4) Verificación contra CA_CONFIG
    cfg = CA_CONFIG["scenarios"]
    veredicto = []
    for esc, rate in derivadas.items():
        actual = cfg[esc].get("growth_rate")
        dif = abs(actual - rate)
        estado = "OK" if dif <= 0.002 else "AJUSTAR"
        if estado == "AJUSTAR":
            veredicto.append(f"  - {esc}: CA_CONFIG={actual*100:.1f}% vs calibrada={rate*100:.1f}% → {estado}")
        log.info("Escenario %-11s calibrada %5.1f%%/año | CA_CONFIG %5.1f%%/año | %s",
                 esc, rate * 100, (actual or 0) * 100, estado)

    # 5) Reporte
    out = Path(PATHS["results_reports"])
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Calibración empírica de tasas de crecimiento (ZMM 1998–2020)",
        "",
        "## TCMA observada en los LULC del pipeline",
        "",
    ]
    if areas:
        for y, a in areas.items():
            lines.append(f"- LULC {y}: {a:.2f} km² urbanos" if a is not None else f"- LULC {y}: no disponible")
        for k, v in observado.items():
            if v is not None:
                lines.append(f"- TCMA {k}: {v*100:.2f}%/año")
    else:
        lines.append("- No hay LULC en `data/processed/` (ejecutar 02_preprocess primero).")
    lines += ["", "## Referencias históricas publicadas", "", "| Indicador | Periodo | Área / crecimiento | TCMA | Fuente |",
              "|---|---|---|---|---|"]
    for r in REFERENCIAS:
        if "tcma_directa" in r:
            det = f"{r['tcma_directa']*100:.2f}%/año"
        elif r["a0_ha"]:
            det = f"{r['a0_ha']/100:.1f} → {r['a1_ha']/100:.1f} km²"
        else:
            det = f"+{r['crecimiento_total']*100:.1f}%"
        lines.append(f"| {r['indicador']} | {r['t0']}–{r['t1']} | {det} | {r['tcma']*100:.2f}%/año | {r['fuente']} |")
    lines += ["", "## Tasas por escenario (derivadas y en CA_CONFIG)", "", "| Escenario | Regla | Calibrada | CA_CONFIG | Estado |",
              "|---|---|---|---|---|"]
    for esc, rate in derivadas.items():
        actual = cfg[esc].get("growth_rate")
        estado = "OK" if abs(actual - rate) <= 0.002 else "AJUSTAR"
        regla = {
            "no_plan": "TCMA observada de dispersión",
            "plan_trad": "TCMA × (3.11/3.53) — instrumentos tradicionales",
            "ia_optimo": "TCMA poblacional ZMM (densificación)",
        }[esc]
        lines.append(f"| {esc} | {regla} | {rate*100:.1f}% | {actual*100:.1f}% | {estado} |")
    if veredicto:
        lines += ["", "## Acción requerida", ""] + veredicto
    lines += ["", "---", "Generado por `python scripts/06_calibrar_tasas.py`."]
    md = "\n".join(lines) + "\n"
    (out / "calibracion_tasas.md").write_text(md, encoding="utf-8")
    log.info("Reporte: %s", out / "calibracion_tasas.md")


if __name__ == "__main__":
    main()
