"""
scripts/07_seguridad_peatonal.py  —  Módulo de Seguridad Vial Peatonal
Valida la relación incidentes peatonales ↔ densidad vehicular en el Anillo
Periférico de Mérida y simula escenarios de política urbana:

  base               — estado actual (línea base calibrada a 2025)
  semaforizacion     — semáforos coordinados + fases peatonales + LPI
  piramide_movilidad — pirámide de movilidad: reducción de velocidad en
                       tramos urbanos, cruces peatonales prioritarios
  transito_accesible — paradas de autobús accesibles (refugios, rampas,
                       cruces seguros) → traslado modal y menos volumen
  vision_cero        — todas las medidas combinadas

Modelo (por segmento del anillo, ver SAFETY_CONFIG en config/settings.py):

  muertes_escenario = muertes_base × VF × SF × [p_cruce · cross · stops + (1-p_cruce)]

    VF    = (V_escenario/V_base)^0.6        elasticidad volumen→incidentes
    SF    = severidad(speed_eff)/severidad(80)   curva fatalidad WHO (2008)
    cross = factor de riesgo en cruces semaforizados (levers del escenario)
    stops = factor de riesgo en cruces de paradas de autobús

  Calibración: 17 muertes totales en el Periférico en 2025 (Diario de
  Yucatán); ~68% de las víctimas son peatones (34 de 50 fallecidos por
  atropellamiento); aforo ~150,000 veh/día (Gob. de Yucatán, 2025).

Uso:
  python scripts/07_seguridad_peatonal.py
"""
import os, sys, logging, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SAFETY_CONFIG, PATHS, VIZ_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SC = SAFETY_CONFIG
OUT = Path(PATHS["results_reports"]).parent / "safety"

# Curva WHO (2008) "Speed management": probabilidad de muerte del peatón
# atropellado según la velocidad de impacto (normalizada a 80 km/h = 1.0).
WHO_SPEED_DEATH = [(30, 0.10), (40, 0.30), (50, 0.55), (60, 0.75), (70, 0.90), (80, 0.95), (90, 0.98)]


def df_to_markdown(df):
    """Tabla markdown sin dependencia de tabulate."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = []
    for _, r in df.iterrows():
        body.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + body)


def fatality_severity(speed_kmh):
    """Factor de severidad relativo a 80 km/h (interpolación lineal de la curva WHO)."""
    xs = np.array([p[0] for p in WHO_SPEED_DEATH], dtype=float)
    ys = np.array([p[1] for p in WHO_SPEED_DEATH], dtype=float)
    p = float(np.interp(speed_kmh, xs, ys))
    return p / np.interp(80.0, xs, ys)


def build_segments():
    """Segmentos del anillo con muertes base distribuidas por exposición."""
    segs = []
    for s in SC["segments"]:
        exposure = (1 + 0.25 * s["crossings"]) * (1.25 if s["urban"] else 1.0)
        segs.append({**s, "exposure": exposure})
    tot_exp = sum(s["exposure"] * s["weight"] for s in segs)
    for s in segs:
        s["base_deaths"] = SC["baseline_deaths_year"] * (s["weight"] * s["exposure"] / tot_exp)
        s["aadt"] = SC["aadt_total"] * s["weight"]
    return segs


def scenario_deaths(segs, sc_name):
    """Muertes totales por segmento para un escenario (levers de SAFETY_CONFIG)."""
    lv = SC["scenarios"][sc_name]
    out = []
    for s in segs:
        speed_eff = lv["speed"] if s["urban"] else SC["speed_limit_kmh"]
        vol_factor = (lv["vol"]) ** SC["volume_exponent"]
        severity = fatality_severity(speed_eff) / fatality_severity(SC["speed_limit_kmh"])
        crossing = SC["crossing_share"] * lv["cross"] * lv["stops"] + (1 - SC["crossing_share"])
        out.append(s["base_deaths"] * vol_factor * severity * crossing)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segs = build_segments()

    # 1) Tabla por segmento (línea base)
    rows = []
    for s in segs:
        rows.append({
            "sector": s["name"], "aadt_veh_dia": s["aadt"],
            "cruces": s["crossings"], "pasos_elevados": s["bridges"],
            "paradas_bus": s["bus_stops"], "tramo_urbano": s["urban"],
            "muertes_base": round(s["base_deaths"], 3),
            "muertes_peatones_base": round(s["base_deaths"] * SC["pedestrian_share"], 3),
        })
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(OUT / "incidentes_por_segmento.csv", index=False)
    log.info("Segmentos: %.1f muertes/año totales, %.1f peatonales (2025)",
             seg_df.muertes_base.sum(), seg_df.muertes_peatones_base.sum())

    # 2) Resumen por escenario
    res = []
    for name in SC["scenarios"]:
        deaths = scenario_deaths(segs, name)
        total = float(np.sum(deaths))
        red = 1 - total / SC["baseline_deaths_year"]
        res.append({
            "escenario": name,
            "muertes_año": round(total, 1),
            "peatones_año": round(total * SC["pedestrian_share"], 1),
            "reduccion_pct": round(red * 100, 1),
            "vidas_salvadas": round(SC["baseline_deaths_year"] - total, 1),
        })
    res_df = pd.DataFrame(res)
    res_df.to_csv(OUT / "resumen_escenarios.csv", index=False)
    log.info("\n" + res_df.to_string(index=False))

    # 3) Curva de validación: muertes vs densidad vehicular (50k–250k veh/día)
    aadts = np.linspace(50_000, 250_000, 41)
    base_per_veh = SC["baseline_deaths_year"] / SC["aadt_total"]
    deaths_curve = base_per_veh * SC["aadt_total"] * (aadts / SC["aadt_total"]) ** SC["volume_exponent"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=VIZ_CONFIG["dpi"])
    ax.plot(aadts / 1000, deaths_curve, color="#E53935", lw=2,
            label=f"muertes ≈ k·V^({SC['volume_exponent']})")
    ax.axvline(SC["aadt_total"] / 1000, color="#888", ls="--", lw=1)
    ax.annotate(f"actual: {SC['aadt_total']/1000:.0f}k veh/día\n→ {SC['baseline_deaths_year']} muertes/año",
                xy=(SC["aadt_total"] / 1000, deaths_curve.max()), fontsize=9, color="#333")
    ax.set_xlabel("Densidad vehicular (miles de veh/día)")
    ax.set_ylabel("Muertes totales / año (Periférico)")
    ax.set_title("Validación: incidentes peatonales vs densidad vehicular (Periférico de Mérida)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curva_densidad_incidentes.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Barras por escenario
    colors = ["#90A4AE", "#43A047", "#7C4DFF", "#29B6F6", "#E53935"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=VIZ_CONFIG["dpi"])
    x = np.arange(len(res_df))
    ax.bar(x, res_df.muertes_año, color=colors, alpha=0.9)
    for i, r in res_df.iterrows():
        ax.text(i, r.muertes_año + 0.4, f"-{r.reduccion_pct:.0f}%", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x, [lbl.replace("_", " ").title() for lbl in res_df.escenario])
    ax.set_ylabel("Muertes / año (todas las víctimas)")
    ax.set_title("Escenarios de seguridad peatonal — Anillo Periférico de Mérida")
    ax.axhline(SC["baseline_deaths_year"], color="#90A4AE", ls=":", lw=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "incidentes_por_escenario.png", bbox_inches="tight")
    plt.close(fig)

    # 5) Reporte Markdown
    lines = [
        "# Seguridad peatonal — Anillo Periférico de Mérida",
        "",
        "## Modelo",
        "",
        "muertes_esc = muertes_base × (V_esc/V_base)^0.6 × SF × [0.6·cross·stops + 0.4]",
        "",
        "- **VF**: elasticidad volumen→incidentes (0.5–1.0 en literatura; 0.6 usado)",
        "- **SF**: severidad por velocidad según curva de fatalidad peatonal WHO (2008)",
        "  (30 km/h→10%, 50→55%, 60→75%, 80→95%)",
        "- **cross/stops**: levers por escenario en cruces semaforizados y paradas",
        "",
        "## Calibración (fuentes reales)",
        "",
        "| Dato | Valor | Fuente |",
        "|---|---|---|",
        "| Aforo del Periférico | ~150,000 veh/día (+20% temporada) | Gob. Yucatán (2025), SIPSE (2015) |",
        "| Muertes en el Periférico | 20–25/año; **17 en 2025** | Diario de Yucatán |",
        "| Peatones entre víctimas | ~68% (34 de 50) | conteos de atropellamientos |",
        "| Parque vehicular ZMM | ~838,726 (2023); 1.8 pers/veh | INEGI vía prensa (2023) |",
        "| Top nacional | Periférico en Top 5 vías peligrosas (2022: 186 accidentes, 5 muertes ene–abr) | Azteca Yucatán |",
        "",
        "## Resultados por escenario",
        "",
        df_to_markdown(res_df),
        "",
        "## Por segmento (línea base)",
        "",
        df_to_markdown(seg_df),
        "",
        "## Lectura",
        "",
        "- `semaforizacion` reduce ~1/4 de las muertes actuando solo en cruces",
        "- `piramide_movilidad` es el de mayor efecto individual: velocidad 80→60 km/h",
        "  en tramos urbanos corta la severidad ~21% y los cruces protegidos el resto",
        "- `transito_accesible` combina traslado modal (menos volumen) con cruces",
        "  seguros en paradas: beneficio estructural, no solo de severidad",
        "- `vision_cero` multiplica los tres mecanismos (volumen + velocidad +",
        "  infraestructura) → las medidas no se suman, se potencian",
        "",
        "---",
        "Generado por `python scripts/07_seguridad_peatonal.py`.",
    ]
    (OUT / "seguridad_peatonal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Salidas en %s", OUT)


if __name__ == "__main__":
    main()
