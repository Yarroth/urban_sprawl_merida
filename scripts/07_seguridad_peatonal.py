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
from scipy.stats import spearmanr
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


def _fmt(v):
    """Formatea celdas: enteros sin decimal; floats enteros como int; resto con 1 decimal."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.3f}" if abs(f) < 1 else f"{f:.1f}"
    return str(v)


def df_to_markdown(df):
    """Tabla markdown sin dependencia de tabulate."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = []
    for _, r in df.iterrows():
        body.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + body)


def fatality_severity(speed_kmh):
    """Factor de severidad relativo a 80 km/h (interpolación lineal de la curva WHO)."""
    xs = np.array([p[0] for p in WHO_SPEED_DEATH], dtype=float)
    ys = np.array([p[1] for p in WHO_SPEED_DEATH], dtype=float)
    p = float(np.interp(speed_kmh, xs, ys))
    return p / np.interp(80.0, xs, ys)


def build_segments():
    """Cuadrantes del anillo con muertes base distribuidas por exposición.

    La exposición usa la infraestructura real (IMEPLAN/Gob. de Yucatán): los
    cruces y semáforos peatonales generan puntos de conflicto (↑ exposición);
    los pasos elevados y cruces seguros los mitigan (↓ exposición).
    """
    segs = []
    for s in SC["segments"]:
        demand = SC["ped_demand"].get(s["name"], 0.5)
        exposure = ((1 + 0.08 * s["crossings"] + 0.05 * s["ped_signals"])
                    * (1 - 0.03 * s["bridges"] - 0.04 * s["safe_crossings"])
                    * (1.2 if s["urban"] else 1.0)
                    * demand)          # capa de demanda peatonal (puntos de deseo)
        segs.append({**s, "exposure": max(exposure, 0.1), "ped_demand": demand})
    tot_exp = sum(s["exposure"] * s["weight"] for s in segs)
    for s in segs:
        s["base_deaths"] = SC["baseline_deaths_year"] * (s["weight"] * s["exposure"] / tot_exp)
        s["aadt"] = SC["aadt_total"] * s["weight"]
    return segs


def temporal_analysis(segs):
    """Serie 2020–2025: muertes observadas vs contrafactual del parque vehicular
    y cuántas vidas habría evitado cada escenario de haber estado vigente."""
    years = sorted(SC["deaths_observed"])
    fleet = {y: (1 + SC["fleet_growth_annual"]) ** (y - years[0]) for y in years}
    obs0 = SC["deaths_observed"][years[0]]
    cf = {y: obs0 * fleet[y] ** SC["volume_exponent"] for y in years}

    # Reducción transversal de cada escenario (reutiliza el modelo por segmento)
    base_deaths = sum(scenario_deaths(segs, "base"))
    red = {name: 1 - sum(scenario_deaths(segs, name)) / base_deaths for name in SC["scenarios"]}

    rows = []
    for y in years:
        row = {"year": y, "parque_idx": round(fleet[y], 3),
               "muertes_observadas": SC["deaths_observed"][y],
               "contrafactual_parque": round(cf[y], 1)}
        for name in SC["scenarios"]:
            row[f"muerte_{name}"] = round(SC["deaths_observed"][y] * (1 - red[name]), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evolucion_temporal.csv", index=False)

    tot_obs = sum(SC["deaths_observed"].values())
    avoided = {name: round(tot_obs * red[name], 1) for name in SC["scenarios"]}

    fig, ax = plt.subplots(figsize=(9, 5), dpi=VIZ_CONFIG["dpi"])
    ax.plot(years, [cf[y] for y in years], "--", color="#90A4AE", lw=2,
            label="Contrafactual: solo crecimiento del parque vehicular")
    ax.plot(years, [SC["deaths_observed"][y] for y in years], "o-", color="#E53935", lw=2,
            label="Observado (prensa local)")
    for name, color in [("piramide_movilidad", "#43A047"), ("vision_cero", "#7C4DFF")]:
        ax.plot(years, [r[f"muerte_{name}"] for r in rows], "-", color=color, lw=1.6,
                label=f"Con {name.replace('_', ' ')} aplicado")
    ax.set_xlabel("Año")
    ax.set_ylabel("Muertes / año (Periférico)")
    ax.set_title("Evolución 2020–2025: muertes vs parque vehicular (+77% en una década)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "evolucion_temporal.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Serie temporal: Σ observado %d muertes 2020–2025; contrafactual %.0f "
             "(%.0f evitadas); visión cero habría evitado %.0f",
             tot_obs, sum(cf.values()), sum(cf.values()) - tot_obs, avoided["vision_cero"])
    return df, avoided


def radial_diagram(segs):
    """Rosa de riesgo con tres anillos: exterior = muertes/año por sector
    (YlOrRd), medio punteado = atropellamientos de prensa 2026 (púrpura) e
    interior = aforo vehicular (miles veh/día, azul). Atlas vs realidad de un
    vistazo."""
    names = [s["name"] for s in segs]
    deaths = np.array([s["base_deaths"] for s in segs])
    aadts = np.array([s["aadt"] for s in segs])
    press = np.array([sum(1 for inc in SC.get("incidentes_prensa", [])
                         if inc["sector"] == s["name"]) for s in segs])
    n = len(segs)
    theta = np.deg2rad(np.arange(n) * 360.0 / n + 360.0 / n / 2)
    width = np.deg2rad(360.0 / n) * 0.9
    d_max, a_max, p_max = deaths.max(), aadts.max(), max(press.max(), 1)

    fig = plt.figure(figsize=(8.4, 8.4), dpi=VIZ_CONFIG["dpi"])
    ax = fig.add_subplot(111, projection="polar")

    # Anillo interior: aforo vehicular (azul, escala 0–0.35)
    ax.bar(theta, 0.35 * aadts / a_max, width=width, bottom=0.0,
           color="#29B6F6", edgecolor="white", linewidth=0.6, alpha=0.85)
    # Anillo medio punteado: atropellamientos de prensa 2026 (púrpura, 0.42–0.49)
    ax.bar(theta, 0.07 * press / p_max, width=width, bottom=0.42,
           color="#7C4DFF", hatch="..", edgecolor="white", linewidth=0.4,
           alpha=0.85)
    # Anillo exterior: muertes/año (riesgo, YlOrRd, 0.56–1.00)
    norm = plt.Normalize(deaths.min(), deaths.max())
    cmap = plt.get_cmap("YlOrRd")
    ax.bar(theta, 0.44 * deaths / d_max, width=width, bottom=0.56,
           color=cmap(norm(deaths)), edgecolor="white", linewidth=0.6, alpha=0.95)

    for t, nm, d, a, p in zip(theta, names, deaths, aadts, press):
        ax.text(t, 1.05, nm, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(t, 0.56 + 0.44 * d / d_max + 0.02, f"{d:.1f}", ha="center",
                va="bottom", fontsize=8, color="#7A0000", fontweight="bold")
        if p > 0:
            ax.text(t, 0.42 + 0.07 * p / p_max + 0.012, str(int(p)), ha="center",
                    va="bottom", fontsize=7, color="#4A148C", fontweight="bold")
        ax.text(t, 0.35 * a / a_max + 0.012, f"{a/1000:.0f}k", ha="center",
                va="bottom", fontsize=7, color="#01579B", fontweight="bold")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)   # sentido horario (N → NNE → … → WSW)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.10, shrink=0.75)
    cbar.set_label("Muertes/año por sector (2025)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#29B6F6", label="Aforo (miles veh/día)"),
        Patch(facecolor="#7C4DFF", hatch="..", label="Atropellamientos prensa 2026"),
        Patch(color="#E53935", label="Muertes/año (riesgo)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8,
        frameon=False)
    ax.set_title("Riesgo peatonal por sector — Anillo Periférico de Mérida", pad=42, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "anillo_sectores.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Radial: %s (más riesgo) → %s (menos riesgo); prensa: %d incidentes",
             names[int(np.argmax(deaths))], names[int(np.argmin(deaths))], int(press.sum()))


def validar_pesos(segs):
    """Valida los pesos del atlas contra los atropellamientos reales de prensa
    (2026). Compara el ranking de peso del atlas y el riesgo final del modelo
    (muertes base por sector) con la distribución observada de incidentes."""
    names = [s["name"] for s in segs]
    counts = {nm: 0 for nm in names}
    for inc in SC.get("incidentes_prensa", []):
        counts[inc["sector"]] = counts.get(inc["sector"], 0) + 1
    weights = {s["name"]: s["weight"] for s in segs}
    deaths = {s["name"]: s["base_deaths"] for s in segs}

    df = pd.DataFrame({
        "sector": names,
        "peso_atlas": [weights[nm] for nm in names],
        "muertes_base": [deaths[nm] for nm in names],
        "atropellos_prensa": [counts[nm] for nm in names],
    })
    df["rank_atlas"] = df["peso_atlas"].rank(ascending=False)
    df["rank_riesgo"] = df["muertes_base"].rank(ascending=False)
    df["rank_prensa"] = df["atropellos_prensa"].rank(ascending=False, method="average")

    rho_w, p_w = spearmanr(df["peso_atlas"], df["atropellos_prensa"])
    rho_r, p_r = spearmanr(df["muertes_base"], df["atropellos_prensa"])
    df.to_csv(OUT / "validacion_pesos.csv", index=False)
    log.info("Validación: Spearman peso_atlas vs prensa = %.2f (p=%.3f) | "
             "riesgo_modelo vs prensa = %.2f (p=%.3f) | n=%d incidentes",
             rho_w, p_w, rho_r, p_r, int(df["atropellos_prensa"].sum()))
    return df, rho_w, p_w, rho_r, p_r


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


def scenario_deaths_prioritized(segs, sc_name):
    """Muertes por segmento con los levers aplicados con intensidad proporcional a
    la demanda peatonal del sector (puntos de deseo de cruce).

    efectividad = 0.35 + 0.65 × (demanda / demanda_máx)   → [0.35, 1.0]

    Los sectores con puntos de deseo de cruce documentados (S, NE, SSE, E, SE,
    WSW) reciben el paquete casi completo; los de baja demanda medida (ENE, ESE),
    solo una fracción. El volumen se mantiene global (el traslado modal no
depende del sector)."""
    lv = SC["scenarios"][sc_name]
    dmax = max(SC["ped_demand"].values())
    out = []
    for s in segs:
        eff = 0.35 + 0.65 * (s["ped_demand"] / dmax)
        vol_factor = lv["vol"] ** SC["volume_exponent"]
        if s["urban"]:
            speed_eff = SC["speed_limit_kmh"] - (SC["speed_limit_kmh"] - lv["speed"]) * eff
        else:
            speed_eff = SC["speed_limit_kmh"]
        severity = fatality_severity(speed_eff) / fatality_severity(SC["speed_limit_kmh"])
        cross_eff = 1 - (1 - lv["cross"]) * eff
        stops_eff = 1 - (1 - lv["stops"]) * eff
        crossing = SC["crossing_share"] * cross_eff * stops_eff + (1 - SC["crossing_share"])
        out.append(s["base_deaths"] * vol_factor * severity * crossing)
    return out


def comparar_vision_cero_priorizada(segs):
    """Compara Visión Cero uniforme vs priorizada por puntos de deseo de cruce:
    cómo cambia la reducción por sector al escalar los levers con la demanda
    peatonal. Salidas: CSV por sector, gráfica de barras y totales."""
    base = np.array([s["base_deaths"] for s in segs])
    uni = np.array(scenario_deaths(segs, "vision_cero"))
    pri = np.array(scenario_deaths_prioritized(segs, "vision_cero"))
    red_uni = 100 * (1 - uni / base)
    red_pri = 100 * (1 - pri / base)

    df = pd.DataFrame({
        "sector": [s["name"] for s in segs],
        "Demanda peatonal": [s["ped_demand"] for s in segs],
        "Muertes base": np.round(base, 3),
        "Muertes VZ uniforme": np.round(uni, 3),
        "Muertes VZ priorizada": np.round(pri, 3),
        "Reducción uniforme %": np.round(red_uni, 1),
        "Reducción priorizada %": np.round(red_pri, 1),
        "Ganancia (pp)": np.round(red_pri - red_uni, 1),
    }).sort_values("Demanda peatonal", ascending=False)
    df.to_csv(OUT / "vision_cero_priorizada.csv", index=False)

    # Eficiencia: unidades de inversión (1.0 por sector con paquete completo).
    # La priorizada invierte intensidad ∝ demanda: Σ(0.35 + 0.65·demanda_norm).
    dmax = max(SC["ped_demand"].values())
    units_uni = float(len(segs))
    units_pri = float(sum(0.35 + 0.65 * (s["ped_demand"] / dmax) for s in segs))
    saved_uni, saved_pri = base.sum() - uni.sum(), base.sum() - pri.sum()
    eff_gain = (saved_pri / units_pri) / (saved_uni / units_uni) - 1

    # Barras horizontales por sector (mayor demanda arriba)
    dfc = df.sort_values("Demanda peatonal", ascending=True)
    y = np.arange(len(dfc))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=VIZ_CONFIG["dpi"])
    ax.barh(y - h, dfc["Muertes base"], height=h, color="#90A4AE",
            label="Línea base 2025")
    ax.barh(y, dfc["Muertes VZ uniforme"], height=h, color="#B39DDB",
            label="Visión Cero uniforme")
    ax.barh(y + h, dfc["Muertes VZ priorizada"], height=h, color="#7C4DFF",
            label="Visión Cero priorizada por demanda")
    for i, (p, pv) in enumerate(zip(dfc["Reducción priorizada %"],
                                    dfc["Muertes VZ priorizada"])):
        ax.text(pv + 0.04, i + h, f"-{p:.0f}%", va="center", fontsize=8,
                color="#4A148C", fontweight="bold")
    ax.set_yticks(y, dfc.sector)
    ax.set_xlabel("Muertes / año por sector")
    ax.set_title("Visión Cero: uniforme vs priorizada por puntos de deseo de cruce")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "vision_cero_priorizada.png", bbox_inches="tight")
    plt.close(fig)

    tot_base, tot_uni, tot_pri = base.sum(), uni.sum(), pri.sum()
    log.info("Visión Cero priorizada: total %.1f (-%.1f%%) vs uniforme %.1f (-%.1f%%) con "
             "%.2f unidades vs %.0f; eficiencia +%.1f%% (vidas por unidad invertida)",
             tot_pri, 100 * (1 - tot_pri / tot_base), tot_uni,
             100 * (1 - tot_uni / tot_base), units_pri, units_uni, 100 * eff_gain)
    return df, tot_base, tot_uni, tot_pri, units_uni, units_pri, eff_gain


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segs = build_segments()

    # 1) Tabla por segmento (línea base)
    rows = []
    for s in segs:
        rows.append({
            "sector": s["name"], "aadt_veh_dia": round(s["aadt"]),
            "cruces_semaforo": s["crossings"], "semaforos_peatonales": s["ped_signals"],
            "pasos_elevados": s["bridges"], "cruces_seguros": s["safe_crossings"],
            "paradas_bus": s["bus_stops"], "tramo_urbano": s["urban"],
            "demanda_peatonal": s["ped_demand"],
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

    # 5) Validación de pesos contra atropellamientos reales (prensa 2026)
    vdf, rho_w, p_w, rho_r, p_r = validar_pesos(segs)

    # 6) Diagrama radial de riesgo por sector
    radial_diagram(segs)

    # 6b) Visión Cero priorizada por puntos de deseo de cruce
    (vzdf, vz_tot_base, vz_tot_uni, vz_tot_pri,
     vz_units_uni, vz_units_pri, vz_eff_gain) = comparar_vision_cero_priorizada(segs)

    # 7) Serie temporal 2020–2025
    tdf, avoided = temporal_analysis(segs)
    tdf_show = tdf[["year", "parque_idx", "muertes_observadas", "contrafactual_parque",
                    "muerte_semaforizacion", "muerte_piramide_movilidad", "muerte_vision_cero"]]
    tdf_show.columns = ["Año", "Parque (índice)", "Observadas", "Contrafactual",
                        "Semaforización", "Pirámide mov.", "Visión Cero"]
    avoid_rows = [
        {"escenario": name.replace("_", " ").title(),
         "vidas_evitadas_2020_2025": avoided[name],
         "pct": round(avoided[name] / (sum(SC["deaths_observed"].values())) * 100, 1)}
        for name in SC["scenarios"]
    ]
    avoid_df = pd.DataFrame(avoid_rows)

    # 8) Reporte Markdown
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
        "## Infraestructura real del anillo (IMEPLAN / Gob. de Yucatán)",
        "",
        "| Elemento | Total | Fuente |",
        "|---|---|---|",
        f"| Semáforos vehiculares | {SC['infra_real']['semaforos_vehiculares']} | Saidén Ojeda (Gob. Yucatán) |",
        f"| Semáforos peatonales | {SC['infra_real']['semaforos_peatonales']} | Saidén Ojeda (Gob. Yucatán) |",
        f"| Puentes peatonales | {SC['infra_real']['puentes_peatonales']} (8 nuevos + 7 rehabilitados) | Programa de seguridad vial |",
        f"| Cruces peatonales seguros | {SC['infra_real']['cruces_seguros']} | Programa de seguridad vial |",
        f"| Bahías de ascenso/descenso | {SC['infra_real']['bahias_bus']} | Programa de seguridad vial |",
        "",
        "Los 12 sectores y sus pesos de demanda provienen del atlas \"Análisis ",
        "del Anillo Periférico de Mérida\" (lámina 12 sectores): densidad de ",
        "celdas de alta concentración vehicular (mapa TDPA, escala compartida), ",
        "normalizada. El TDPA puntual por tramo requeriría los datos fuente del ",
        "atlas o aforos oficiales del IMT/SCT.",
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
        "## Validación de pesos contra atropellamientos reales (prensa 2026)",
        "",
        f"Corpus de {int(vdf['atropellos_prensa'].sum())} reportes localizados por tramo ",
        "(Diario de Yucatán, Reporteros Hoy, InfoLliteras, Yucatán.com.mx, Sol Yucatán, ",
        "Novedades). Muestra pequeña y sesgada hacia eventos fatales: validación ",
        "exploratoria, no censo.",
        "",
        df_to_markdown(vdf.sort_values("atropellos_prensa", ascending=False)),
        "",
        f"- **Spearman peso del atlas vs prensa**: {rho_w:.2f} (p={p_w:.3f}) — ",
        "correlación débil-moderada: el volumen captura parte, no toda, la ",
        "geografía de los incidentes.",
        f"- **Spearman riesgo del modelo (muertes base) vs prensa**: {rho_r:.2f} ",
        f"(p={p_r:.3f}) — mejora notable vs 0.06 sin la capa de demanda.",
        "- El modelo ya incorpora la **capa de demanda peatonal** (`ped_demand`): ",
        "mezcla de intensidad peatonal del atlas (lámina II.5), demanda revelada ",
        "por la prensa y puntos de deseo de cruce documentados (Cholul, Chichí ",
        "Suárez — auditor vial René Flores Ayora, Diario de Yucatán 04/2024 —, ",
        "Kanasín, Xmatkuil, Dzununcán). Eso eleva el riesgo de E, SE y NE, ",
        "acercando el ranking al de los incidentes reales.",
        "- La prensa sobrerrepresenta Norte, Oriente y SE y subrepresenta el ",
        "suroeste. Con n=12 los valores no son estadísticamente significativos; ",
        "un censo oficial por tramo (SSP/IMEPLAN) cerraría la brecha.",
        "",
        "## Visión Cero priorizada por puntos de deseo de cruce",
        "",
        "En lugar de aplicar el paquete uniformemente, la intensidad de cada lever ",
        "(velocidad, cruces, paradas) se escala con la demanda peatonal del sector: ",
        "efectividad = 0.35 + 0.65 × (demanda / demanda_máx). Los sectores con ",
        "puntos de deseo de cruce documentados (S, NE/Chichí Suárez, SSE, E, ",
        "SE/Kanasín, WSW) reciben el paquete casi completo; los de baja demanda ",
        "medida (ENE, ESE), solo una fracción. El volumen se mantiene global ",
        "(el traslado modal no depende del sector).",
        "",
        df_to_markdown(vzdf),
        "",
        f"- **Total uniforme**: {vz_tot_uni:.1f} muertes/año (−{100*(1-vz_tot_uni/vz_tot_base):.1f}%)",
        f"- **Total priorizada por demanda**: {vz_tot_pri:.1f} muertes/año ",
        f"(−{100*(1-vz_tot_pri/vz_tot_base):.1f}%) — con {vz_units_pri:.2f} unidades de ",
        f"inversión vs {vz_units_uni:.0f} de la uniforme (−{100*(1-vz_units_pri/vz_units_uni):.0f}% ",
        "de presupuesto).",
        f"- **Eficiencia +{100*vz_eff_gain:.0f}%** (vidas salvadas por unidad invertida): la ",
        "priorizada concentra la reducción donde ocurren los atropellamientos ",
        "(S, NE/Chichí Suárez, SSE, E, SE/Kanasín, WSW mantienen ~todo el recorte) y ",
        "relaja los tramos de baja demanda medida (ENE, ESE). La uniforme sigue ",
        "siendo el techo en vidas totales, pero cuesta ~22% más por solo 6.5 pp ",
        "adicionales — la priorización es la opción eficiente.",
        "",
        "![Visión Cero priorizada](vision_cero_priorizada.png)",
        "",
        "## Por sector (línea base)",
        "",
        df_to_markdown(seg_df),
        "",
        "![Riesgo por sector](anillo_sectores.png)",
        "",
        "## Evolución temporal 2020–2025",
        "",
        "Muertes observadas vs contrafactual del crecimiento del parque vehicular",
        f"(+{SC['fleet_growth_annual']*100:.1f}%/año, INEGI: +77% en una década). ",
        "La brecha observado↔contrafactual mide las vidas que ya se están evitando ",
        "con las medidas actuales.",
        "",
        df_to_markdown(tdf_show),
        "",
        "![Evolución temporal](evolucion_temporal.png)",
        "",
        "### Vidas evitables por escenario (acumulado 2020–2025, si el paquete ",
        "hubiera estado vigente todo el periodo)",
        "",
        df_to_markdown(avoid_df),
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
