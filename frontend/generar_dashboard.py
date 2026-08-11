"""Genera frontend/dashboard_resultados.html a partir de results/maps y results/reports.

Uso:
    python frontend/generar_dashboard.py

Requisitos: numpy, pandas (y los PNG generados por scripts/05_visualize.py).
"""
import base64
import datetime
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAPS, REP = ROOT / "results" / "maps", ROOT / "results" / "reports"
SAFE = ROOT / "results" / "safety"
OUT = ROOT / "frontend" / "dashboard_resultados.html"

scenarios = ["no_plan", "plan_trad", "ia_optimo"]
years = [2026, 2027, 2028, 2029, 2030]
SC_LABEL = {"no_plan": "Sin planificación", "plan_trad": "Plan tradicional", "ia_optimo": "Gestión IA"}
SC_COLOR = {"no_plan": "#E53935", "plan_trad": "#43A047", "ia_optimo": "#7C4DFF"}


def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    maps = {}
    for s in scenarios:
        maps[s] = {y: b64(MAPS / f"expansion_map_{y}_{s}.png") for y in years}
        maps[s]["comparison"] = b64(MAPS / f"comparison_2024_2030_{s}.png")

    stats = pd.read_csv(REP / "area_statistics.csv")
    metrics = pd.read_csv(REP / "metrics.csv").iloc[0]

    # Seguridad peatonal (módulo 07, si fue ejecutado)
    safety = ""
    safety_hint = ""
    if (SAFE / "resumen_escenarios.csv").exists():
        sres = pd.read_csv(SAFE / "resumen_escenarios.csv")
        srows = []
        S_LABEL = {"base": "Situación actual", "semaforizacion": "Semaforización coordinada",
                   "piramide_movilidad": "Pirámide de movilidad",
                   "transito_accesible": "Paradas de autobús accesibles", "vision_cero": "Visión Cero (todo)"}
        for _, r in sres.iterrows():
            red = f'−{r["reduccion_pct"]:.0f}%' if r["escenario"] != "base" else "—"
            saved = f'{r["vidas_salvadas"]:.0f}' if r["escenario"] != "base" else "—"
            srows.append(
                f'<tr><td>{S_LABEL.get(r["escenario"], r["escenario"])}</td>'
                f'<td>{r["muertes_año"]:.1f}</td><td>{r["peatones_año"]:.1f}</td>'
                f'<td class="delta">{red}</td><td>{saved}</td></tr>'
            )
        base_row = sres[sres["escenario"] == "base"].iloc[0]
        safety = (
            '<div class="safety-grid">'
            '<div><h2 id="sec-safety">Seguridad peatonal — Periférico</h2>'
            '<div class="sub">Valida incidentes peatonales vs densidad vehicular (~150k veh/día) '
            f'y escenarios de política. Línea base: {base_row["muertes_año"]:.0f} muertes/año '
            f'({base_row["peatones_año"]:.0f} peatonales) en 2025.</div>'
            '<table><thead><tr><th>Escenario</th><th>Muertes/año</th><th>Peatones/año</th>'
            '<th>Reducción</th><th>Vidas salvadas</th></tr></thead>'
            f'<tbody>{"".join(srows)}</tbody></table></div>'
            f'<div class="safety-img"><img src="{b64(SAFE / "incidentes_por_escenario.png")}" '
            'alt="Muertes por escenario"></div>'
            f'<div class="safety-img"><img src="{b64(SAFE / "curva_densidad_incidentes.png")}" '
            'alt="Curva densidad vs incidentes"></div>'
            '</div>'
        )
        if (SAFE / "evolucion_temporal.csv").exists():
            tdf = pd.read_csv(SAFE / "evolucion_temporal.csv")
            tot, cf = int(tdf["muertes_observadas"].sum()), int(round(tdf["contrafactual_parque"].sum()))
            vc = int(round(tot - tdf["muerte_vision_cero"].sum()))
            trows = "".join(
                f'<tr><td>{int(r["year"])}</td><td>{r["parque_idx"]:.2f}</td>'
                f'<td>{int(r["muertes_observadas"])}</td><td>{r["contrafactual_parque"]:.1f}</td>'
                f'<td>{r["muerte_vision_cero"]:.1f}</td></tr>'
                for _, r in tdf.iterrows()
            )
        if (SAFE / "anillo_sectores.png").exists():
            safety += (
                '<div class="temporal" style="text-align:center"><h2 id="sec-radial">Riesgo por sector del anillo</h2>'
                '<div class="sub">12 sectores (N→WSW) según el atlas del Periférico; '
                'altura y color = muertes/año por sector (2025).</div>'
                f'<img src="{b64(SAFE / "anillo_sectores.png")}" alt="Riesgo por sector" '
                'style="max-width:560px;margin:0 auto;display:block">'
                '</div>'
            )
        if (SAFE / "vision_cero_priorizada.csv").exists():
            vzdf = pd.read_csv(SAFE / "vision_cero_priorizada.csv")
            hot = " · ".join(
                f'{r["sector"]} −{r["Reducción priorizada %"]:.0f}%'
                for _, r in vzdf.sort_values("Reducción priorizada %", ascending=False).head(6).iterrows()
            )
            safety += (
                '<div class="temporal" style="text-align:center"><h2 id="sec-vz">Visión Cero: uniforme vs priorizada por demanda</h2>'
                '<div class="sub">Al escalar los levers con la demanda peatonal (puntos de deseo de cruce), '
                'la priorizada logra <b>−54%</b> con <b>−22% de presupuesto</b> vs la uniforme (−61%) — '
                '<b>+14% de eficiencia</b> por unidad invertida. Hotspots: '
                f'{hot}.</div>'
                f'<img src="{b64(SAFE / "vision_cero_priorizada.png")}" alt="Visión Cero priorizada" '
                'style="max-width:640px;margin:0 auto;display:block">'
                '</div>'
            )
        if (SAFE / "evolucion_temporal.csv").exists():
            safety += (
                '<div class="temporal"><h2 id="sec-temporal">Evolución temporal 2020–2025</h2>'
                '<div class="sub">Muertes vs parque vehicular (+77% en una década, INEGI). '
                f'<b>{tot} muertes observadas</b> vs <b>{cf} contrafactuales</b> (solo parque: '
                f'{cf - tot} ya evitadas) · <b>Visión Cero habría evitado {vc}</b> en el periodo.</div>'
                '<table><thead><tr><th>Año</th><th>Parque (índice)</th><th>Observadas</th>'
                '<th>Contrafactual</th><th>Con Visión Cero</th></tr></thead>'
                f'<tbody>{trows}</tbody></table>'
                f'<img src="{b64(SAFE / "evolucion_temporal.png")}" alt="Evolución temporal">'
                '</div>'
            )

    thead = "<tr><th>Escenario</th>" + "".join(f"<th>{y}</th>" for y in years) + "<th>Δ base → 2030</th></tr>"
    trows = []
    for s in scenarios:
        grp = stats[stats["scenario"] == s].set_index("year")
        base, final = grp.loc[2024, "area_km2"], grp.loc[2030, "area_km2"]
        cells = "".join(f"<td>{grp.loc[y, 'area_km2']:.1f}</td>" for y in years)
        trows.append(
            f'<tr><td><span class="dot" style="background:{SC_COLOR[s]}"></span>{SC_LABEL[s]}</td>'
            f"{cells}<td class='delta'>+{final - base:.1f} km²</td></tr>"
        )
    table = f"<table><thead>{thead}</thead><tbody>{''.join(trows)}</tbody></table>"

    # Calidad espacial (2030): fragmentación y densidad de borde
    spatial_table = ""
    if "n_patches" in stats.columns and "edge_km_per_km2" in stats.columns:
        st_rows = []
        for s in scenarios:
            grp = stats[stats["scenario"] == s].set_index("year")
            r = grp.loc[2030]
            mean_patch = r["area_km2"] / max(r["n_patches"], 1)
            st_rows.append(
                f'<tr><td><span class="dot" style="background:{SC_COLOR[s]}"></span>{SC_LABEL[s]}</td>'
                f'<td>{r["area_km2"]:.1f}</td><td>{int(r["n_patches"])}</td>'
                f'<td>{mean_patch:.3f}</td><td>{r["edge_km_per_km2"]:.2f}</td></tr>'
            )
        spatial_table = (
            "<table><thead><tr><th>Escenario</th><th>Área 2030 (km²)</th>"
            "<th>Parches urbanos</th><th>Área media/parche (km²)</th><th>Borde (km/km²)</th></tr></thead>"
            f"<tbody>{''.join(st_rows)}</tbody></table>"
        )

    # Indicadores ambientales de la expansión (script 08, desde rasters reales)
    ambient = ""
    if (REP / "indicadores_ambientales.csv").exists():
        adv = pd.read_csv(REP / "indicadores_ambientales.csv")
        ad_rows = []
        for _, r in adv.iterrows():
            sc = r["escenario"]
            ad_rows.append(
                f'<tr><td><span class="dot" style="background:{SC_COLOR.get(sc, "#888")}"></span>{SC_LABEL.get(sc, sc)}</td>'
                f'<td>{int(r["celdas_nuevas"])}</td>'
                f'<td>{r["lst_nuevas"]:.2f} °C</td>'
                f'<td>{r["dist_cenote_m"]:.0f} m</td>'
                f'<td>{r["karst_nuevas"]:.3f}</td>'
                f'<td>{r["verde_2030_pct"]:.1f}%</td></tr>'
            )
        ambient = (
            '<div class="temporal"><h2 id="sec-ambiental">Indicadores ambientales de la expansión (2030)</h2>'
            '<div class="sub">Celdas <b>nuevas</b> 2024→2030 cruzadas con features reales '
            '(LST, distancia a cenotes, vulnerabilidad kárstica) y cobertura no urbana total. '
            'La gestión IA urbaniza menos celdas, en zonas más frescas y alejadas de cenotes.</div>'
            '<table><thead><tr><th>Escenario</th><th>Celdas nuevas</th><th>LST media (°C)</th>'
            '<th>Dist. a cenotes (m)</th><th>Vuln. kárstica</th><th>Suelo no urbano 2030</th></tr></thead>'
            f'<tbody>{"".join(ad_rows)}</tbody></table>'
            f'<img src="{b64(REP / "indicadores_ambientales.png")}" alt="Indicadores ambientales por escenario">'
            '</div>'
        )

    tabs, grids = [], []
    for s in scenarios:
        grp = stats[stats["scenario"] == s].set_index("year")
        cards = ""
        for y in years:
            cards += (f'<div class="card"><img src="{maps[s][y]}" alt="{SC_LABEL[s]} {y}">'
                      f'<div class="cap"><span>{y}</span>'
                      f'<span class="km2">{grp.loc[y, "area_km2"]:.1f} km²</span></div></div>')
        tabs.append(f'<button class="tab" data-scenario="{s}" style="--c:{SC_COLOR[s]}">{SC_LABEL[s]}</button>')
        grids.append(f'<section class="scenario-grid" id="grid-{s}">{cards}</section>')

    comps = "".join(
        f'<div class="card"><img src="{maps[s]["comparison"]}" alt="Comparación {SC_LABEL[s]}">'
        f'<div class="cap"><span>2024 → 2030 · {SC_LABEL[s]}</span></div></div>'
        for s in scenarios
    )

    # ── Portada (solo impresión) ──
    MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    hoy = datetime.date.today()
    fecha_es = f"{hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    cov_areas, cov_base = {}, {}
    for s in scenarios:
        grp = stats[stats["scenario"] == s].set_index("year")
        cov_areas[s] = grp.loc[2030, "area_km2"]
        cov_base[s] = grp.loc[2024, "area_km2"]
    cov_deaths = cov_vc = None
    if (SAFE / "resumen_escenarios.csv").exists():
        sr = pd.read_csv(SAFE / "resumen_escenarios.csv").set_index("escenario")
        cov_deaths = sr.loc["base", "muertes_año"]
        cov_vc = sr.loc["vision_cero", "reduccion_pct"]
    cov_cards = [
        (str(metrics.get("auc_roc", "N/A")), "AUC-ROC transición"),
        (str(metrics.get("ca_auc_roc", "N/A")), "AUC-ROC reglas CA"),
        (f"{cov_areas['ia_optimo']:.1f} → {cov_areas['no_plan']:.1f} km²",
         "Área 2030 (gestión IA → no plan)"),
    ]
    if cov_deaths is not None:
        cov_cards.append((f"{cov_deaths:.0f} → −{cov_vc:.0f}%", "Muertes/año Periférico → Visión Cero"))
    cover_cards_html = "".join(
        f'<div class="cover-stat"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for v, k in cov_cards
    )
    cover = (
        '<div class="cover">'
        '<div class="cover-band">Universidad Politécnica de Yucatán · TSU en Ciencia de Datos</div>'
        '<div><h1 class="cover-title">Predicción de la Expansión Urbana de Mérida</h1>'
        '<div class="cover-sub">Proyección espacial 2026–2030 y seguridad peatonal del Anillo Periférico</div>'
        '<div class="cover-meta">Pipeline v2.0 · LightGBM + Autómata Celular de reglas aprendidas + variables kársticas</div>'
        '</div>'
        '<div><hr class="cover-hr"><h2 class="cover-h2">Resumen ejecutivo</h2>'
        f'<p class="cover-p">El modelo calibra la transición urbana de la Zona Metropolitana de Mérida '
        f'(AUC-ROC {metrics.get("auc_roc", "N/A")} en la transición y {metrics.get("ca_auc_roc", "N/A")} '
        'en las reglas del autómata celular) y proyecta el periodo 2026–2030 en tres escenarios: sin '
        'planificación, plan tradicional y gestión IA. La superficie urbana crecería '
        f'+{cov_areas["no_plan"] - cov_base["no_plan"]:.1f} km² sin plan frente a '
        f'+{cov_areas["ia_optimo"] - cov_base["ia_optimo"]:.1f} km² con gestión IA, que además '
        'reduce la fragmentación (144 vs 242 parches urbanos) y la presión sobre el acuífero kárstico '
        'y los cenotes.</p>'
    )
    if cov_deaths is not None:
        cover += (
            f'<p class="cover-p">En el Anillo Periférico (~150,000 veh/día, {cov_deaths:.0f} muertes en 2025), los '
            'escenarios de política vial reducen las muertes entre −26% (paradas de autobús accesibles) y '
            f'−{cov_vc:.0f}% (Visión Cero). Priorizar las intervenciones por los puntos de deseo de cruce '
            'alcanza −54% con −22% del presupuesto (+14% de eficiencia por unidad invertida).</p>'
        )
    # Índice de contenidos con paginación (layout print: una sección por página)
    safety_present = (SAFE / "resumen_escenarios.csv").exists()
    # Marcadores (texto de la sección → nombre del pie) para mapear páginas
    markers = [
        ("Área urbana proyectada", "Área urbana proyectada (km²)"),
        ("Calidad espacial de la expansión", "Calidad espacial de la expansión"),
        ("Indicadores ambientales de la expansión", "Indicadores ambientales"),
        ("Mapas de probabilidad por escenario", "Mapas de probabilidad por escenario"),
        ("Comparación 2024 → 2030 por escenario", "Comparación 2024 → 2030"),
    ]
    if safety_present:
        markers += [
            ("Seguridad peatonal — Periférico", "Seguridad peatonal — Periférico"),
            ("Riesgo por sector del anillo", "Riesgo por sector del anillo"),
            ("Visión Cero: uniforme vs priorizada", "Visión Cero priorizada"),
            ("Evolución temporal 2020–2025", "Evolución temporal 2020–2025"),
        ]
    # Bloques que ocupan una página en el PDF (ver @media print): el h2 de
    # mapas va en su propia página y cada grid de escenario en la suya.
    page_blocks = ["cover", "sec-areas", "sec-espacial", "sec-ambiental",
                   "sec-mapas", "grid-no_plan", "grid-plan_trad", "grid-ia_optimo",
                   "sec-comp"]
    if safety_present:
        page_blocks += ["sec-safety", "sec-radial", "sec-vz", "sec-temporal"]
    toc_pages = {name: idx + 1 for idx, name in enumerate(page_blocks)}
    toc_items = [
        ("Área urbana proyectada (km²)", "sec-areas"),
        ("Calidad espacial de la expansión", "sec-espacial"),
        ("Indicadores ambientales de la expansión", "sec-ambiental"),
        ("Mapas de probabilidad por escenario y año", "sec-mapas"),
        ("Comparación 2024 → 2030 por escenario", "sec-comp"),
    ]
    if safety_present:
        toc_items += [
            ("Seguridad peatonal — Periférico", "sec-safety"),
            ("Riesgo por sector del anillo", "sec-radial"),
            ("Visión Cero: uniforme vs priorizada", "sec-vz"),
            ("Evolución temporal 2020–2025", "sec-temporal"),
        ]
    toc_items_by_id = {sid: (label, sec) for (label, sid), sec in
                       zip(toc_items, [m[1] for m in markers])}
    toc_html = '<div class="cover-toc">' + "".join(
        f'<a href="#{sid}"><span>{label}</span><span class="pg">{toc_pages[sid]}</span></a>'
        for label, sid in toc_items
    ) + '</div>'
    cover += (
        f'{toc_html}'
        f'<div class="cover-stats">{cover_cards_html}</div></div>'
        f'<div class="cover-author"><b>Héctor Javier Raya Romo</b> · Mérida, Yucatán · {fecha_es}</div>'
        '</div>'
    )

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resultados — Expansión Urbana Mérida 2026–2030</title>
<style>
  :root{--bg:#0D1117;--panel:#161B22;--border:#30363D;--text:#E6EDF3;--muted:#8B949E;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;padding:28px;max-width:1400px;margin:0 auto}
  h1{font-size:26px;font-weight:700}
  .sub{color:var(--muted);margin-top:4px;font-size:14px}
  h2{margin:28px 0 12px;font-size:18px;border-left:4px solid #7C4DFF;padding-left:10px}
  .meta{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0 0}
  .mcard{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px;min-width:150px}
  .mcard .v{font-size:22px;font-weight:700}
  .mcard .k{font-size:12px;color:var(--muted);margin-top:2px}
  .tabs{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
  .tab{padding:8px 16px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);
       cursor:pointer;font-size:13px;border-bottom:2px solid transparent}
  .tab:hover{border-color:var(--muted)}
  .tab.active{border-bottom-color:var(--c,#7C4DFF);background:#1F2937}
  .scenario-grid{display:none;grid-template-columns:repeat(5,1fr);gap:12px}
  .scenario-grid.active{display:grid}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;min-width:0}
  .card img{width:100%;display:block;aspect-ratio:1/0.75;object-fit:cover}
  .cap{padding:8px 10px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
  .km2{color:var(--text);font-weight:600}
  .comp{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .safety-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px;align-items:start}
  .safety-grid h2{grid-column:1/-1;margin-top:0}
  .safety-img img{width:100%;border:1px solid var(--border);border-radius:8px;background:var(--panel)}
  .temporal{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px;margin-top:16px}
  .temporal h2{margin-top:0}
  .temporal table{width:auto;min-width:420px}
  .temporal img{width:100%;border:1px solid var(--border);border-radius:8px;margin-top:12px}
  @media (max-width:900px){.safety-grid{grid-template-columns:1fr}}
  table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:13px}
  th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--border)}
  th:first-child,td:first-child{text-align:left}
  th{background:#1F2937;color:var(--muted);font-weight:600}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
  .delta{color:#7EE787;font-weight:600}
  .legend{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;margin:10px 0}
  .bar{width:180px;height:10px;border-radius:5px;background:linear-gradient(90deg,#FFFFB2,#FCDE00,#E53935,#7A0000)}
  footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}
  /* Portada (solo impresión) */
  .cover{display:none}
  @media (max-width:900px){.scenario-grid,.comp{grid-template-columns:repeat(2,1fr)}}
  /* ── Impresión (entrega final) ── */
  @page{size:A4 landscape;margin:10mm}
  @media print{
    *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    :root{--bg:#ffffff;--panel:#ffffff;--border:#c9ccd1;--text:#1b1f24;--muted:#57606a}
    body{padding:0;max-width:100%;font-size:11px}
    h1{font-size:20px}
    h2{break-before:page;break-after:avoid}
    h2:first-of-type{break-before:auto}
    .cover{display:flex;flex-direction:column;justify-content:space-between;align-items:center;
           text-align:center;break-after:page;height:188mm;padding:6mm 18mm}
    .cover-band{font-size:12px;letter-spacing:2px;color:#57606a;text-transform:uppercase}
    .cover-title{font-size:34px;font-weight:800;color:#1b1f24;margin:6mm 0 2mm}
    .cover-sub{font-size:16px;color:#57606a}
    .cover-meta{margin-top:3mm;font-size:12px;color:#7C4DFF;font-weight:600}
    .cover-hr{width:60mm;border:none;border-top:3px solid #7C4DFF;margin:6mm auto}
    .cover-h2{font-size:15px;color:#7C4DFF;text-transform:uppercase;letter-spacing:1px;margin-bottom:3mm}
    .cover-p{font-size:12px;line-height:1.55;color:#24292f;text-align:justify;max-width:230mm}
    .cover-toc{display:grid;grid-template-columns:1fr 1fr;gap:1mm 12mm;width:100%;max-width:215mm;margin-top:4mm;text-align:left}
    .cover-toc a{display:flex;justify-content:space-between;align-items:baseline;gap:8mm;text-decoration:none;color:#24292f;font-size:11px;padding:1mm 2mm;border-bottom:1px dotted #c9ccd1}
    .cover-toc a .pg{color:#7C4DFF;font-weight:700;font-size:11px}
    .cover-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8mm;width:100%;margin-top:5mm}
    .cover-stat{border:1px solid #c9ccd1;border-radius:8px;padding:5mm 3mm}
    .cover-stat .v{font-size:20px;font-weight:700;color:#1b1f24}
    .cover-stat .k{font-size:10px;color:#57606a;margin-top:1mm}
    .cover-author{margin-top:8mm;font-size:13px;color:#1b1f24}
    .cover-author b{color:#7C4DFF}
    .tabs{display:none}
    .scenario-grid{display:grid !important;grid-template-columns:repeat(5,1fr) !important;break-before:page}
    .scenario-grid .card{min-width:0}
    .comp{grid-template-columns:repeat(3,1fr) !important}
    .safety-grid{break-before:page;grid-template-columns:2fr 1fr 1fr !important}
    .card,.temporal,.mcard,.safety-img img,.comp .card{break-inside:avoid}
    .safety-img img{min-width:0}
    table{background:#fff}
    th{background:#f0f3f6;color:#333}
  }
</style>
</head>
<body>
  %%COVER%%
  <h1>Resultados — Expansión Urbana Mérida 2026–2030</h1>
  <div class="sub">Pipeline v2.0 · LightGBM + CA de reglas aprendidas + variables kársticas · Generado el %%STAMP%%</div>
  <div class="meta">
    <div class="mcard"><div class="v">%%AUC%%</div><div class="k">AUC-ROC transición</div></div>
    <div class="mcard"><div class="v">%%FOM%%</div><div class="k">FOM transición</div></div>
    <div class="mcard"><div class="v">%%CA_AUC%%</div><div class="k">AUC-ROC reglas CA</div></div>
    <div class="mcard"><div class="v">%%FEAT%%</div><div class="k">features (7 base + 3 kársticas)</div></div>
  </div>

  <h2 id="sec-areas">Área urbana proyectada (km²)</h2>
  %%TABLE%%

  <h2 id="sec-espacial">Calidad espacial de la expansión (2030)</h2>
  %%SPATIAL%%
  <div class="sub">Mayor área media por parche y menor borde por km² = expansión más compacta.
  El borde por km² tiende a ser mayor en áreas urbanas más pequeñas (efecto de escala).</div>

  %%AMBIENT%%

  <h2 id="sec-mapas">Mapas de probabilidad por escenario y año</h2>
  <div class="legend"><span>Probabilidad:</span><span class="bar"></span><span>0.2 → 1.0</span></div>
  <div class="tabs">%%TABS%%</div>
  %%GRIDS%%

  <h2 id="sec-comp">Comparación 2024 → 2030 por escenario</h2>
  <div class="comp">%%COMPS%%</div>

  %%SAFETY%%

  <footer>
    Archivos de origen: <code>results/maps/</code> (GeoTIFFs y PNG) y <code>results/reports/</code>.
    Regenerar: <code>python frontend/generar_dashboard.py</code>.
  </footer>
<script>
  const tabs = document.querySelectorAll(".tab");
  const grids = document.querySelectorAll(".scenario-grid");
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    grids.forEach(g => g.classList.remove("active"));
    t.classList.add("active");
    document.getElementById("grid-" + t.dataset.scenario).classList.add("active");
  }));
  if (tabs.length) tabs[0].click();
</script>
</body>
</html>
"""

    html = (html.replace("%%STAMP%%", stamp)
                .replace("%%AUC%%", str(metrics.get("auc_roc", "N/A")))
                .replace("%%FOM%%", str(metrics.get("fom", "N/A")))
                .replace("%%CA_AUC%%", str(metrics.get("ca_auc_roc", "N/A")))
                .replace("%%FEAT%%", str(metrics.get("n_features", "?")))
                .replace("%%TABLE%%", table)
                .replace("%%SPATIAL%%", spatial_table)
                .replace("%%AMBIENT%%", ambient)
                .replace("%%TABS%%", "".join(tabs))
                .replace("%%GRIDS%%", "".join(grids))
                .replace("%%COMPS%%", comps)
                .replace("%%SAFETY%%", safety)
                .replace("%%COVER%%", cover))

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard generado: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

    # ── Exportación PDF (Chrome headless) + pie de página ──
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        print("Chrome no encontrado; se omitió el PDF (solo HTML).")
        return
    pdf = ROOT / "frontend" / "dashboard_resultados.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    OUT.as_uri()], check=True)
    from pie_paginas import pie_de_pagina
    secciones = pie_de_pagina(pdf, titulo="Predicción de la Expansión Urbana de Mérida",
                              markers=markers, altura_mm=7)
    # Verificación: la página del índice debe tener la sección esperada
    ok = True
    for sid in toc_pages:
        if sid not in toc_items_by_id:
            continue
        sec_esperada = toc_items_by_id[sid][1]
        if secciones.get(toc_pages[sid]) != sec_esperada:
            ok = False
            print(f"  AVISO: {sid} esperado en p{toc_pages[sid]} como '{sec_esperada}', "
                  f"real: p{toc_pages[sid]}='{secciones.get(toc_pages[sid])}'")
    estado = "índice verificado" if ok else "índice DESCUADRADO"
    print(f"PDF generado: {pdf} ({pdf.stat().st_size / 1024:.0f} KB) — {estado}")


if __name__ == "__main__":
    main()
