"""Genera frontend/dashboard_resultados.html a partir de results/maps y results/reports.

Uso:
    python frontend/generar_dashboard.py

Requisitos: numpy, pandas (y los PNG generados por scripts/05_visualize.py).
"""
import base64
import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAPS, REP = ROOT / "results" / "maps", ROOT / "results" / "reports"
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
  .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .card img{width:100%;display:block;aspect-ratio:1/0.75;object-fit:cover}
  .cap{padding:8px 10px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
  .km2{color:var(--text);font-weight:600}
  .comp{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:13px}
  th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--border)}
  th:first-child,td:first-child{text-align:left}
  th{background:#1F2937;color:var(--muted);font-weight:600}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
  .delta{color:#7EE787;font-weight:600}
  .legend{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;margin:10px 0}
  .bar{width:180px;height:10px;border-radius:5px;background:linear-gradient(90deg,#FFFFB2,#FCDE00,#E53935,#7A0000)}
  footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}
  @media (max-width:900px){.scenario-grid,.comp{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
  <h1>Resultados — Expansión Urbana Mérida 2026–2030</h1>
  <div class="sub">Pipeline v2.0 · LightGBM + CA de reglas aprendidas + variables kársticas · Generado el %%STAMP%%</div>
  <div class="meta">
    <div class="mcard"><div class="v">%%AUC%%</div><div class="k">AUC-ROC transición</div></div>
    <div class="mcard"><div class="v">%%FOM%%</div><div class="k">FOM transición</div></div>
    <div class="mcard"><div class="v">%%CA_AUC%%</div><div class="k">AUC-ROC reglas CA</div></div>
    <div class="mcard"><div class="v">%%FEAT%%</div><div class="k">features (7 base + 3 kársticas)</div></div>
  </div>

  <h2>Área urbana proyectada (km²)</h2>
  %%TABLE%%

  <h2>Mapas de probabilidad por escenario y año</h2>
  <div class="legend"><span>Probabilidad:</span><span class="bar"></span><span>0.2 → 1.0</span></div>
  <div class="tabs">%%TABS%%</div>
  %%GRIDS%%

  <h2>Comparación 2024 → 2030 por escenario</h2>
  <div class="comp">%%COMPS%%</div>

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
                .replace("%%TABS%%", "".join(tabs))
                .replace("%%GRIDS%%", "".join(grids))
                .replace("%%COMPS%%", comps))

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard generado: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
