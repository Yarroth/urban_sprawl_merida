"""frontend/generar_dashboard_integrado_pdf.py — PDF imprimible del prototipo A.

Convierte `frontend/merida_combined_dashboard.html` (dashboard integrado con 9
secciones: resumen, vista 3D, 3 escenarios, inversión 1,500 MDP, indicadores,
planificador IA, propuestas ciudadanas, marco legal y comparativa final) en un
PDF A4 apaisado con portada, tema claro de impresión y pie de página.

El documento original define su propio tema oscuro (:root), así que este
generador inyecta (a) un bloque @media print que sobrescribe las variables a un
tema claro y muestra todos los paneles, (b) una portada con resumen ejecutivo,
y (c) un script que fuerza la visibilidad de los paneles, sitúa el año en 2030,
inicializa los charts de Chart.js (CDN) y dibuja los canvases 3D antes de
exportar con Chrome headless.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "frontend" / "merida_combined_dashboard.html"
OUT = ROOT / "frontend" / "dashboard_integrado.pdf"

COVER = """
<div class="cover">
  <div>
    <div class="cover-band">Universidad Politécnica de Yucatán · TSU en Ciencia de Datos</div>
    <div class="cover-title">Dashboard Integrado — Expansión Urbana de Mérida</div>
    <div class="cover-sub">CA + LightGBM · Inversión 1,500 MDP · Yucatán 2025–2030</div>
    <hr class="cover-hr">
    <div class="cover-h2">Resumen ejecutivo</div>
    <p class="cover-p">
      El <b>Escenario C</b> (inversión estratégica con jerarquía de movilidad: BRT,
      ciclovías, peatones y protección kárstica) es la opción fundamentada para la
      Zona Metropolitana de Mérida: expande el área urbana clasificada a <b>16.4 km²</b>
      en 2030 frente a <b>17.4 km²</b> del escenario sin plan (+1.8 vs +2.7 km²),
      urbaniza <b>36% menos celdas</b> (1 953 vs 3 047), es <b>40% más compacta</b>
      (144 vs 242 parches) y su expansión aterriza <b>1.7 °C más fresca</b>
      (LST 29.5 vs 31.2 °C) y <b>178 m más lejos de los cenotes</b> (3.1 vs 2.9 km).
    </p>
    <p class="cover-p">
      La decisión no es preferencia política: es <b>requisito constitucional y
      legal</b> (Arts. 1º y 4º, Ley de Movilidad de Yucatán Art. 39, NOM-004-SEDATU,
      jurisprudencia SCJN) y recoge las <b>8 propuestas del Colectivo Haciendo
      Ciudad</b>. Los 1,500 MDP del Gobierno de Yucatán se asignan con prioridad a
      peatones (300), ciclistas (250), transporte público (550) y protección del
      acuífero (200).
    </p>
    <div class="cover-stats">
      <div class="cover-stat"><div class="v">−36%</div><div class="k">Celdas urbanizadas · C vs A (1 953 vs 3 047)</div></div>
      <div class="cover-stat"><div class="v">144 <span class="u">parches</span></div><div class="k">Compactación 2030 · vs 242 sin plan</div></div>
      <div class="cover-stat"><div class="v">−1.7 °C</div><div class="k">LST de la expansión · C vs A (29.5 vs 31.2)</div></div>
      <div class="cover-stat"><div class="v">+178 <span class="u">m</span></div><div class="k">Más lejos de cenotes · C vs A</div></div>
    </div>
  </div>
  <div class="cover-author">Autor: <b>Héctor Javier Raya Romo</b> · Mérida, Yucatán · %DATE%</div>
</div>
"""

# Bloque @media print: tema claro (sobrescribe el :root oscuro del documento)
PRINT_CSS = """
  @page{size:A4 landscape;margin:10mm}
  @media print{
    *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    :root{
      --bg0:#ffffff;--bg1:#f7f6f2;--bg2:#ffffff;--bg3:#f0f0ec;
      --border:rgba(20,24,40,.12);--border-hi:rgba(20,24,40,.2);
      --text-1:#1b1f24;--text-2:#57606a;--text-3:#8b949e;
      --c-red:#c0392b;--c-green:#1a7f4d;--c-purple:#7C4DFF;
      --c-cyan:#0e7490;--c-amber:#b45309;
    }
    body{background:#fff;color:#1b1f24}
    body::before{display:none}
    .header{display:none}
    .tabs{display:none !important}
    .panel{display:block !important;break-before:page;padding:4mm 0}
    #tab-resumen{break-before:auto}
    body{font-size:10px}
    table{font-size:9px}
    .section-title{font-size:12px;margin-bottom:3mm}
    p{font-size:10px}
    .data-card,.benefit-card{padding:3mm}
    .data-card-value{font-size:17px}
    .data-card-sub{font-size:9px}
    .investment-item{padding:2.5mm}
    .legal-box{padding:3mm}
    .scenario-row{padding:1.5mm 0}
    .metric-card{padding:2.5mm}
    .metric-name{font-size:9px}
    .comparison-cell{font-size:9px;padding:2mm}
    .sec-num{font-size:15px;font-weight:700;color:#7C4DFF;border-bottom:2px solid #7C4DFF;
             padding-bottom:2mm;margin-bottom:4mm;letter-spacing:.02em}
    .cover{display:flex;flex-direction:column;justify-content:space-between;align-items:center;
           text-align:center;break-after:page;height:188mm;padding:8mm 18mm}
    .cover-band{font-size:12px;letter-spacing:2px;color:#57606a;text-transform:uppercase}
    .cover-title{font-size:32px;font-weight:800;color:#1b1f24;margin:6mm 0 2mm}
    .cover-sub{font-size:15px;color:#57606a}
    .cover-hr{width:60mm;border:none;border-top:3px solid #7C4DFF;margin:6mm auto}
    .cover-h2{font-size:14px;color:#7C4DFF;text-transform:uppercase;letter-spacing:1px;margin-bottom:3mm}
    .cover-p{font-size:11.5px;line-height:1.55;color:#24292f;text-align:justify;max-width:250mm}
    .cover-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:7mm;width:100%;margin-top:6mm}
    .cover-stat{border:1px solid #c9ccd1;border-radius:8px;padding:5mm 3mm}
    .cover-stat .v{font-size:22px;font-weight:700;color:#1b1f24}
    .cover-stat .v .u{font-size:12px;font-weight:600;color:#57606a}
    .cover-stat .k{font-size:9.5px;color:#57606a;margin-top:1.5mm;line-height:1.4}
    .cover-author{margin-top:8mm;font-size:13px;color:#1b1f24}
    .cover-author b{color:#7C4DFF}
    .view-layout{grid-template-columns:120px 1fr !important}
    .metrics-stack{grid-column:1/-1 !important;display:grid !important;grid-template-columns:repeat(3,1fr) !important;gap:3mm}
    .city-canvas{height:115px}
    .ctrl-panel{padding:2mm}
    .cities-grid{gap:4mm}
    .chart-card{padding:6px}
    .chart-area{height:95px}
    .charts-grid{grid-template-columns:repeat(3,1fr) !important}
    .data-grid{grid-template-columns:repeat(4,1fr) !important}
    .chart-card,.data-grid,.metric-card,.legal-box,.investment-item,.comparison-table,
    .benefit-card,.scenario-row{break-inside:avoid}
    .charts-grid{grid-template-columns:repeat(2,1fr) !important}
    .data-grid{grid-template-columns:repeat(4,1fr) !important}
    table{background:#fff}
    th{background:#f0f3f6;color:#333}
    .custom-slider,.param-slider,.run-btn,.ctrl-group input[type=range]{display:none !important}
  }
"""


def main():
    if not FRAG.exists():
        sys.exit(f"No existe el prototipo: {FRAG}")
    html = FRAG.read_text(encoding="utf-8")

    date = datetime.now().strftime("%d de agosto de 2026")
    cover = COVER.replace("%DATE%", date)

    # Encabezado de sección impreso (numerado) al inicio de cada panel
    TITLES = [
        ("tab-resumen", "1 · Contexto, hallazgo y metodología"),
        ("tab-view3d", "2 · Vista 3D — expansión comparativa a 2030"),
        ("tab-escenarios", "3 · Tres escenarios de política pública"),
        ("tab-inversion", "4 · Asignación estratégica de 1,500 MDP"),
        ("tab-indicadores", "5 · Indicadores de sostenibilidad y series 2024–2030"),
        ("tab-ai-planner", "6 · Planificador IA — parámetros del modelo CA"),
        ("tab-colectivo", "7 · Propuestas del Colectivo Haciendo Ciudad"),
        ("tab-marco-legal", "8 · Marco legal obligatorio"),
        ("tab-comparativa", "9 · Comparativa final y recomendación"),
    ]
    for pid, title in TITLES:
        needle = f'<div id="{pid}" class="panel">'
        if needle in html:
            html = html.replace(needle, needle + f'<div class="sec-num">{title}</div>', 1)

    # Script: paneles visibles (Chart.js mide 0x0 en display:none), año 2030
    extra_js = """
<script>
  document.querySelectorAll('.panel').forEach(p => p.style.display = 'block');
  const slider = document.getElementById('year-slider');
  if (slider) { slider.value = 5; updateYear(5); }  // 2030 (último índice, YEARS=6)
  function tryCharts(attempt) {
    if (window.Chart) {
      try { initCharts(); } catch (e) { console.error(e); }
      return;
    }
    if (attempt < 20) setTimeout(() => tryCharts(attempt + 1), 400);
  }
  window.addEventListener('load', () => {
    const s2 = document.getElementById('year-slider');
    if (s2) { s2.value = 5; updateYear(5); }
    tryCharts(0);
    setTimeout(() => { try { drawAll(); } catch (e) {} tryCharts(0); }, 400);
  });
</script>
"""
    html = html.replace("</body>", extra_js + "</body>", 1)

    # Inyecta portada tras <body> y print CSS antes de </head>
    html = html.replace("<body>", "<body>\n" + cover + "\n", 1)
    html = html.replace("</head>", "<style>\n" + PRINT_CSS + "\n</style>\n</head>", 1)

    tmp = ROOT / "frontend" / "dashboard_integrado_tmp.html"
    tmp.write_text(html, encoding="utf-8")

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        sys.exit("Chrome no encontrado; no se pudo exportar el PDF.")

    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=8000",
         f"--print-to-pdf={OUT}", tmp.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    tmp.unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT / "frontend"))
    from pie_paginas import pie_de_pagina
    pie_de_pagina(
        OUT, titulo="Dashboard Integrado — Expansión Urbana de Mérida",
        markers=[
            ("Contexto del Proyecto", "1 · Resumen"),
            ("2 · Vista 3D", "2 · Vista 3D"),
            ("3 · Tres escenarios", "3 · Escenarios"),
            ("4 · Asignación estratégica", "4 · Inversión 1,500 MDP"),
            ("5 · Indicadores de sostenibilidad", "5 · Indicadores"),
            ("6 · Planificador IA", "6 · Planificador IA"),
            ("7 · Propuestas del Colectivo", "7 · Propuestas ciudadanas"),
            ("8 · Marco legal obligatorio", "8 · Marco legal"),
            ("9 · Comparativa final y recomendación", "9 · Comparativa final"),
        ],
        altura_mm=7,
    )
    print(f"PDF generado: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
