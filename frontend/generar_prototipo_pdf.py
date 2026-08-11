"""frontend/generar_prototipo_pdf.py — PDF imprimible del prototipo 3D interactivo.

Envuelve `frontend/merida_urban_expansion_3d_comparison.html` (un fragmento que
depende de variables CSS externas) en un documento completo con tema claro de
impresión, portada con resumen ejecutivo, las 4 secciones visibles (Vista 3D,
Indicadores, Modelo, Planificador IA) y pie de página con sección + numeración.

Requiere: Chrome instalado (para la exportación) y el flujo de `pie_paginas.py`.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "frontend" / "merida_urban_expansion_3d_comparison.html"
OUT = ROOT / "frontend" / "prototipo_3d.pdf"

# Datos del prototipo (2030 = último año, divergencia máxima entre escenarios)
S = {
    "area": {"no": "1017", "si": "958", "ai": "911"},
    "frag": {"no": "0.67", "si": "0.26", "ai": "0.20"},
    "verde": {"no": "5", "si": "23", "ai": "29"},
    "lst": {"no": "36.7", "si": "33.5", "ai": "32.0"},
    "karst": {"no": "0.68", "si": "0.36", "ai": "0.24"},
    "vida": {"no": "53", "si": "80", "ai": "87"},
}

COVER = """
<div class="cover">
  <div>
    <div class="cover-band">Universidad Politécnica de Yucatán · TSU en Ciencia de Datos</div>
    <div class="cover-title">Simulador interactivo de expansión urbana</div>
    <div class="cover-sub">Mérida, Yucatán · 2024 → 2030 · 30 m de resolución</div>
    <hr class="cover-hr">
    <div class="cover-h2">Resumen ejecutivo</div>
    <p class="cover-p">
      Este prototipo interactivo simula la expansión de la mancha urbana de la Zona
      Metropolitana de Mérida con tres escenarios comparativos: <b>sin planificación</b>
      (CA + LightGBM sin restricciones), <b>planificación tradicional</b> (misma IA con
      corredores verdes obligatorios y exclusión de cenotes) y <b>gestión por IA</b>
      (variables kársticas LST · acuífero · cenotes + CA de reglas aprendidas y
      optimización multiobjetivo). A 2030, la gestión por IA evita <b>106 km²</b> de
      expansión frente al escenario sin plan (911 vs 1017 km²), reduce la fragmentación
      de 0.67 a 0.20, multiplica la cobertura verde por ~6 (5% → 29%), baja la
      temperatura superficial 4.7 °C (36.7 → 32.0 °C) y reduce la vulnerabilidad del
      acuífero kárstico de 0.68 a 0.24.
    </p>
    <p class="cover-p">
      Las dos contribuciones originales del modelo —<b>variables kársticas</b> y
      <b>CA de reglas aprendidas</b> por un segundo LightGBM sobre el estado de
      vecindad— son específicas de Mérida y están ausentes en los trabajos previos
      para la ZMM.
    </p>
    <div class="cover-stats">
      <div class="cover-stat"><div class="v">911 <span class="u">km²</span></div><div class="k">Área urbana 2030 · gestión IA (vs 1017 sin plan)</div></div>
      <div class="cover-stat"><div class="v">0.20</div><div class="k">Fragmentación 2030 · IA (vs 0.67 sin plan)</div></div>
      <div class="cover-stat"><div class="v">−4.7 °C</div><div class="k">LST 2030 · IA vs sin planificación</div></div>
      <div class="cover-stat"><div class="v">29%</div><div class="k">Cobertura verde 2030 · IA (vs 5% sin plan)</div></div>
    </div>
  </div>
  <div class="cover-author">Autor: <b>Héctor Javier Raya Romo</b> · Mérida, Yucatán · %DATE%</div>
</div>
"""

# CSS: variables que el fragmento espera (tema claro para impresión)
VARS = """
  :root{
    --font-sans:'Segoe UI',system-ui,-apple-system,sans-serif;
    --color-background-primary:#ffffff;
    --color-background-secondary:#f6f5f1;
    --color-text-primary:#1b1f24;
    --color-text-secondary:#57606a;
    --color-text-tertiary:#8b949e;
    --color-border-primary:#d0d3d8;
    --color-border-secondary:#d8dbe0;
    --color-border-tertiary:#e5e7ea;
    --color-border-info:#7C4DFF;
    --border-radius-md:8px;
  }
"""

PRINT_CSS = """
  @page{size:A4 landscape;margin:10mm}
  @media print{
    *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    body{font-family:var(--font-sans);background:#fff;color:#1b1f24;font-size:11px}
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
    .tabs{display:none !important}
    h2.sr-only{display:none !important}
    .panel{display:block !important;break-before:page;padding:4mm 0}
    #tab-view3d{break-before:auto}
    .sec-title{font-size:16px;font-weight:700;color:#7C4DFF;margin:0 0 4mm;border-bottom:2px solid #7C4DFF;padding-bottom:2mm}
    .canvas-row{grid-template-columns:repeat(3,1fr);gap:6mm}
    canvas{height:160px;border-radius:6px}
    .time-ctrl,.info-banner{break-inside:avoid}
    .metrics-grid{grid-template-columns:repeat(4,1fr);break-inside:avoid}
    .charts-row{grid-template-columns:repeat(2,1fr);gap:6mm}
    .chart-card{break-inside:avoid}
    .param-row input[type=range],.obj-check,.btn-run,#ai-result-box,#ai-status{display:none !important}
    #tab-model svg{width:170mm;height:auto;display:block;margin:4mm auto;break-inside:avoid}
    .param-row .param-val{min-width:auto}
    .params-grid{grid-template-columns:1fr 1fr;break-inside:avoid}
    .legend-row{margin-top:5mm}
  }
"""


def main():
    if not FRAG.exists():
        sys.exit(f"No existe el fragmento: {FRAG}")
    frag = FRAG.read_text(encoding="utf-8")

    date = datetime.now().strftime("%d de agosto de 2026")
    cover = COVER.replace("%DATE%", date)

    # Inyecta títulos de sección visibles solo en impresión
    titles = {
        '<div id="tab-view3d" class="panel active">':
            '<div class="sec-title">1 · Vista 3D — expansión comparativa a 2030</div>',
        '<div id="tab-indicators" class="panel">':
            '<div class="sec-title">2 · Indicadores 2024–2030 por escenario</div>',
        '<div id="tab-ai-planner" class="panel">':
            '<div class="sec-title">3 · Planificador urbano gestionado por IA — parámetros por defecto</div>',
        '<div id="tab-model" class="panel">':
            '<div class="sec-title">4 · Arquitectura del modelo (CA + LightGBM + variables kársticas)</div>',
    }
    for needle, replacement in titles.items():
        frag = frag.replace(needle, needle + replacement, 1)

    # Script: sitúa el año en 2030 (divergencia máxima), dibuja 3D y charts
    extra_js = """
<script>
  // Para impresión: paneles visibles (Chart.js mide 0x0 en display:none), año 2030
  document.querySelectorAll('.panel').forEach(p => p.style.display = 'block');
  const slider = document.getElementById('year-slider');
  if (slider) { slider.value = 6; updateYear(6); }
  function tryCharts(attempt) {
    if (window.Chart) {
      try { initCharts(); } catch (e) { console.error(e); }
      return;
    }
    if (attempt < 20) setTimeout(() => tryCharts(attempt + 1), 400);
  }
  window.addEventListener('load', () => {
    const slider2 = document.getElementById('year-slider');
    if (slider2) { slider2.value = 6; updateYear(6); }
    tryCharts(0);
    setTimeout(() => { drawAll(); tryCharts(0); }, 400);
  });
</script>
"""
    frag = frag.replace("</body>", extra_js + "</body>") if "</body>" in frag else frag + extra_js

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Simulador de expansión urbana de Mérida — versión imprimible</title>
<style>
{VARS}
{PRINT_CSS}
</style>
</head>
<body>
{cover}
{frag}
</body>
</html>
"""

    tmp = ROOT / "frontend" / "prototipo_3d_tmp.html"
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
        OUT, titulo="Simulador de expansión urbana de Mérida",
        markers=[
            ("Vista 3D", "1 · Vista 3D"),
            ("Indicadores 2024", "2 · Indicadores"),
            ("Planificador urbano gestionado", "3 · Planificador IA"),
            ("Arquitectura del modelo", "4 · Arquitectura del modelo"),
        ],
        altura_mm=7,
    )
    print(f"PDF generado: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
