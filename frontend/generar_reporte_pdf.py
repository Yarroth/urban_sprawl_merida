"""frontend/generar_reporte_pdf.py — Genera el PDF del reporte técnico.

Convierte los reportes markdown del pipeline a un HTML autocontenido (imágenes
embebidas en base64) y lo exporta con Chrome headless a:

    frontend/reporte_tecnico.pdf   (A4 vertical, tema claro)

Documentos incluidos (en orden):
  1. results/reports/final_report.md       — reporte del pipeline (v2.0)
  2. results/reports/calibracion_tasas.md  — calibración empírica de tasas
  3. results/safety/seguridad_peatonal.md  — seguridad peatonal del Periférico

Uso:
    python frontend/generar_reporte_pdf.py
"""
import base64
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONT = ROOT / "frontend"
OUT = FRONT / "reporte_tecnico.pdf"
DOCS = [
    (ROOT / "results" / "reports" / "final_report.md",      "Reporte del pipeline (v2.0)"),
    (ROOT / "results" / "reports" / "calibracion_tasas.md", "Calibración empírica de tasas"),
    (ROOT / "results" / "safety" / "seguridad_peatonal.md", "Seguridad peatonal — Periférico"),
]

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
         "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def inline(text, base_dir=None):
    """Convierte el markdown en línea (negritas, cursivas, código, links, imágenes)."""
    # Imágenes ![alt](ruta) → base64 (ruta relativa al .md)
    def _img(m):
        alt, src = m.group(1), m.group(2)
        p = Path(src)
        if not p.exists() and base_dir:
            p = Path(base_dir) / src
        if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            b = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{b}" alt="{alt}">'
        return f'<img src="{src}" alt="{alt}">'
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img, text)
    # Links [texto](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # Código `...`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Negritas
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # Cursivas (evita coger parte de las negritas ya convertidas)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def md_to_html(md_text, base_dir):
    """Convierte un documento markdown a HTML (bloques simples)."""
    lines = md_text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Tabla: fila con | seguida de separador ---
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            thead = "".join(f"<th>{inline(c, base_dir)}</th>" for c in header)
            trows = "".join(
                "<tr>" + "".join(f"<td>{inline(c, base_dir)}</td>" for c in row) + "</tr>"
                for row in rows)
            out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table>")
            continue

        # Encabezados
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2), base_dir)}</h{level}>")
            i += 1
            continue

        # Regla horizontal
        if re.match(r"^-{3,}$", stripped):
            out.append("<hr>")
            i += 1
            continue

        # Bloque de cita
        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(inline(lines[i].strip().lstrip(">").strip(), base_dir))
                i += 1
            out.append("<blockquote>" + "<br>".join(quote) + "</blockquote>")
            continue

        # Listas no ordenadas
        if re.match(r"^\s*[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i].strip()):
                items.append(f"<li>{inline(lines[i].strip()[2:].strip(), base_dir)}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        # Listas ordenadas
        if re.match(r"^\s*\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i].strip()):
                txt = re.sub(r"^\s*\d+\.\s+", "", lines[i].strip())
                items.append(f"<li>{inline(txt, base_dir)}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue

        # Párrafo normal (acumula líneas consecutivas)
        para = [inline(lines[i].strip(), base_dir)]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|>\s|[-*]\s|\d+\.\s|\s*\|)", lines[i]):
            para.append(" " + inline(lines[i].strip(), base_dir))
            i += 1
        out.append(f"<p>{''.join(para)}</p>")
    return "\n".join(out)


def main():
    hoy = datetime.date.today()
    fecha_es = f"{hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    fecha_iso = hoy.strftime("%Y-%m-%d")

    secciones = []
    for idx, (path, titulo) in enumerate(DOCS):
        html = md_to_html(path.read_text(encoding="utf-8"), path.parent)
        cls = 'doc first' if idx == 0 else 'doc'
        # Las imágenes con ruta relativa se resuelven contra el directorio del .md
        secciones.append(f'<div class="{cls}"><h1 class="doc-title">{titulo}</h1>{html}</div>')

    toc = "".join(f"<li>{t}</li>" for _, t in DOCS)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte Técnico — Expansión Urbana Mérida</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1b1f24;font-size:11px;background:#fff}}
  @page{{size:A4 portrait;margin:14mm}}
  .cover{{display:flex;flex-direction:column;justify-content:space-between;align-items:center;text-align:center;height:260mm;padding:10mm 16mm;break-after:page}}
  .cover-band{{font-size:11px;letter-spacing:2px;color:#57606a;text-transform:uppercase}}
  .cover-title{{font-size:30px;font-weight:800;margin:8mm 0 2mm}}
  .cover-sub{{font-size:15px;color:#57606a}}
  .cover-meta{{margin-top:4mm;font-size:12px;color:#7C4DFF;font-weight:600}}
  .cover-hr{{width:60mm;border:none;border-top:3px solid #7C4DFF;margin:8mm auto}}
  .cover-toc{{list-style:none;font-size:13px;line-height:2.2;color:#24292f}}
  .cover-toc b{{color:#7C4DFF}}
  .cover-author{{margin-top:8mm;font-size:13px}}
  .cover-author b{{color:#7C4DFF}}
  .doc{{break-before:page}}
  .doc.first{{break-before:auto}}
  h1.doc-title{{font-size:20px;border-left:5px solid #7C4DFF;padding-left:10px;margin-bottom:6mm;break-after:avoid}}
  h2{{font-size:15px;color:#7C4DFF;margin:7mm 0 3mm;break-after:avoid}}
  h3{{font-size:13px;margin:5mm 0 2mm;break-after:avoid}}
  h4,h5,h6{{font-size:12px;margin:4mm 0 2mm}}
  p{{margin:2.2mm 0;line-height:1.55;text-align:justify}}
  table{{width:100%;border-collapse:collapse;margin:3mm 0;font-size:10.5px;break-inside:avoid}}
  th,td{{padding:1.6mm 2.4mm;text-align:left;border:1px solid #c9ccd1}}
  th{{background:#f0f3f6;font-weight:600}}
  td{{vertical-align:top}}
  ul,ol{{margin:2mm 0 2mm 6mm}}
  li{{margin:1mm 0;line-height:1.5}}
  blockquote{{border-left:3px solid #7C4DFF;background:#f6f8fa;padding:3mm 4mm;margin:3mm 0;border-radius:4px;color:#57606a}}
  code{{background:#f0f3f6;border:1px solid #e2e5e9;border-radius:4px;padding:0.4mm 1.2mm;font-size:10px}}
  img{{max-width:72%;height:auto;display:block;margin:5mm auto;border:1px solid #c9ccd1;border-radius:6px;break-inside:avoid}}
  hr{{border:none;border-top:1px solid #c9ccd1;margin:5mm 0}}
  a{{color:#7C4DFF}}
  .nota{{margin-top:8mm;font-size:10px;color:#57606a}}
</style>
</head>
<body>
  <div class="cover">
    <div class="cover-band">Universidad Politécnica de Yucatán · TSU en Ciencia de Datos</div>
    <div>
      <h1 class="cover-title">Reporte Técnico</h1>
      <div class="cover-sub">Predicción de la Expansión Urbana de Mérida 2026–2030 y seguridad peatonal del Anillo Periférico</div>
      <div class="cover-meta">Pipeline v2.0 · LightGBM + Autómata Celular de reglas aprendidas + variables kársticas</div>
      <hr class="cover-hr">
      <ul class="cover-toc">
        {toc}
      </ul>
    </div>
    <div class="cover-author"><b>Héctor Javier Raya Romo</b> · Mérida, Yucatán · {fecha_es}</div>
  </div>
  {''.join(secciones)}
</body>
</html>"""

    tmp_html = FRONT / "reporte_tecnico_tmp.html"
    tmp_html.write_text(html, encoding="utf-8")
    print(f"HTML temporal: {tmp_html} ({tmp_html.stat().st_size / 1024:.0f} KB)")

    chrome = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).exists():
        sys.exit("Chrome no encontrado en /Applications")
    subprocess.run([
        chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={OUT}", tmp_html.as_uri(),
    ], check=True)
    tmp_html.unlink(missing_ok=True)
    print(f"PDF generado: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
