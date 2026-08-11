"""frontend/pie_paginas.py — Pie de página (sección + numeración) para los PDFs.

Chrome headless no soporta `counter(page)` ni `string()` en elementos fixed
(imprimen 0/vacío), así que tras la exportación se superpone el pie con
reportlab + pypdf. Las secciones se mapean página a página buscando los
títulos (marcadores) en el texto extraído de cada página, de modo que el pie
es correcto aunque la paginación cambie.

Uso:
    from pie_paginas import pie_de_pagina
    pie_de_pagina("frontend/reporte_tecnico.pdf", titulo="Reporte Técnico",
                  markers=[("Reporte de Predicción", "Pipeline v2.0"), ...])
"""
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

GRIS = (0.34, 0.36, 0.42)


def mapear_secciones(pdf_path, markers, primera="Portada"):
    """Devuelve {página: nombre_de_sección} buscando cada marcador en el texto.

    markers: lista de (texto_a_buscar, nombre_seccion) en orden de aparición.
    La página 1 se considera `primera` (portada) y no se escanea.
    """
    reader = PdfReader(str(pdf_path))
    secciones = {}
    current = primera
    for i, page in enumerate(reader.pages, 1):
        if i > 1:
            txt = (page.extract_text() or "").replace("\n", " ")
            for marker, name in markers:
                if marker in txt:
                    current = name
                    break
        secciones[i] = current
    return secciones


def pie_de_pagina(pdf_path, titulo, markers, primera="Portada",
                  izquierda_mm=14, derecha_mm=14, altura_mm=9, tamano=7.5):
    """Superpone en cada página: [título] [sección] [página N de M].

    Modifica el PDF in situ (regenera el archivo). `markers` se usa para
    mapear cada página a su sección (ver mapear_secciones).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    secciones = mapear_secciones(pdf_path, markers, primera=primera)
    mm = 72 / 25.4

    buf = io.BytesIO()
    overlay = canvas.Canvas(buf)
    for i, page in enumerate(reader.pages, 1):
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        overlay.setPageSize((w, h))
        y = altura_mm * mm
        overlay.setStrokeColorRGB(*GRIS)
        overlay.setLineWidth(0.4)
        overlay.line(izquierda_mm * mm, y + 2.2 * mm, w - derecha_mm * mm, y + 2.2 * mm)
        overlay.setFillColorRGB(*GRIS)
        overlay.setFont("Helvetica", tamano)
        overlay.drawString(izquierda_mm * mm, y, titulo)
        overlay.drawCentredString(w / 2, y, secciones.get(i, ""))
        overlay.drawRightString(w - derecha_mm * mm, y, f"Página {i} de {total}")
        overlay.showPage()
    overlay.save()

    writer = PdfWriter()
    buf.seek(0)
    overlay_reader = PdfReader(buf)
    for i, page in enumerate(reader.pages):
        page.merge_page(overlay_reader.pages[i])
        writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return secciones
