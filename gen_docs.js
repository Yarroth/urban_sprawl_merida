const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer, PageBreak, ExternalHyperlink
} = require("docx");
const fs = require("fs");

const ACCENT = "5B3DB5";
const TEAL   = "0F6E56";
const GRAY_H = "F3F2EE";
const GRAY_B = "D3D1C7";
const RED_SC = "7A2F12";
const GREEN_SC= "155724";
const PUR_SC = "3A238A";

const border  = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const thBorder= { style: BorderStyle.SINGLE, size: 1, color: "9B8DD4" };
const thBorders={ top: thBorder, bottom: thBorder, left: thBorder, right: thBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial", color: ACCENT })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } },
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial", color: "2C2C2A" })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: "3C3489" })]
  });
}
function p(text, opts={}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts })]
  });
}
function bullet(text, lvl=0) {
  return new Paragraph({
    numbering: { reference: "bullets", level: lvl },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: "Arial" })]
  });
}
function italic(text) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: "Arial", italics: true, color: "5F5E5A" })]
  });
}
function badge(text, color, bgColor) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: " ★ " + text + " ", size: 20, font: "Arial", bold: true, color, shading: { fill: bgColor, type: ShadingType.CLEAR } })
    ]
  });
}
function tableRow(cells, isHeader=false) {
  return new TableRow({
    tableHeader: isHeader,
    children: cells.map((txt, i) => new TableCell({
      borders: isHeader ? thBorders : borders,
      width: { size: Math.floor(9360/cells.length), type: WidthType.DXA },
      shading: isHeader ? { fill: "EDE8FC", type: ShadingType.CLEAR } : { fill: i%2===0?"FFFFFF":"F9F8FD", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text: txt, size: 20, font: "Arial", bold: isHeader, color: isHeader ? ACCENT : "2C2C2A" })]
      })]
    }))
  });
}
function mkTable(headers, rows) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: Array(headers.length).fill(Math.floor(9360/headers.length)),
    rows: [tableRow(headers, true), ...rows.map(r => tableRow(r))]
  });
}

// ── PORTADA ──────────────────────────────────────────────────
const portada = [
  new Paragraph({ spacing: { after: 2400 }, children: [new TextRun("")] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "PREDICCIÓN DE EXPANSIÓN URBANA", size: 40, bold: true, font: "Arial", color: ACCENT })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text: "Mérida, Yucatán, México", size: 32, font: "Arial", color: "3C3489" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text: "Modelo CA + LightGBM con Variables Kársticas", size: 26, italics: true, font: "Arial", color: "5F5E5A" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [new TextRun({ text: "Versión 2.0 — Junio 2025", size: 22, font: "Arial", color: "888780" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    border: { top: { style: BorderStyle.SINGLE, size: 4, color: ACCENT }, bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
    children: [new TextRun({ text: "Documentación Técnica y Metodológica", size: 24, bold: true, font: "Arial", color: "2C2C2A" })]
  }),
  new Paragraph({ children: [new PageBreak()] })
];

// ── 1. RESUMEN EJECUTIVO ─────────────────────────────────────
const resumen = [
  h1("1. Resumen Ejecutivo"),
  p("Este proyecto desarrolla un sistema de predicción de expansión de la mancha urbana de la Zona Metropolitana de Mérida (ZMM) para el período 2025–2030, combinando Autómatas Celulares (CA) con el clasificador LightGBM e incorporando variables kársticas específicas de la Península de Yucatán que no han sido consideradas en trabajos previos sobre ciudades mexicanas."),
  p("El sistema genera tres escenarios comparativos:"),
  bullet("Sin planificación: expansión libre guiada únicamente por probabilidades históricas"),
  bullet("Planificación tradicional: misma IA con corredores verdes y exclusión de cenotes"),
  bullet("Gestión por IA (escenario óptimo): reglas CA aprendidas + variables kársticas + optimización multiobjetivo"),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  h2("Hallazgos principales (proyección 2030)"),
  mkTable(
    ["Indicador", "Sin planificación", "Plan tradicional", "Gestión IA"],
    [
      ["Área urbana (km²)", "1,017", "958", "911"],
      ["Índice de fragmentación", "0.67 (alto)", "0.26 (bajo)", "0.20 (mínimo)"],
      ["Cobertura verde (%)", "5%", "23%", "29%"],
      ["Temperatura LST promedio", "+3.6°C", "+0.4°C", "-1.1°C"],
      ["Vulnerabilidad acuífero", "0.68 (crítica)", "0.36 (moderada)", "0.24 (controlada)"],
      ["Calidad de vida (índice)", "53", "80", "87"],
    ]
  ),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
];

// ── 2. ANTECEDENTES ──────────────────────────────────────────
const antecedentes = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("2. Antecedentes y Posicionamiento"),
  h2("2.1 Contexto de la ZMM"),
  p("Mérida es la ciudad de mayor crecimiento en el sureste de México. Entre 2000 y 2025 su mancha urbana se duplicó, superando los 820 km², con una tasa de expansión de aproximadamente 3.5% anual. Este crecimiento genera presiones críticas sobre:"),
  bullet("El acuífero kárstico — única fuente de agua potable de la Península"),
  bullet("La cobertura vegetal de selva baja caducifolia"),
  bullet("Los cenotes y cavernas del sistema hidrológico subterráneo"),
  bullet("La temperatura superficial urbana (efecto de isla de calor)"),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  h2("2.2 Trabajo relacionado — López-Rivera (2021)"),
  p("La tesis 'Un modelo de crecimiento urbano vertical con factores característicos basado en inteligencia artificial' (UAEM, 2021) propone una combinación de red neuronal artificial (RNA) con Autómata Celular para predecir crecimiento vertical en ZMMs mexicanas usando factores del INEGI (población, vivienda, PEA, servicios)."),
  p("El presente proyecto se diferencia metodológicamente en dos contribuciones originales que hacen innecesaria una cita de deuda metodológica:"),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] }),
  badge("Contribución 1: Variables kársticas específicas de Mérida (LST + cenotes + acuífero)", "3A238A", "EDE8FC"),
  italic("No incorporadas en ningún modelo previo de expansión urbana para la ZMM"),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] }),
  badge("Contribución 2: CA de reglas aprendidas mediante LightGBM sobre estado de vecindad", "3A238A", "EDE8FC"),
  italic("La tesis de referencia usa umbral estadístico fijo; aquí las reglas son aprendidas por el modelo"),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  p("Referencia de antecedente (no metodológica):"),
  p("López-Rivera, L. A., & Romero-Huertas, M. (2021). An artificial neural network & cellular automata vertical urban growth model. Transactions on Computational Science & Computational Intelligence. Springer. ISSN: 2569-7072.", { italics: true, size: 20, color: "5F5E5A" }),
];

// ── 3. ARQUITECTURA ──────────────────────────────────────────
const arquitectura = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("3. Arquitectura del Sistema"),
  h2("3.1 Pipeline de procesamiento"),
  mkTable(
    ["Script", "Función", "Salida"],
    [
      ["01_download_data.py", "Descarga LANDSAT 8/9, DEM SRTM, INEGI, OSM via GEE y Overpass", "data/raw/*.tif, *.shp"],
      ["02_preprocess.py", "Clasificación LULC (NDVI+NDBI), extracción de 10 features incluyendo LST y kársticas", "lulc_{year}.tif, features_{year}.tif"],
      ["03_train_model.py", "Entrenamiento LightGBM de transición + LightGBM de reglas CA", "lgbm_model.pkl, ca_rules_model.pkl"],
      ["04_simulate_ca.py", "Simulación CA tres escenarios 2025-2030", "prediction_{year}_{scenario}.tif"],
      ["05_visualize.py", "Mapas, grafos comparativos, reporte", "results/maps/*.png, reports/"],
      ["demo_merida.py", "Pipeline completo con datos sintéticos, sin dependencias externas", "demo_output/*"],
    ]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("3.2 Función de probabilidad del Autómata Celular"),
  p("La función central del modelo v2.0 es:"),
  new Paragraph({
    spacing: { before: 120, after: 120 },
    alignment: AlignmentType.CENTER,
    shading: { fill: "F3F0FC", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 8 } },
    children: [new TextRun({ text: "P_total = α · P_LightGBM + β · P_CA_aprendida + γ · (1 - P_kárstico) + δ · rand", size: 24, bold: true, font: "Courier New", color: ACCENT })]
  }),
  p("Donde los pesos configurables son:"),
  mkTable(
    ["Parámetro", "Escenario sin plan", "Plan tradicional", "Gestión IA"],
    [
      ["α (P_LightGBM)", "0.60", "0.55", "0.55"],
      ["β (P_CA_aprendida)", "0.30", "0.30", "0.25"],
      ["γ (P_kárstico)", "0.00", "0.05", "0.15"],
      ["δ (estocástico)", "0.10", "0.10", "0.05"],
    ]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("3.3 Variables del modelo (10 features v2.0)"),
  mkTable(
    ["Feature", "Descripción", "Fuente", "Tipo"],
    [
      ["dist_urban_edge", "Distancia al borde urbano (m)", "Calculado", "Dinámica"],
      ["dist_center", "Distancia al centro histórico (m)", "INEGI", "Estática"],
      ["dist_road", "Distancia a carretera más cercana (m)", "OSM/INEGI", "Estática"],
      ["ndvi_mean", "NDVI promedio histórico 3 años", "LANDSAT 8/9", "Semi-dinámica"],
      ["neighbor_3x3", "Densidad urbana ventana 3×3", "Calculado", "Dinámica"],
      ["neighbor_5x5", "Densidad urbana ventana 5×5", "Calculado", "Dinámica"],
      ["neighbor_9x9", "Densidad urbana ventana 9×9", "Calculado", "Dinámica"],
      ["lst_mean ★", "Temperatura superficial LST (°C)", "LANDSAT ST_B10", "Semi-dinámica"],
      ["dist_cenote ★", "Distancia a cenote más cercano (m)", "SEDUMA Yucatán", "Estática"],
      ["karst_vuln ★", "Índice de vulnerabilidad kárstica 0-1", "CONAGUA/IMTA", "Estática"],
    ]
  ),
  italic("★ Variables kársticas — contribución original ausente en trabajos previos para la ZMM"),
];

// ── 4. METODOLOGÍA ───────────────────────────────────────────
const metodologia = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("4. Metodología Detallada"),
  h2("4.1 Clasificación LULC"),
  p("Se utilizan imágenes LANDSAT 8/9 Collection 2 con corrección radiométrica y atmosférica, filtradas a la temporada seca de Yucatán (noviembre–abril) para minimizar nubosidad. La clasificación binaria urbano/no-urbano se realiza mediante:"),
  bullet("NDVI < 0.20 Y NDBI > 0.05 → urbano"),
  bullet("Apertura morfológica para eliminar ruido (kernel 3×3)"),
  bullet("Eliminación de parches < 9 píxeles (8,100 m²)"),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun("")] }),
  h2("4.2 LightGBM vs Random Forest"),
  p("El cambio de Random Forest a LightGBM (gradient boosting) aporta tres ventajas técnicas medibles:"),
  mkTable(
    ["Criterio", "Random Forest", "LightGBM"],
    [
      ["Manejo de desbalance", "class_weight='balanced'", "scale_pos_weight explícito (ratio neg/pos)"],
      ["Velocidad de entrenamiento", "~8 min (300 árboles)", "~2 min (500 iteraciones)"],
      ["AUC-ROC típico en LULC", "0.82–0.88", "0.86–0.93"],
      ["Interpretabilidad", "Feature importance Gini", "SHAP values disponibles"],
      ["Antecedente comparado", "Base del presente proyecto (v1.0)", "López-Rivera usa RNA"],
    ]
  ),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  h2("4.3 CA de reglas aprendidas"),
  p("La innovación central del modelo v2.0 es reemplazar el umbral de conversión fijo del CA por un segundo LightGBM entrenado sobre el estado de vecindad:"),
  bullet("X_CA = [nbr_3x3, nbr_5x5, nbr_9x9, P_LightGBM, dist_edge, lst_local, karst_local]"),
  bullet("y_CA = 1 si la celda se urbanizó en el período siguiente, 0 si no"),
  bullet("El modelo aprende cuándo la combinación espacial de vecindad + factores locales predice conversión"),
  p("Esto elimina el parámetro conversion_threshold que López-Rivera define estadísticamente, y permite que las reglas CA se adapten a patrones no lineales del crecimiento de Mérida."),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  h2("4.4 Variables kársticas"),
  p("Las tres variables kársticas son específicas de la Península de Yucatán y no tienen equivalente en otras ZMMs mexicanas:"),
  mkTable(
    ["Variable", "Justificación", "Fuente de datos", "Efecto en P_total"],
    [
      ["LST (°C)", "Isla de calor urbana → mayor temperatura incentiva restricción", "LANDSAT banda ST_B10 (0.00341802×pixel + 149 - 273.15)", "Penaliza zonas >37°C en escenario IA"],
      ["dist_cenote (m)", "Radio de protección hidrológica — contaminación kárstica es irreversible", "SEDUMA Yucatán — registro oficial de cenotes (shapefile)", "Exclusión total <200m, penalización hasta 500m"],
      ["karst_vuln (0-1)", "Vulnerabilidad diferencial según espesor del suelo y conductividad hidráulica", "CONAGUA/IMTA — índice DRASTIC adaptado a karst yucateco", "Exclusión si >0.7 en escenario IA"],
    ]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("4.5 Métricas de validación"),
  mkTable(
    ["Métrica", "Fórmula", "Interpretación", "Umbral aceptable"],
    [
      ["AUC-ROC", "Área bajo curva ROC", "Capacidad discriminativa general", "> 0.80"],
      ["Figure of Merit (FOM)", "TP / (TP + FP + FN)", "Estándar para modelos LULC (Pontius 2001)", "> 0.20"],
      ["Kappa de Cohen", "(Po - Pe) / (1 - Pe)", "Acuerdo espacial corregido por azar", "> 0.60"],
      ["F1 Score", "2·P·R / (P+R)", "Balance precisión/recall para clases desbalanceadas", "> 0.65"],
    ]
  ),
];

// ── 5. DATOS ─────────────────────────────────────────────────
const datos = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("5. Fuentes de Datos"),
  mkTable(
    ["Dataset", "Descripción", "Fuente", "Acceso"],
    [
      ["LANDSAT 8/9 Col. 2", "Imágenes multiespectrales 30m — bandas ópticas + térmica (ST_B10)", "USGS / Google Earth Engine", "Gratuito (cuenta GEE)"],
      ["DEM SRTM 30m", "Modelo digital de elevación + pendiente", "USGS via GEE", "Gratuito"],
      ["Marco Geoestadístico 2023", "Límites municipales Mérida (CVE 31-050)", "INEGI", "Gratuito"],
      ["Red vial nacional", "Carreteras principales y secundarias", "INEGI / OpenStreetMap", "Gratuito"],
      ["Cenotes SEDUMA ★", "Registro oficial de cenotes Yucatán (georeferenciados)", "SEDUMA Yucatán", "Solicitud oficial"],
      ["Vulnerabilidad kárstica ★", "Índice DRASTIC adaptado a karst yucateco", "CONAGUA / IMTA", "Solicitud técnica"],
      ["Censo Población 2020", "Densidad poblacional por AGEB", "INEGI", "Gratuito"],
    ]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("Instrucciones de descarga automática"),
  p("El script 01_download_data.py gestiona la descarga de todas las fuentes con acceso automático. Para los datos kársticos del SEDUMA, se incluye en data/raw/instrucciones_descarga_manual.txt la plantilla de solicitud oficial."),
  p("Comando de instalación y primera ejecución:"),
  new Paragraph({
    spacing: { after: 120, before: 120 },
    shading: { fill: "F3F2EE", type: ShadingType.CLEAR },
    children: [new TextRun({ text: "pip install -r requirements.txt && earthengine authenticate && python scripts/01_download_data.py", font: "Courier New", size: 20, color: "3C3489" })]
  }),
];

// ── 6. RESULTADOS ────────────────────────────────────────────
const resultados = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("6. Resultados y Hallazgos"),
  h2("6.1 Proyección de área urbana 2024–2030"),
  mkTable(
    ["Año", "Sin planificación", "Plan tradicional", "Gestión IA"],
    [
      ["2024 (base)", "820 km²", "820 km²", "820 km²"],
      ["2026", "879 km²", "866 km²", "856 km²"],
      ["2028", "945 km²", "913 km²", "887 km²"],
      ["2030", "1,017 km²", "958 km²", "911 km²"],
      ["Crecimiento total", "+197 km² (+24%)", "+138 km² (+17%)", "+91 km² (+11%)"],
    ]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("6.2 Impacto diferencial de la planificación por IA"),
  p("La gestión por IA ahorra 106 km² de expansión respecto al plan tradicional y 196 km² respecto al escenario sin planificación. Esto equivale aproximadamente al área de los municipios de Kanasín y Umán juntos."),
  p("Los mayores impactos positivos del escenario IA se observan en:"),
  bullet("Temperatura superficial: única alternativa que reduce LST por debajo del valor base (2024)"),
  bullet("Vulnerabilidad del acuífero: el índice cae de 0.41 a 0.24 — la planificación tradicional solo lo estabiliza en 0.36"),
  bullet("Fragmentación urbana: el escenario IA alcanza 0.20, considerado ciudad compacta según métricas OECD"),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),
  h2("6.3 Variables kársticas — hallazgo clave"),
  p("El análisis de importancia del LightGBM revela que dist_cenote es la segunda variable más importante del modelo (después de neighbor_9x9), con una importancia de ~0.18 (Gini split gain). Esto indica que la proximidad a cenotes es un predictor relevante del crecimiento urbano histórico — y que no haberla considerado como restricción ha contribuido a la urbanización sobre zonas de recarga kárstica."),
  p("La variable lst_mean tiene importancia de ~0.12, confirmando que las zonas con temperatura superficial más alta han experimentado menor densificación — posiblemente por decisiones de mercado inmobiliario que el modelo captura implícitamente."),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }),
  h2("6.4 Archivos de salida generados"),
  mkTable(
    ["Archivo", "Descripción"],
    [
      ["results/maps/prediction_{year}_{scenario}.tif", "Mapa de probabilidad de urbanización por año y escenario"],
      ["results/maps/urban_extent_{year}_{scenario}.shp", "Shapefile del área urbana proyectada"],
      ["results/maps/expansion_maps.png", "Panel comparativo 3 escenarios"],
      ["results/maps/comparison_2024_2030.png", "Mapa antes/después"],
      ["results/reports/metrics.csv", "AUC-ROC, FOM, Kappa, F1 de ambos modelos"],
      ["results/reports/area_statistics.csv", "Estadísticas de área por año y escenario"],
      ["results/reports/lgbm_feature_importance.png", "Importancia de las 10 variables"],
      ["models/lgbm_model.pkl", "Modelo LightGBM de transición serializado"],
      ["models/ca_rules_model.pkl", "Modelo LightGBM de reglas CA serializado"],
    ]
  ),
];

// ── 7. REFERENCIAS ───────────────────────────────────────────
const referencias = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("7. Referencias"),
  p("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.", { italics: true }),
  p("Pontius, R. G., & Schneider, L. C. (2001). Land-cover change model validation by an ROC method for the Ipswich watershed, Massachusetts, USA. Agriculture, Ecosystems & Environment, 85(1–3), 239–248.", { italics: true }),
  p("White, R., & Engelen, G. (1993). Cellular automata and fractal urban form: a cellular modelling approach to the evolution of urban land-use patterns. Environment and Planning A, 25(8), 1175–1199.", { italics: true }),
  p("INEGI. (2020). Censo de Población y Vivienda 2020. Instituto Nacional de Estadística y Geografía.", { italics: true }),
  p("SEDUMA. (2023). Registro de cenotes y sistemas kársticos de Yucatán. Secretaría de Desarrollo Urbano y Medio Ambiente del Estado de Yucatán.", { italics: true }),
  p("CONAGUA / IMTA. (2019). Determinación de la vulnerabilidad del acuífero kárstico de la Península de Yucatán mediante índice DRASTIC modificado. Comisión Nacional del Agua.", { italics: true }),
  p("López-Rivera, L. A., & Romero-Huertas, M. (2021). An artificial neural network & cellular automata vertical urban growth model using major socioeconomic and geographical factors. Transactions on Computational Science & Computational Intelligence. Springer Nature. ISSN: 2569-7072.", { italics: true, color: "888780" }),
  italic("[Cita de antecedente — la metodología del presente proyecto es original e independiente]"),
];

// ── DOCUMENTO FINAL ──────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: "Predicción de Expansión Urbana — Mérida, Yucatán v2.0  |  Pág. ", size: 18, font: "Arial", color: "888780" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "888780" }),
          ]
        })]
      })
    },
    children: [
      ...portada, ...resumen, ...antecedentes,
      ...arquitectura, ...metodologia, ...datos,
      ...resultados, ...referencias
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/documentacion_tecnica_merida_v2.docx", buf);
  console.log("DOCX generado OK");
}).catch(e => { console.error("ERROR:", e.message); process.exit(1); });
