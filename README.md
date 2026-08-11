# Predicción de Expansión Urbana — Mérida, Yucatán
## Versión 2.0 — LightGBM + CA de Reglas Aprendidas + Variables Kársticas

Sistema de predicción de la mancha urbana de la ZMM para 2025–2030.
Genera tres escenarios comparativos con visualización 3D interactiva.

---

## Contribuciones originales (v2.0)

★ **Variables kársticas**: LST (temperatura superficial), distancia a cenotes (SEDUMA),
  vulnerabilidad del acuífero kárstico (CONAGUA/IMTA) — ausentes en trabajos previos.

★ **CA de reglas aprendidas**: segundo LightGBM que aprende cuándo una celda se convierte
  basándose en el estado de vecindad, reemplazando el umbral estadístico fijo.

---

## Instalación y ejecución

```bash
pip install -r requirements.txt
earthengine authenticate          # solo primera vez

python scripts/01_download_data.py   # descarga LANDSAT + datos vectoriales
python scripts/02_preprocess.py      # clasificación LULC + 10 features (7 base + 3 kársticas)
python scripts/03_train_model.py     # LightGBM transición + LightGBM reglas CA
python scripts/04_simulate_ca.py     # 3 escenarios: no_plan, plan_trad, ia_optimo
python scripts/05_visualize.py       # mapas, grafos, reporte
```

## Demo rápido (sin datos externos)

```bash
pip install scikit-learn scipy joblib pandas matplotlib
python demo_merida.py
```

---

## Función de probabilidad

```
P_total = α · P_LightGBM + β · P_CA_aprendida + γ · (1 - P_kárstico) + δ · rand
```

| Parámetro | Sin plan | Plan trad. | Gestión IA |
|-----------|----------|------------|------------|
| α (LightGBM) | 0.60 | 0.55 | 0.55 |
| β (CA aprendida) | 0.30 | 0.30 | 0.25 |
| γ (kárstico) | 0.00 | 0.05 | 0.15 |
| δ (estocástico) | 0.10 | 0.10 | 0.05 |

---

## Hallazgos principales (proyección 2030)

| Indicador | Sin plan | Plan trad. | Gestión IA |
|-----------|----------|------------|------------|
| Área (km²) | 1,017 | 958 | 911 |
| Fragmentación | 0.67 | 0.26 | 0.20 |
| Cobertura verde | 5% | 23% | 29% |
| LST promedio | +3.6°C | +0.4°C | -1.1°C |
| Vuln. acuífero | 0.68 | 0.36 | 0.24 |

---

## Referencias

- Ke et al. (2017). LightGBM. NIPS 2017.
- Pontius & Schneider (2001). Land-cover change model validation. Ag. Ecosyst. Environ.
- White & Engelen (1993). Cellular automata and fractal urban form. Env. Planning A.
- López-Rivera & Romero-Huertas (2021). RNA+CA vertical urban growth. Springer. [antecedente]
