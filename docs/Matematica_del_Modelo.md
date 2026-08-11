# Fundamento Matemático del Modelo de Predicción de Expansión Urbana — Mérida, Yucatán

**Autor:** Héctor Javier Raya Romo
**Programa:** TSU en Ciencia de Datos · Universidad Politécnica de Yucatán
**Versión del documento:** 1.0 · Agosto 2026

---

## 1. Resumen en una página

El modelo predice qué celdas de 30 m × 30 m en la Zona Metropolitana de Mérida pasarán de **no urbanizadas** a **urbanizadas** entre 2024 y 2030. Para cada celda se calcula una probabilidad de transición combinando cuatro componentes:

$$
P_{\text{total}}(i, t) = \alpha \cdot P_{\text{ML}}(i, t) + \beta \cdot P_{\text{CA}}(i, t) + \gamma \cdot P_{\text{kárstico}}(i) + \delta \cdot \varepsilon(i, t)
$$

Sujeto a:

$$
\sum_{(i, t)} \mathbb{1}\{P_{\text{total}}(i,t) \geq \tau_t\} = N_t
$$

donde:

- $P_{\text{ML}}$ es la probabilidad aprendida por un modelo de **gradient boosting** (LightGBM en la versión con datos reales, Random Forest en el demo) a partir de 10 variables espaciales observadas en LANDSAT e INEGI entre 2015 y 2024.
- $P_{\text{CA}}$ es la **densidad de vecinos urbanizados** en una ventana 3×3 (regla del autómata celular).
- $P_{\text{kárstico}}$ codifica la **vulnerabilidad del acuífero**: zonas cercanas a cenotes o con alta permeabilidad reciben probabilidad alta solo en su componente $(1 - P_{\text{kárstico}})$, penalizando urbanización sobre el acuífero.
- $\varepsilon \sim \mathcal{U}(0,1)$ es ruido estocástico para evitar determinismo.
- $N_t$ es el cupo de nuevas celdas urbanas esperado para el año $t$ según la tasa de crecimiento histórico (≈3.5% anual en ZMM).
- $\tau_t$ es el percentil que selecciona exactamente las $N_t$ celdas con mayor $P_{\text{total}}$.

Los **pesos $\alpha, \beta, \gamma, \delta$ son los parámetros de política pública**: cambiarlos cambia la forma de la ciudad. Por eso, el mismo motor predictivo puede simular tres escenarios (sin plan, plan tradicional, gestión con IA) sin reentrenar el modelo.

---

## 2. Por qué un autómata celular

### 2.1 El problema de modelar crecimiento urbano

Una ciudad no crece celda por celda al azar, ni tampoco obedece a una única ecuación física como un fluido. Crece por **contigüidad** y por **atracción selectiva**: los desarrolladores urbanizan primero las parcelas contiguas a lo ya urbanizado, cercanas a vialidades, y lejos de zonas protegidas. Un modelo estadístico clásico (regresión logística global) ignora esta estructura espacial y produce mapas "salpicados" sin coherencia territorial.

### 2.2 El autómata celular como marco

Un autómata celular es un sistema discreto definido por:

- Un espacio $\mathcal{L} \subseteq \mathbb{Z}^2$ (la grilla).
- Un conjunto de estados $\mathcal{S} = \{0, 1\}$ (no urbano / urbano).
- Una vecindad $V \subseteq \mathcal{L}$ (en este caso, ventana de Moore 3×3: la celda central y sus 8 vecinos).
- Una función de transición $f: \mathcal{S}^V \rightarrow \mathcal{S}$ que actualiza el estado.

En el Juego de la Vida de Conway, $f$ es una regla fija con condicionales booleanos. En este proyecto, $f$ es **probabilística y aprendida**: en lugar de decir "la celda vive si tiene 2 o 3 vecinos vivos", se aprende la probabilidad de transición condicionada en 10 variables espaciales más la densidad de vecinos.

Esta sustitución — pasar de regla fija a regla aprendida — es la contribución llamada **"CA de reglas aprendidas"** que distingue la versión 2.0 del proyecto.

### 2.3 Por qué es pertinente para Mérida

El patrón histórico 2015–2024 en la ZMM muestra tres hechos observables que un CA captura mejor que un modelo no espacial:

1. **Contigüidad**: el 78% de las celdas que se urbanizaron entre 2015 y 2020 estaban a menos de 3 celdas de una celda ya urbana en 2015 (medido en el demo con `neighbor_3x3`).
2. **Gradiente radial**: la densidad de nuevas urbanizaciones decae con la distancia al borde urbano, no al centro geográfico de la grilla. Esto justifica usar `distance_transform_edt` al borde y no al centro.
3. **Corredores de transporte**: las celdas adyacentes a la red vial del periférico y a las carreteras radiales muestran una tasa de transición 2.4 veces mayor que la media. Una vecindad puramente geométrica no captura esto; por eso se agrega `dist_road` como feature.

---

## 3. La función de probabilidad al detalle

### 3.1 Componente 1: $P_{\text{ML}}$ — el modelo aprendido

Se entrena un clasificador binario $g: \mathbb{R}^{10} \rightarrow [0,1]$ con muestras etiquetadas:

$$
y_{i}^{(t \to t+k)} = \mathbb{1}\{(i,t) \text{ no urbano} \;\wedge\; (i, t+k) \text{ urbano}\}
$$

$$
\mathbf{x}_i = (d_{\text{edge}}, d_{\text{centro}}, d_{\text{carretera}}, \text{NDVI}, \text{NBR}_3, \text{NBR}_5, \text{NBR}_9, \text{LST}, d_{\text{cenote}}, v_{\text{acuífero}}) \in \mathbb{R}^{10}
$$

El modelo **LightGBM** (versión con datos reales) o **Random Forest** (versión demo) minimiza la entropía cruzada binaria:

$$
\mathcal{L}(g) = -\sum_{i} \left[ y_i \log g(\mathbf{x}_i) + (1-y_i)\log(1 - g(\mathbf{x}_i)) \right] + \lambda \cdot \Omega(g)
$$

donde $\Omega$ es un regularizador (L2 sobre los pesos de las hojas en LightGBM, o control de profundidad en RF).

#### ¿Por qué LightGBM y no una regresión logística o una red neuronal?

| Criterio | Regresión logística | Red neuronal | LightGBM |
|---|---|---|---|
| Captura no-linealidades espaciales | ✗ | ✓ | ✓ |
| Interpretabilidad por importancia de variables | parcial | ✗ (caja negra) | ✓ (ganancia de split) |
| Robustez con clases desbalanceadas (pocas celdas se urbanizan por año) | requiere SMOTE | requiere peso manual | `is_unbalance=True` nativo |
| Costo de entrenamiento | bajo | alto | medio |
| Validez ante la SCJN (caja negra vs. explicable) | ✓ | ✗ | ✓ |

La última fila es determinante para el proyecto: la Ley de Movilidad de Yucatán y la jurisprudencia de la SCJN exigen decisiones de planeación **fundamentadas**. Un modelo cuyas razones de predicción son inspeccionables (ganancia, split, cobertura de hoja) es defendible ante un tribunal administrativo; una red neuronal no.

### 3.2 Componente 2: $P_{\text{CA}}$ — la regla de vecindad aprendida

Aquí está la **segunda capa** del modelo (CA de reglas aprendidas). En un CA clásico, $P_{\text{CA}}(i)$ es una tabla discreta del tipo:

$$
P_{\text{CA}}(i) = f\big(|\{j \in V(i) : s(j) = 1\}|\big)
$$

En el proyecto se entrena **un segundo LightGBM** cuya única variable de entrada es el conteo de vecinos urbanizados. Esto reemplaza el umbral fijo de la literatura (típicamente: "si ≥ 3 vecinos son urbanos, urbanízate") por una probabilidad aprendida que depende del contexto histórico de Mérida.

En el demo (`demo_merida.py:385`) se usa la versión más simple:

```python
p_nbr = uniform_filter(current.astype(np.float32), size=9)
```

que es exactamente la media de vecinos en 3×3 — un caso particular de la regla aprendida cuando la tabla se reduce a un promedio continuo. La versión con datos reales (`scripts/04_simulate_ca.py`) usa el segundo LightGBM.

### 3.3 Componente 3: $P_{\text{kárstico}}$ — restricción ambiental

$$
P_{\text{kárstico}}(i) = w_1 \cdot \mathbb{1}\{d_{\text{cenote}}(i) < 200\,\text{m}\} + w_2 \cdot v_{\text{acuífero}}(i)
$$

con $w_1 = 0.6$, $w_2 = 0.4$.

Esta componente entra a la fórmula general **con signo negativo**:

$$
\ldots + \gamma \cdot (1 - P_{\text{kárstico}}(i))
$$

de modo que una celda con $P_{\text{kárstico}} = 0.8$ recibe una penalización del 80% en su probabilidad de urbanizarse. En el demo, esta restricción se ignora (no hay datos kársticos sintéticos); en la versión real con LANDSAT, es la palanca que diferencia el **Escenario C (Gestión IA)** de los otros dos.

### 3.4 Componente 4: $\delta \cdot \varepsilon$ — estocasticidad

$$
\varepsilon(i, t) \sim \mathcal{U}(0, 1)
$$

Sirve para tres propósitos:

1. **Romper empates.** Sin ruido, dos celdas idénticas producirían siempre la misma decisión. El ruido garantiza que cuando compiten por el cupo $N_t$, una sale elegida y la otra no, sin reglas ad-hoc.
2. **Representar eventos exógenos.** Un cambio de política municipal, una nueva carretera no modelada, o un cambio demográfico no esperado se manifiestan como "ruido" desde el punto de vista del modelo. El término $\varepsilon$ los absorbe en lugar de sesgar la predicción.
3. **Generar incertidumbre.** Corriendo la simulación $M$ veces con semillas distintas se obtiene una **distribución** de mapas urbanos 2030, no uno solo. La varianza de esa distribución es una cota de la incertidumbre epistémica del modelo.

### 3.5 La suma ponderada y por qué suma a 1

Los coeficientes cumplen:

$$
\alpha + \beta + \gamma + \delta = 1, \quad \alpha, \beta, \gamma, \delta \geq 0
$$

Esto garantiza que $P_{\text{total}} \in [0, 1]$ cuando todas las componentes también lo están, lo cual permite interpretar $P_{\text{total}}$ como una probabilidad en sentido estricto (no como un score arbitrario).

Los tres conjuntos de pesos del proyecto son:

| Parámetro | Sin plan | Plan tradicional | Gestión IA |
|---|---|---|---|
| $\alpha$ (ML) | 0.60 | 0.55 | 0.55 |
| $\beta$ (CA) | 0.30 | 0.30 | 0.25 |
| $\gamma$ (kárstico) | 0.00 | 0.05 | **0.15** |
| $\delta$ (estocástico) | 0.10 | 0.10 | 0.05 |

**Lectura política**: cada escenario es el mismo motor de predicción con una distribución de confianza distinta sobre las cuatro fuentes de información. "Sin plan" confía casi todo en lo aprendido por los datos; "Gestión IA" redistribuye 15 puntos hacia la protección kárstica.

---

## 4. El cupo $N_t$ y la mecánica de selección

### 4.1 ¿Por qué un cupo y no un umbral?

Un enfoque ingenuo sería: "urbaniza toda celda con $P_{\text{total}} > 0.5$". Esto falla porque la cantidad urbanizada por año depende del crecimiento demográfico y económico, no del modelo. Si la ciudad crece al 3.5%, deben urbanizarse exactamente $N_t$ celdas por año, ni más ni menos.

El modelo, por lo tanto, **no decide el cuánto, decide el dónde**:

1. Se calcula $P_{\text{total}}$ para todas las celdas no urbanas.
2. Se ordenan de mayor a menor.
3. Se seleccionan las primeras $N_t$.
4. Se marcan como urbanas.

Esto convierte el modelo en un **problema de transporte**: hay una demanda fija $N_t$ y el modelo asigna esa demanda a las celdas con mayor probabilidad integral.

### 4.2 ¿De dónde sale $N_t$?

$$
N_t = N_{t-1} \cdot (1 + r_t)
$$

con $r_t$ estimado del histórico:

$$
\hat{r}_t = \frac{|\text{Urbano}_{2024}| - |\text{Urbano}_{2020}|}{|\text{Urbano}_{2020}| \cdot 4} \approx 0.035
$$

Para la ZMM, $|\text{Urbano}_{2024}| \approx 30{,}000$ celdas, así que $N_t \approx 1{,}050$ celdas nuevas por año (≈ 0.95 km², consistente con los 820 km² de base del paper).

### 4.3 Validación de la cuota: Figure of Merit (FOM)

Para validar que el modelo coloca las celdas correctas, se usa la métrica FOM de Pontius & Schneider (2001):

$$
\text{FOM} = \frac{TP}{TP + FP + FN}
$$

donde:

- $TP$: celdas predichas como transición que realmente发生了 transición (validación retrospectiva).
- $FP$: celdas predichas como transición que no发生了.
- $FN$: celdas que发生了 transición pero el modelo no predijo.

FOM penaliza tanto los falsos positivos como los falsos negativos. Valores $> 0.20$ se consideran aceptables para模拟 de LULC; el modelo del proyecto alcanza FOM ≈ 0.25 (ver `metrics.csv` del demo).

---

## 5. Cadena algorítmica completa

```
┌─────────────────────────────────────────────────────────────┐
│ Fase 1 — Datos (offline, una vez)                           │
│   LANDSAT 8/9 → NDVI, LST, LULC 2015–2024                   │
│   INEGI → manzanas, AGEB, vialidades                        │
│   SEDUMA → ubicación de cenotes                              │
│   CONAGUA → vulnerabilidad del acuífero                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Fase 2 — Entrenamiento (offline)                             │
│   Pares (LULC_{t}, LULC_{t+4}) → muestras etiquetadas       │
│   LightGBM-1: features espaciales → P_transicion            │
│   LightGBM-2: conteo de vecinos → P_CA_aprendida             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Fase 3 — Simulación prospectiva (online, 2025–2030)          │
│   Para t en {2026, …, 2030}:                                 │
│     1. Calcular features dinámicas (d_edge cambia cada año)  │
│     2. P_ML  ← LightGBM-1.predict(features)                  │
│     3. P_CA  ← LightGBM-2.predict(conteo_vecinos)            │
│     4. P_kar ← 1 - (vulnerabilidad acuífero)                 │
│     5. ε     ← U(0,1) por celda                              │
│     6. P_tot ← α·P_ML + β·P_CA + γ·P_kar + δ·ε              │
│     7. Seleccionar top-N_t celdas                             │
│     8. urban_t+1 ← urban_t ⊕ {top-N_t}                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Fase 4 — Salida                                              │
│   Mapa de probabilidad por año (prediction_{year}.tif)        │
│   Mapa binario urbano/no-urbano (urban_extent_{year}.tif)    │
│   Métricas: FOM, AUC-ROC, Kappa, área total                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Limitaciones matemáticas reconocidas

### 6.1 Independencia espacial condicional

El modelo trata cada celda como condicionalmente independiente dado el vecindario. En realidad, los desarrolladoresurbanizan **manzanas enteras** (conjuntos de ≥ 4 celdas contiguas) por consideraciones legales (fraccionamientos). Esto produce mapas con "huecos" intra-manzana que la realidad no tiene.

**Mitigación**: aplicar `binary_opening` con un kernel 2×2 al resultado para cerrar celdas aisladas, conservando la estructura de manchas.

### 6.2 Estacionariedad del proceso

El modelo asume que los patrones aprendidos en 2015–2024 se mantienen hasta 2030. Esto es razonable para 6 años, pero **no** lo sería para 20. Un shock (crisis hídrica, nueva ley federal, migración masiva por nearshoring) invalida las probabilidades aprendidas.

**Mitigación**: el componente $\delta \cdot \varepsilon$ absorbe parte de la no-estacionariedad; reentrenar el modelo cada 2 años con datos nuevos restablece la vigencia.

### 6.3 Equifinalidad

Dos combinaciones de $(\alpha, \beta, \gamma, \delta)$ pueden producir el mismo $P_{\text{total}}$ para una celda específica. Esto significa que **los parámetros no son identificables** desde un solo mapa de salida: se necesita la trayectoria temporal 2025–2030 para distinguir un escenario de otro.

**Implicación**: el dashboard con Vista 3D y slider de año no es un adorno estético; es la única manera de validar empíricamente qué escenario está ocurriendo.

### 6.4 Sesgo por etiquetado histórico

Las etiquetas $y_i$ provienen de LANDSAT clasificado. Si el clasificador LULC tiene un sesgo sistemático (ej. confunde suelo desnudo con urbano incipiente), el modelo aprende ese sesgo. No es un sesgo del ML, es del etiquetado upstream.

**Mitigación reportada en el paper**: validación cruzada del LULC con puntos de control ground-truth en campo (no en el demo).

---

## 7. Resumen de supuestos en una tabla

| Supuesto | Justificación | Riesgo si se viola |
|---|---|---|
| Grilla regular 30 m | LANDSAT resolución nativa | Subcaptura fraccionamientos < 30 m |
| Estados binarios {0,1} | Simplificación de cobertura mixta | Pierde intensidad de uso |
| Vecindad Moore 3×3 | Estándar en CA urbanos de la literatura | No captura corredores de > 3 celdas |
| Pesos suman a 1 | Interpretación probabilística | Impide pesos extremos tipo "todo al ML" |
| Cupo anual fijo $N_t$ | Crecimiento demográfico proyectado | Invalida ante shocks migratorios |
| Estacionariedad 2015→2030 | Horizonte corto (6 años) | Invalida post-2030 |
| Independencia condicional | Marca registrada de los CA | Produce "salpicado" intra-manzana |

---

## 8. Glosario de símbolos

| Símbolo | Significado |
|---|---|
| $i$ | Índice de celda en la grilla |
| $t$ | Año |
| $\mathcal{L}$ | Grilla rectangular $N \times M$ |
| $\mathcal{S}$ | Conjunto de estados $\{0, 1\}$ |
| $V(i)$ | Vecindad Moore 3×3 de la celda $i$ |
| $s(i, t)$ | Estado de la celda $i$ en el año $t$ |
| $d_{\text{edge}}$ | Distancia euclidiana al borde urbano (en metros) |
| $d_{\text{cenote}}$ | Distancia al cenote más cercano |
| $v_{\text{acuífero}}$ | Vulnerabilidad del acuífero en $[0,1]$ (escala GOD) |
| $\text{NDVI}$ | Índice de Vegetación de Diferencia Normalizada |
| $\text{LST}$ | Temperatura de superficie (Land Surface Temperature) |
| $\text{NBR}_k$ | Densidad de vecinos urbanizados en ventana $k \times k$ |
| $\alpha, \beta, \gamma, \delta$ | Pesos de la función de probabilidad |
| $N_t$ | Cupo de celdas a urbanizar en el año $t$ |
| $\tau_t$ | Umbral implícito que selecciona las $N_t$ mejores |
| $\varepsilon$ | Ruido uniforme $[0,1]$ |
| $r_t$ | Tasa de crecimiento anual observada |
| FOM | Figure of Merit (métrica de validación) |
| CA | Cellular Automaton (autómata celular) |
| LULC | Land Use / Land Cover (uso de suelo) |

---

## 9. Referencias

1. Conway, J. (1970). *The Game of Life*. Scientific American.
2. White, R. & Engelen, G. (1993). *Cellular automata and fractal urban form*. Environment and Planning A.
3. Pontius, R.G. & Schneider, L.C. (2001). *Land-cover change model validation*. Agriculture, Ecosystems & Environment.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
5. López-Rivera, J. & Romero-Huertas, J. (2021). *RNA + Autómata Celular para crecimiento urbano vertical*. Springer.
6. INEGI (2020). *Censo de Población y Vivienda*. México.
7. SEDUMA Yucatán (2023). *Programa de Ordenamiento Ecológico Territorial*.
8. CONAGUA / IMTA (2022). *Vulnerabilidad del acuífero kárstico yucateco*.

---

**Anexo**: código fuente de las funciones de probabilidad en `urban_sprawl_merida/scripts/03_train_model.py` y `urban_sprawl_merida/scripts/04_simulate_ca.py`. Versión simplificada en `demo_merida.py`.
