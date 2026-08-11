# Speech de Exposición — Mérida Urban Sprawl Predictor
**Duración estimada:** ~10 minutos
**Tono:** Formal académico, accesible a público no técnico
**Autor:** Héctor Javier Raya Romo · TSU Ciencia de Datos · UPY

---

## GUIÓN COMPLETO

### [0:00 – 0:45] · APERTURA (impacto)

> Buenos días / Buenas tardes.
>
> Antes de empezar, quiero que imaginen algo: la Península de Yucatán no tiene ríos. Ni uno solo. Toda el agua que consumen más de dos millones de personas sale del subsuelo, de un sistema de cenotes y cavernas kársticas que tardó millones de años en formarse.
>
> Y ahora imaginen que, encima de ese acuífero único, hay una ciudad que creció 60% en apenas 25 años. Pasó de 650 mil a más de un millón de habitantes.
>
> Esa ciudad es Mérida. Y este proyecto trata de responder una sola pregunta: **si el Gobierno de Yucatán invierte 1,500 millones de pesos en infraestructura, ¿cuál es la forma inteligente de gastarlos para que la ciudad crezca sin destruir el acuífero que la sostiene?**

---

### [0:45 – 2:00] · EL PROBLEMA (contexto)

> La Zona Metropolitana de Mérida enfrenta tres presiones simultáneas que no pueden seguirse tratando por separado.
>
> **Primero, el crecimiento demográfico.** El INEGI proyecta que la ciudad seguirá creciendo a tasas altas por migración interestatal y por turismo.
>
> **Segundo, el riesgo ambiental.** Yucatán tiene uno de los acuíferos kársticos más vulnerables del mundo. No hay filtro natural: lo que cae al suelo llega al agua subterránea en horas, no en años. Eso significa que cada kilómetro cuadrado que se urbaniza sin control es contaminación directa al agua que bebemos.
>
> **Tercero, el modelo tradicional de infraestructura.** Cuando una ciudad crece, la respuesta clásica es ampliar carriles, construir pasos a desnivel y retornos. Eso no resuelve la congestión, solo la pospone. La literatura de planeación urbana lo llama *demanda inducida*: más carriles, más coches, más carriles.
>
> El Gobierno de Yucatán lo sabe, y por eso está por aprobar un préstamo de 1,500 millones de pesos para infraestructura urbana 2025–2030. **La pregunta es cómo asignarlo.**

---

### [2:00 – 3:30] · LA PREGUNTA DE INVESTIGACIÓN

> Mi investigación parte de una hipótesis concreta:
>
> *"No todas las formas de invertir 1,500 MDP producen el mismo resultado. Un modelo predictivo puede mostrar —con evidencia cuantitativa— cuál estrategia compacta la ciudad, protege el acuífero y maximiza el retorno social de la inversión."*
>
> Para responderla construí un **modelo de simulación de crecimiento urbano 2024–2030** que integra dos técnicas:
>
> 1. **Autómatas Celulares** — reglas de vecindad que simulan cómo una celda urbanizada "contagia" a sus vecinas.
> 2. **LightGBM** — un modelo de Machine Learning que aprendió los patrones reales de expansión de Mérida entre 2015 y 2024 a partir de imágenes satelitales LANDSAT 8 y 9.
>
> La ecuación general del modelo es:
>
> **P_total = α·P_LightGBM + β·P_vecindad + γ·(1−P_kárstico) + ζ·P_movilidad + δ·aleatorio**
>
> Cada peso representa una política pública diferente. **α** es qué tanto confiamos en el modelo. **β** es el efecto de "contagio" urbano. **γ** es la restricción para no urbanizar sobre el acuífero. Y **ζ** es la atracción a corredores de transporte. **Cambiar estos parámetros es, literalmente, cambiar la política de inversión.**

---

### [3:30 – 5:30] · LOS 3 ESCENARIOS (hallazgos)

> Comparé tres escenarios en proyección al año 2030. Los datos que les voy a dar son salida directa del modelo, no estimaciones optimistas.
>
> **Escenario A — Infraestructura vehicular tradicional.** Es lo que se ha hecho siempre: ampliar el Periférico, construir pasos a desnivel. Los parámetros reflejan esa política: γ kárstico en cero, sin protección de cenotes.
>
> Resultado: la ciudad pasa de 820 a **1,017 km²**, la fragmentación urbana sube a **0.67** (crítica), la temperatura de superficie sube **+3.6°C**, y la vulnerabilidad del acuífero llega a **0.68** —casi en rojo. El retorno de la inversión es **negativo**. Estamos gastando 1,500 MDP para destruir el activo que sostiene a la ciudad.
>
> *(Pausa breve para que aterrice el dato.)*
>
> **Escenario B — Plan mixto reaccionario.** Invierte en transporte público pero después de que el sprawl ya ocurrió. Parámetros moderados: γ = 0.05, protección de cenotes de 200 metros.
>
> Resultado: 958 km², fragmentación 0.26, ROI 2.1 a 1. Es mejor que el A, pero todavía estamos pagando para corregir lo que pudimos prevenir.
>
> **Escenario C — Inversión estratégica, proactiva.** Aquí es donde aplica la Jerarquía de Movilidad: primero peatones, después ciclistas, después transporte público, al final el auto. Los parámetros son α = 0.55, β = 0.25, γ = 0.15 —protección kárstica fuerte—, ζ = 0.05.
>
> Resultado: la ciudad crece **solo 91 km²** hasta 911 km². Es compacta. La fragmentación baja a **0.20**. La temperatura de superficie **baja 0.9°C** — es la única de las tres proyecciones que reduce el calor urbano. La vulnerabilidad del acuífero se controla en **0.24**. La calidad de vida sube a 87 sobre 100. Y el retorno de inversión es de **4.5 a 1** — más del doble que el plan B.
>
> *(Aquí paso a la pestaña Vista 3D del dashboard y deslizo el año de 2024 a 2030.)*
>
> Como pueden ver en la visualización isométrica, las tres ciudades en 2030 son radicalmente distintas. La roja se expande sin control sobre el acuífero. La verde crece más ordenadamente. La morada —el escenario C— se mantiene compacta y deja corredor verde y zonas de protección de cenotes.

---

### [5:30 – 7:00] · LOS DATOS QUE LO RESPALDAN

> Para que estos números sean defendibles, les resumo las fuentes y el proceso de validación.
>
> **Datos de entrada:**
> - Imágenes satelitales **LANDSAT 8 y 9** entre 2015 y 2024, descargadas del USGS. Calculé NDVI (vegetación), LST (temperatura de superficie) e índices de urbanización.
> - Capas del **INEGI**: manzanas urbanas, vialidades, densidad de población por AGEB.
> - **SEDUMA Yucatán**: programa de ordenamiento ecológico territorial y ubicación de cenotes.
> - **CONAGUA / IMTA**: datos del acuífero kárstico y zonas de recarga.
>
> **Validación del modelo:**
> Entrené con datos 2015–2022 y validé prediciendo 2023. El error medio fue menor al 8% por celda. Luego corrí la simulación prospectiva 2024–2030 bajo cada combinación de parámetros.
>
> **Variables del modelo** — son diez features espaciales por celda de 30 metros:
> pendiente, distancia a vialidades, distancia a cenotes, distancia a zonas ya urbanizadas, NDVI actual, LST actual, densidad poblacional, tipo de suelo, restricción kárstica, y un factor de movilidad.
>
> Lo que hace el modelo de LightGBM es aprender, para cada celda, cuál es la probabilidad de pasar de "no urbanizada" a "urbanizada" entre un año y otro, en función de esas diez variables. El autómata celular luego aplica la regla de vecindad: si muchas celdas vecinas son urbanas, la probabilidad de la celda central aumenta. Así se reproduce el patrón de crecimiento por contigüidad que se observa en Mérida.
>
> **Esto no es una opinión. Es un modelo entrenado con datos públicos verificables.**

---

### [7:00 – 8:30] · EL MARCO LEGAL Y CIUDADANO

> Una de las partes que más me importa del proyecto es esta: **el Escenario C no es una preferencia política, es un requisito legal.**
>
> La Constitución, en su artículo 4°, establece el derecho a la movilidad en condiciones de seguridad, accesibilidad, eficiencia y sostenibilidad. El artículo 1° obliga a todas las autoridades a garantizar derechos humanos.
>
> El Plan Estatal de Desarrollo "Renacimiento Maya 2024–2030" prioriza explícitamente transporte público, ciclovías y movilidad activa.
>
> La Ley de Movilidad y Seguridad Vial de Yucatán, reformada en diciembre de 2025, tiene un artículo —el 39— que es la pieza clave de todo este estudio: **obliga a los municipios a desarrollar sus programas de movilidad con fundamentación en datos y análisis de demanda.** Este proyecto es exactamente eso.
>
> La NOM-004-SEDATU-2023 dice que las vialidades deben diseñarse conforme a su uso real. Y el Periférico de Mérida ya no es una vía de rodeo: es una vialidad urbana rodeada de hospitales, universidades y fraccionamientos. Eso obliga a reducir velocidades y priorizar al peatón.
>
> Y hay jurisprudencia de la SCJN que exige coexistencia de modos de transporte y accesibilidad universal.
>
> Además, integré al análisis las **8 propuestas del Colectivo Haciendo Ciudad** —Cicloturixes, Observatorio de Movilidad Sostenible, Poder AntiGandalla, Reflexión Acción Feminista— y las ocho aparecen reflejadas en la asignación de los 1,500 MDP.
>
> Cuando una decisión cumple simultáneamente Constitución, ley estatal, norma técnica, jurisprudencia, plan de desarrollo y propuestas ciudadanas, deja de ser debatible como preferencia: **se vuelve exigible.**

---

### [8:30 – 9:30] · LA RECOMENDACIÓN Y LA DISTRIBUCIÓN DEL DINERO

> Mi recomendación formal, como investigador, es adoptar el Escenario C como marco de referencia para SEDUMA, los municipios de la Zona Metropolitana de Mérida y las autoridades estatales.
>
> Los 1,600 millones asignados se distribuyen así — y aquí los datos concretos:
>
> - **400 MDP** al sistema BRT de dos líneas troncales con 12 nodos de desarrollo orientado al transporte.
> - **300 MDP** a banquetas, cruces seguros a nivel de calle e infraestructura peatonal accesible.
> - **250 MDP** a una red ciclista protegida, continua y conectada — más de 100 km.
> - **200 MDP** a protección de cenotes y monitoreo del acuífero.
> - **200 MDP** a corredores verdes y parques urbanos.
> - **150 MDP** a integración tarifaria y cobertura de transporte público.
> - **50 MDP** a auditorías de seguridad vial y accesibilidad.
> - **50 MDP** a sistemas de información geográfica y monitoreo anual del modelo.
>
> Noten el orden: **peatones primero, ciclistas después, transporte público después, y al final el auto**. Esa es la Jerarquía de Movilidad, y el presupuesto la refleja.
>
> El retorno estimado es de **4.5 pesos por cada peso invertido**, se preservan **26 mil millones de pesos en tierra**, se ahorran **47 millones anuales en tratamiento de agua**, y se reduce la temperatura de superficie en **0.9°C** — un alivio concreto para la isla de calor urbana.

---

### [9:30 – 10:00] · CIERRE

> Para cerrar, tres ideas que quiero que se lleven:
>
> **Primera.** Mérida tiene una ventana de oportunidad corta. Si la inversión de 1,500 MDP se ejecuta con lógica tradicional, el daño al acuífero kárstico será irreversible en términos prácticos. Si se ejecuta con lógica preventiva, se gana tiempo y se mejora la calidad de vida de más de un millón de personas.
>
> **Segunda.** La ciencia de datos no sustituye la política pública: la informa. Un modelo predictivo no dice "qué se debe hacer", pero dice "qué pasa si se hace A, B o C". Esa información es la que permite a un gobierno defender una decisión con evidencia.
>
> **Tercera.** Y la más importante para mí como estudiante de Ciencia de Datos: **los datos mejoraron cuando se cruzaron con voces ciudadanas y con marco legal.** El modelo puro no era suficiente. El modelo + las propuestas del Colectivo Haciendo Ciudad + la Ley de Movilidad + la NOM-004 es lo que hace que el Escenario C no sea discutible.
>
> El dashboard que están viendo es la herramienta que construí para que cualquier persona —funcionario, ciudadano, académico— pueda explorar estos datos sin necesitar un posgrado en machine learning.
>
> Mi nombre es Héctor Javier Raya Romo, soy estudiante de TSU en Ciencia de Datos en la Universidad Politécnica de Yucatán, y este es el **Mérida Urban Sprawl Predictor**.
>
> Gracias. Quedo abierto a sus preguntas.

---

## NOTAS DE APOYO PARA EL EXPOSITOR

### Cómo usar el dashboard durante el speech

| Momento del speech | Pestaña a mostrar | Qué señalar |
|---|---|---|
| 0:00 apertura | (ninguna, solo hablar) | — |
| 2:00 pregunta | **Resumen** | "Hallazgo Principal: Escenario C" |
| 3:30 escenarios | **Vista 3D** | Deslizar de 2024 a 2030 en tiempo real |
| 5:30 datos | **3 Escenarios** | Las tablas con α/β/γ/ζ |
| 7:00 legal | **Indicadores** | Tabla comparativa + gráfica de LST (la única que baja) |
| 7:30 ciudadano | **Propuestas Ciudadanas** | Tabla de 8 propuestas con ✓ |
| 7:45 legal | **Marco Legal** | Resaltar Art. 39 |
| 8:30 dinero | **Inversión 1,500 MDP** | Cada bloque verde con monto |
| 9:00 cierre | **Comparativa Final** | El bloque "Adoptar Escenario C" |
| Q&A técnico | **Planificador IA** | Mover sliders en vivo si preguntan por sensibilidad |

### Frases de transición útiles

- *"Antes de seguir, vale la pena detenerse en este dato..."*
- *"Comparemos las tres barras rojas, ámbar y verde..."* (apunta a la pantalla)
- *"¿Por qué este número es importante? Porque..."*
- *"Si alguien me preguntara cuál es la cifra que más me preocupa de toda la investigación, es esta..."*

### Posibles preguntas y respuestas cortas

| Pregunta | Respuesta corta |
|---|---|
| ¿Y si los datos de INEGI están mal? | El modelo se entrena con datos Landsat directamente, no con censo. INEGI es solo una capa auxiliar. |
| ¿Qué pasa si Mérida recibe más migración de la prevista? | El Escenario C es robusto: aún con 20% más migración, la fragmentación se mantiene bajo 0.25. |
| ¿Por qué no metiste LSTM o redes neuronales? | Por interpretabilidad regulatoria. La SCJN exige decisiones fundamentadas. Un modelo de caja negra no sirve. |
| ¿Esto es replicable en otros municipios? | Sí. El Art. 39 aplica a los 106 municipios. La metodología es genérica; los datos son los que cambian. |
| ¿Quién paga el modelo? | 50 MDP anuales de mantenimiento del modelo están incluidos en la línea de SIG y monitoreo. |
| ¿Cómo convenciste a SEDUMA? | El proyecto es el argumento. Cuando vean que cumple Art. 39, Plan Estatal, NOM-004 y propuestas ciudadanas simultáneamente, no hay margen para ignorarlo. |

---

## POWER PHRASES (frases de impacto para memorizar)

1. *"No hay ríos en Yucatán. Toda el agua sale del subsuelo."*
2. *"La pregunta no es si Mérida va a crecer, sino cómo."*
3. *"Más carriles, más coches, más carriles — eso es demanda inducida."*
4. *"El Escenario C no es preferencia política, es requisito constitucional."*
5. *"El retorno es de 4.5 pesos por cada peso invertido."*
6. *"Los datos mejoraron cuando se cruzaron con voces ciudadanas."*
7. *"Un modelo predictivo no dice qué se debe hacer, dice qué pasa si se hace A, B o C."*

---

**Tip final:** practica las primeras 3 frases hasta que salgan naturales. Si arrancas bien, el nerviosismo baja y el resto fluye.