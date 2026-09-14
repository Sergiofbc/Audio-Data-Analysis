# Audio Data Analysis — IA vs. Agente Humano en Cobranza

Prueba técnica Creceré AI (Data Scientist / Data Analyst Junior). Compara el desempeño de
un agente de IA contra agentes humanos en gestión de cartera, a partir de 100 llamadas
grabadas (50 humanas / 50 IA).

**Entregable ejecutivo:** [`reporte_final.html`](reporte_final.html) — ábrelo en el navegador.
**Todo el desarrollo:** carpeta [`analisis/`](analisis/).

## Estructura

| Etapa | Qué hace | Notebook |
|---|---|---|
| 1 | Perfilado técnico del audio crudo (formato, ruido, censura, actividad de voz) y benchmark de motores de STT | [`analisis/etapa1.ipynb`](analisis/etapa1.ipynb) |
| 2 | Transcripción completa de los 100 audios con diarización (ElevenLabs) | [`analisis/etapa2.ipynb`](analisis/etapa2.ipynb) |
| 3 | Extracción de variables de negociación, validación manual, y comparación estadística Humano vs. IA en 5 capas | [`analisis/etapa3.ipynb`](analisis/etapa3.ipynb) |

Detalle de cada etapa en [`analisis/README.md`](analisis/README.md).

## Cómo evolucionó el planteamiento

La pregunta inicial (Etapa 0) era simple: ¿la IA logra los mismos casos de éxito que el
agente humano? Al construir las variables sobre las 100 transcripciones, esa pregunta
resultó **menos directamente accionable de lo que parecía al principio**, por dos razones
que vale la pena dejar explícitas en vez de esconder:

1. **La muestra no es un experimento controlado.** El brief de la prueba lo dice desde el
   principio: los audios "provienen de distintas campañas y momentos de gestión". No es
   el mismo caso repartido al azar entre agente humano y agente IA — son poblaciones de
   llamadas distintas desde el origen, y eso hay que asumirlo antes de comparar nada.
2. **El corpus mezcla dos casos de uso, no uno.** La comparación inicial ("¿quién cierra
   más?") asumía implícitamente que todas las llamadas eran del mismo tipo de gestión.
   La extracción de variables (Etapa 3) mostró que no: la IA hace casi exclusivamente
   **negociación nueva** (oferta de descuento sobre una deuda, 74% de sus llamadas),
   mientras que el 54% de las llamadas humanas son **seguimiento/recaudo** sobre un
   acuerdo que ya existía (recordatorio de cuota, confirmación de medios de pago,
   renegociación de fecha) — una tarea estructuralmente más simple, sin negociación de
   descuento, y con una tasa de éxito humana bastante más alta (70% vs. 44% en
   negociación nueva).

Comparar "todo contra todo" sin esa distinción sobreestima la brecha entre agentes.
El análisis final (`etapa3.ipynb`) compara siempre **el mismo tipo de gestión**, y trata
la brecha en seguimiento no como una debilidad de la IA hoy, sino como una oportunidad de
alcance: es la tarea más simple y ya se sabe que se puede automatizar bien.

## Variables construidas

- **Etapa 1:** `perfil_audio.csv` y `actividad_voz.csv` — 44 métricas acústicas objetivas
  (SNR, censura, turnos, solapamiento, F0) sobre los 100 audios crudos, sin depender de
  ningún LLM.
- **Etapa 3:** `gestion_variables.csv` — 28 variables de negociación por llamada (tipo de
  gestión, monto, descuento, resultado, objeciones, quién corta la llamada, tono del
  cliente), extraídas semánticamente de las transcripciones y validadas a mano sobre el
  25% de la muestra. Esquema y prompt documentados en `analisis/src/extract_variables.py`.

## Licencias / datos

Los audios en crudo, el entorno virtual y las claves de API **no están en este repositorio**
(ver `.gitignore`) — son datos sensibles de la prueba técnica, no artefactos de código.
