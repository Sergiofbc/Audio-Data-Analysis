# Prueba técnica Creceré AI — Etapa 1: perfilado del audio crudo

Caracterización técnica de los 100 audios (50 humanos / 50 IA) **antes** de transcribir.
Sin llamadas a servicios externos: solo `numpy` y `scipy`.

## Por qué esta etapa existe

Tres decisiones del pipeline dependen de datos que están en la señal, no en el texto:
si hace falta diarizar, qué motor de STT contratar, y qué audios son poco confiables.
Medirlas primero cuesta minutos de CPU; equivocarse cuesta rehacer las 6.35 h de transcripción.

## Cómo se ejecuta

```bash
python3 src/audio_profile.py   <carpeta_raiz> data           # perfil_audio.csv + eventos_censura.json
python3 src/speech_activity.py <carpeta_raiz> data/actividad_voz.csv
```

`<carpeta_raiz>` debe contener `audios_humanos_censurados/` y `audios_ia_censurados/`.

## Qué mide cada script

**`audio_profile.py`** — una fila por audio.
Contenedor (códec, sample rate, canales, bits, duración), niveles (RMS, pico, DC offset,
clipping), piso de ruido y SNR, ocupación espectral, y detección de los pitidos de censura.

El detector de pitidos exige cuatro condiciones a la vez: frecuencia dominante en 1 kHz,
energía concentrada en ese bin (>0.95), amplitud plana (CV ≤ 0.20) y ≥100 ms de duración.
Un umbral laxo marca vocales sostenidas como pitidos; con el criterio estricto los eventos
detectados tienen todos frecuencia 1000.0 Hz y nivel −15.0 dBFS, lo que confirma que son
un tono sintético insertado y no habla.

**`speech_activity.py`** — una fila por audio.
VAD anclado al piso de ruido de cada llamada (no al pico, para que un call center ruidoso
no cuente como "hablando todo el rato"), turnos, pausas, tonos de red, y solapamiento.

El solapamiento se estima sin modelos: se busca la F0 dominante del frame, se cancela con
un filtro peine (`y[n] = x[n] − x[n−T]`), y se vuelve a buscar periodicidad en el residuo.
Si el residuo sigue siendo periódico a una F0 que no es armónico de la primera, hay dos
voces. **Validación:** medido sobre los pitidos —fuente única por construcción— el método
arroja 2.3 % de falso positivo. Ese es el piso del estimador y así se reporta.

## Hallazgos

| | Humano | IA | Mann–Whitney |
|---|---|---|---|
| Formato | PCM s16le, 8 kHz, **mono** | idem | 100/100 archivos |
| SNR (dB) | 41.5 | 53.8 | p < 1e-13 |
| Piso de ruido (dBFS) | −57.2 | −67.9 | p < 1e-12 |
| Habla solapada (%) | 11.4 | 7.3 | p < 1e-9 |
| Audio censurado (%) | 5.0 | 8.4 | p < 1e-5 |
| F0 mediana (Hz) | 216 | 148 | p < 1e-8 |

1. **Mono en los 100 archivos** → no hay split por canal; la diarización es obligatoria
   y su error se propaga a casi toda métrica conversacional.
2. **El ruido está de un solo lado.** 9/50 audios humanos bajo 35 dB de SNR y 19/50 sobre
   12 % de solapamiento; 0/50 en ambos casos para IA. Si el STT degrada con el ruido,
   degradará asimétricamente y sesgará la comparación contra los humanos. La calidad de
   audio entra al estudio como covariable, no como nota al pie.
3. **1 623 pitidos tapan 23.3 min (6.1 % del corpus).** Mediana de 0.78 s por pitido:
   lo suficiente para un nombre, una cédula o un monto. Un audio llega al 21 % tapado.
4. **El 8 kHz no pierde nada**: sobre 3 400 Hz vive el 0.03 % de la energía. Pedir mayor
   sample rate no recuperaría información que nunca se grabó.

## Salidas

- `data/perfil_audio.csv` — 24 columnas × 100 filas.
- `data/actividad_voz.csv` — 20 columnas × 100 filas.
- `data/eventos_censura.json` — inicio, fin, frecuencia y nivel de cada pitido.
  Alimenta la inserción de marcadores `[CENSURADO]` en la transcripción por timestamp.

## Siguiente etapa

Benchmark A/B de STT sobre 6 audios (3 humanos y 3 de IA, incluyendo los peores por SNR)
con una referencia transcrita a mano, midiendo WER y error de atribución de hablante.
Configuración idéntica para los dos grupos: un solo motor, un solo juego de keyterms.
