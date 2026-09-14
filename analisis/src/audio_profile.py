"""
Etapa 1 - Perfilado del audio crudo (pre-transcripcion).

Objetivo: caracterizar tecnicamente los 100 audios ANTES de gastar un solo
credito de STT, para decidir con evidencia:
  - si hace falta diarizacion o basta con separar canales,
  - que motor de STT contratar,
  - que audios son riesgosos (mala calidad) y hay que marcar,
  - cuanto contenido esta censurado con pitidos.

Sin dependencias externas mas alla de numpy/scipy.
"""

from __future__ import annotations

import json
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# --- Parametros de analisis -------------------------------------------------
FRAME_MS = 25          # ventana de analisis
HOP_MS = 10            # salto entre ventanas
SILENCE_DB = -45       # umbral de silencio relativo al pico

# El pitido de censura es un tono sintetico: calibrado sobre la muestra resulta
# ser exactamente 1000 Hz, con toda la energia en un bin (tonalidad ~0.99),
# amplitud plana (CV < 0.15) y nivel constante -15 dBFS. La voz humana, aunque
# tonal en las vocales, nunca cumple las cuatro condiciones a la vez.
BEEP_FREQ_HZ = 1000.0
BEEP_FREQ_TOL = 40     # Hz de tolerancia alrededor del tono nominal
BEEP_TONALITY = 0.95   # energia minima concentrada en el bin dominante
BEEP_AMP_CV = 0.20     # variacion maxima de amplitud dentro del pitido
BEEP_MIN_MS = 100      # duracion minima para contar como pitido


@dataclass
class AudioProfile:
    file: str
    grupo: str
    # --- contenedor / codec
    sample_rate: int
    canales: int
    bits: int
    n_muestras: int
    duracion_s: float
    bytes: int
    # --- niveles
    rms_dbfs: float
    pico_dbfs: float
    dc_offset: float
    pct_clipping: float
    # --- ruido y voz
    ruido_dbfs: float          # p10 del RMS por frame
    voz_dbfs: float            # p90 del RMS por frame
    snr_db: float              # voz - ruido
    pct_silencio: float
    # --- banda util
    freq_p95_hz: float         # frecuencia bajo la cual esta el 95% de la energia
    energia_sobre_3400: float  # % de energia arriba de 3.4 kHz (banda telefonica)
    # --- huella de compresion previa
    niveles_unicos: int        # valores distintos de muestra (256 => venia de G.711)
    huella_g711: bool
    # --- censura
    n_pitidos: int
    pitidos_s: float
    pct_pitidos: float
    pitido_freq_hz: float
    # --- turnos (proxy, sin diarizacion)
    n_segmentos_voz: int
    dur_media_segmento_s: float


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def _frames(x: np.ndarray, sr: int) -> np.ndarray:
    """Corta la senal en frames solapados (matriz n_frames x n_muestras)."""
    n = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(x) < n:
        return np.zeros((0, n))
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i + n] for i in idx])


def _detect_beeps(frames: np.ndarray, sr: int, rms_fr: np.ndarray):
    """Detecta los pitidos de censura (tono puro de 1 kHz insertado sobre el dato sensible).

    Cuatro condiciones simultaneas por frame: frecuencia dominante en 1 kHz,
    energia concentrada en ese bin, y -a nivel de evento- amplitud plana y
    duracion suficiente. La voz cumple una o dos, nunca las cuatro.
    """
    if len(frames) == 0:
        return []

    win = np.hanning(frames.shape[1])
    energia = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sr)

    total = energia.sum(axis=1) + 1e-12
    dom = energia.argmax(axis=1)
    lo = np.clip(dom - 1, 0, energia.shape[1] - 1)
    hi = np.clip(dom + 1, 0, energia.shape[1] - 1)
    pico = np.array([energia[i, lo[i]:hi[i] + 1].sum() for i in range(len(dom))])

    es_tono = (
        (pico / total > BEEP_TONALITY)
        & (np.abs(freqs[dom] - BEEP_FREQ_HZ) <= BEEP_FREQ_TOL)
    )

    eventos = []
    min_frames = max(1, int(BEEP_MIN_MS / HOP_MS))
    i = 0
    while i < len(es_tono):
        if not es_tono[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(es_tono) and es_tono[j + 1]:
            j += 1
        n_fr = j - i + 1
        amp = rms_fr[i:j + 1]
        cv = float(amp.std() / (amp.mean() + 1e-12))
        if n_fr >= min_frames and cv <= BEEP_AMP_CV:
            eventos.append({
                "inicio_s": round(i * HOP_MS / 1000, 2),
                "fin_s": round((j * HOP_MS + FRAME_MS) / 1000, 2),
                "freq_hz": round(float(np.median(freqs[dom[i:j + 1]])), 1),
                "nivel_dbfs": round(_db(float(amp.mean())), 1),
            })
        i = j + 1

    return eventos


def _segmentos_voz(activos: np.ndarray) -> tuple[int, float]:
    """Cuenta bloques contiguos de actividad vocal (proxy de turnos)."""
    if activos.sum() == 0:
        return 0, 0.0
    d = np.diff(activos.astype(int))
    inicios = np.where(d == 1)[0] + 1
    finales = np.where(d == -1)[0] + 1
    if activos[0]:
        inicios = np.r_[0, inicios]
    if activos[-1]:
        finales = np.r_[finales, len(activos)]
    duraciones = (finales - inicios) * HOP_MS / 1000
    # ignoramos micro-segmentos (<200 ms): son ruido, no turnos
    duraciones = duraciones[duraciones >= 0.2]
    if len(duraciones) == 0:
        return 0, 0.0
    return int(len(duraciones)), float(duraciones.mean())


def profile(path: Path, grupo: str) -> tuple[AudioProfile, list]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if dtype == np.uint8:
        x = x - 128
    full_scale = float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x_norm = x / full_scale

    # --- niveles globales
    rms = float(np.sqrt(np.mean(x_norm ** 2)))
    pico = float(np.max(np.abs(x_norm)))
    dc = float(np.mean(x_norm))
    clipped = float(np.mean(np.abs(x_norm) >= 0.999) * 100)

    # --- frames
    fr = _frames(x_norm, sr)
    rms_fr = np.sqrt((fr ** 2).mean(axis=1)) if len(fr) else np.array([0.0])
    ruido = float(np.percentile(rms_fr, 10))
    voz = float(np.percentile(rms_fr, 90))
    umbral = max(voz * 10 ** (SILENCE_DB / 20), 1e-6)
    activos = rms_fr > umbral
    pct_sil = float((~activos).mean() * 100)

    # --- contenido espectral promedio (solo sobre frames con voz)
    if len(fr) and activos.sum() > 0:
        win = np.hanning(fr.shape[1])
        spec = np.abs(np.fft.rfft(fr[activos] * win, axis=1)) ** 2
        freqs = np.fft.rfftfreq(fr.shape[1], 1 / sr)
        media = spec.mean(axis=0)
        acum = np.cumsum(media) / media.sum()
        f95 = float(freqs[np.searchsorted(acum, 0.95)])
        e_alta = float(media[freqs > 3400].sum() / media.sum() * 100)
    else:
        f95, e_alta = 0.0, 0.0

    # --- huella de compresion previa (G.711 deja 256 niveles)
    niveles = int(len(np.unique(x.astype(np.int64))))

    # --- censura
    eventos = _detect_beeps(fr, sr, rms_fr) if len(fr) else []
    dur_beeps = sum(e["fin_s"] - e["inicio_s"] for e in eventos)
    freq_beep = float(np.median([e["freq_hz"] for e in eventos])) if eventos else 0.0

    n_seg, dur_seg = _segmentos_voz(activos)
    dur = n / sr

    p = AudioProfile(
        file=path.name,
        grupo=grupo,
        sample_rate=sr,
        canales=ch,
        bits=sw * 8,
        n_muestras=n,
        duracion_s=round(dur, 2),
        bytes=path.stat().st_size,
        rms_dbfs=round(_db(rms), 2),
        pico_dbfs=round(_db(pico), 2),
        dc_offset=round(dc, 6),
        pct_clipping=round(clipped, 4),
        ruido_dbfs=round(_db(ruido), 2),
        voz_dbfs=round(_db(voz), 2),
        snr_db=round(_db(voz) - _db(ruido), 2),
        pct_silencio=round(pct_sil, 2),
        freq_p95_hz=round(f95, 1),
        energia_sobre_3400=round(e_alta, 3),
        niveles_unicos=niveles,
        huella_g711=niveles <= 300,
        n_pitidos=len(eventos),
        pitidos_s=round(dur_beeps, 2),
        pct_pitidos=round(dur_beeps / dur * 100, 2) if dur else 0.0,
        pitido_freq_hz=round(freq_beep, 1),
        n_segmentos_voz=n_seg,
        dur_media_segmento_s=round(dur_seg, 2),
    )
    return p, eventos


def main(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grupos = {
        "audios_humanos_censurados": "humano",
        "audios_ia_censurados": "ia",
    }
    filas, censura = [], {}
    for carpeta, grupo in grupos.items():
        for f in sorted((root / carpeta).glob("*.wav")):
            p, eventos = profile(f, grupo)
            filas.append(asdict(p))
            censura[f.name] = eventos
            print(f"{grupo:7s} {f.name[:8]} {p.duracion_s:7.1f}s "
                  f"SNR={p.snr_db:5.1f}dB beeps={p.n_pitidos:2d}", flush=True)

    import csv
    with open(out_dir / "perfil_audio.csv", "w", newline="", encoding="utf-8") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(filas)
    with open(out_dir / "eventos_censura.json", "w", encoding="utf-8") as fh:
        json.dump(censura, fh, ensure_ascii=False, indent=1)
    print(f"\n{len(filas)} audios perfilados -> {out_dir/'perfil_audio.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
