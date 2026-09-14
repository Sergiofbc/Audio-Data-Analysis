"""
Etapa 1b - Actividad de voz, turnos y solapamiento (sin modelos, sin STT).

El perfil basico mostro que los 100 audios son MONO. Eso obliga a diarizar y
hace que el solapamiento sea un riesgo real para la transcripcion, no una
curiosidad. Aqui se mide, sin llamar a ningun servicio:

  1. VAD robusto: umbral anclado al piso de ruido de cada llamada (no al pico),
     para que un call center ruidoso no se cuente como "hablando todo el rato".
  2. Turnos y pausas.
  3. Solapamiento: proxy por doble periodicidad. En un frame sonoro se estima
     la F0 dominante; luego se cancelan esa F0 y sus armonicos con un filtro
     peine y se vuelve a buscar periodicidad en el residuo. Si el residuo sigue
     siendo fuertemente periodico a una F0 que no es multiplo ni submultiplo de
     la primera, hay dos voces sonando a la vez.
  4. Tonos de red (ringback / ocupado / DTMF) que no son habla y ensucian el STT.
"""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np

FRAME_MS = 32
HOP_MS = 16
VAD_OVER_NOISE_DB = 12      # cuanto debe superar el frame al piso de ruido
VAD_ABS_DBFS = -52          # suelo absoluto: por debajo no es voz util
F0_MIN, F0_MAX = 70, 320    # rango de F0 de voz adulta
VOICED_R = 0.45             # autocorrelacion normalizada minima para "sonoro"
OVERLAP_R = 0.35            # periodicidad minima del residuo para 2a voz
NETWORK_TONES = {425: "ringback/ocupado (LatAm)", 440: "tono de linea", 350: "dial (US)"}


def load(path: Path):
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def framing(x, sr):
    n, hop = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    if len(x) < n:
        return np.zeros((0, n)), hop
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i + n] for i in idx]), hop


def norm_autocorr(frames: np.ndarray) -> np.ndarray:
    """Autocorrelacion normalizada por frame, via FFT."""
    n = frames.shape[1]
    f = np.fft.rfft(frames - frames.mean(axis=1, keepdims=True), n=2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :n]
    return ac / (ac[:, :1] + 1e-12)


def f0_from_ac(ac: np.ndarray, sr: int):
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    hi = min(hi, ac.shape[1] - 1)
    seg = ac[:, lo:hi]
    lag = seg.argmax(axis=1) + lo
    r = seg.max(axis=1)
    return sr / lag, r, lag


def comb_cancel(frames: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """Resta a cada frame una copia de si mismo desplazada un periodo.

    y[n] = x[n] - x[n-T]  cancela la componente periodica de periodo T
    (y sus armonicos) y deja el resto: otra voz, si la hay.
    """
    out = np.zeros_like(frames)
    for i, T in enumerate(lag):
        T = int(T)
        if T <= 0 or T >= frames.shape[1]:
            out[i] = frames[i]
            continue
        out[i, T:] = frames[i, T:] - frames[i, :-T]
    return out


def network_tones(frames, sr, rms):
    """Segundos ocupados por tonos de red (ringback, ocupado): no son habla.

    Criterio estricto a proposito. Una F0 femenina de ~215 Hz pone su segundo
    armonico justo encima de 425/440 Hz, asi que un umbral laxo marca voz como
    tono de red. Se exige tono casi puro (>0.9), amplitud plana y medio segundo
    continuo, que la voz no sostiene.
    """
    if len(frames) == 0:
        return 0.0, ""
    win = np.hanning(frames.shape[1])
    e = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    dom = e.argmax(axis=1)
    lo = np.clip(dom - 1, 0, e.shape[1] - 1)
    hi = np.clip(dom + 1, 0, e.shape[1] - 1)
    pico = np.array([e[i, lo[i]:hi[i] + 1].sum() for i in range(len(dom))])
    tonal = (pico / (e.sum(axis=1) + 1e-12)) > 0.90

    total_s, etiquetas = 0.0, []
    min_fr = max(1, int(500 / HOP_MS))
    for f, name in NETWORK_TONES.items():
        m = tonal & (np.abs(freqs[dom] - f) <= 20) & (rms > 10 ** (-60 / 20))
        # exigir rachas continuas de >=500 ms con amplitud estable
        i, seg = 0, 0.0
        while i < len(m):
            if not m[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            a = rms[i:j + 1]
            if (j - i + 1) >= min_fr and a.std() / (a.mean() + 1e-12) <= 0.25:
                seg += (j - i + 1) * HOP_MS / 1000
            i = j + 1
        if seg > 0:
            total_s += seg
            etiquetas.append(name)
    return total_s, "; ".join(etiquetas)


def analyse(path: Path, grupo: str) -> dict:
    x, sr = load(path)
    frames, hop = framing(x, sr)
    if len(frames) == 0:
        return {}

    rms = np.sqrt((frames ** 2).mean(axis=1))
    piso = np.percentile(rms, 5)
    umbral = max(piso * 10 ** (VAD_OVER_NOISE_DB / 20), 10 ** (VAD_ABS_DBFS / 20))
    voz = rms > umbral

    # --- periodicidad
    ac = norm_autocorr(frames)
    f0, r, lag = f0_from_ac(ac, sr)
    sonoro = voz & (r > VOICED_R)

    # --- segunda voz en el residuo
    res = comb_cancel(frames, lag)
    ac2 = norm_autocorr(res)
    f0b, r2, _ = f0_from_ac(ac2, sr)
    ratio = np.maximum(f0, f0b) / np.maximum(np.minimum(f0, f0b), 1e-9)
    no_armonico = np.min(np.abs(ratio[:, None] - np.arange(1, 5)[None, :]), axis=1) > 0.25
    # el residuo debe conservar energia: si cancelo casi todo, no habia 2a voz
    energia_res = np.sqrt((res ** 2).mean(axis=1)) / (rms + 1e-12)
    solape = sonoro & (r2 > OVERLAP_R) & no_armonico & (energia_res > 0.3)

    dur = len(x) / sr
    seg_voz = voz.sum() * hop / sr

    # turnos y pausas sobre el VAD robusto
    d = np.diff(voz.astype(int))
    ini = np.where(d == 1)[0] + 1
    fin = np.where(d == -1)[0] + 1
    if voz[0]:
        ini = np.r_[0, ini]
    if voz[-1]:
        fin = np.r_[fin, len(voz)]
    durs = (fin - ini) * hop / sr
    durs = durs[durs >= 0.25]
    pausas = (ini[1:] - fin[:-1]) * hop / sr if len(ini) > 1 else np.array([])
    pausas = pausas[pausas >= 0.2]

    tonos_s, tono_lbl = network_tones(frames, sr, rms)

    return {
        "file": path.name,
        "grupo": grupo,
        "duracion_s": round(dur, 2),
        "voz_s": round(seg_voz, 2),
        "pct_voz": round(seg_voz / dur * 100, 2),
        "pct_silencio": round(100 - seg_voz / dur * 100, 2),
        "piso_ruido_dbfs": round(20 * np.log10(piso + 1e-12), 2),
        "n_turnos": int(len(durs)),
        "turnos_por_min": round(len(durs) / dur * 60, 2),
        "dur_turno_mediana_s": round(float(np.median(durs)) if len(durs) else 0, 2),
        "dur_turno_p90_s": round(float(np.percentile(durs, 90)) if len(durs) else 0, 2),
        "pausa_mediana_s": round(float(np.median(pausas)) if len(pausas) else 0, 2),
        "pausa_max_s": round(float(pausas.max()) if len(pausas) else 0, 2),
        "pct_sonoro": round(sonoro.mean() * 100, 2),
        "pct_solape_sobre_voz": round(solape.sum() / max(sonoro.sum(), 1) * 100, 2),
        "solape_s": round(solape.sum() * hop / sr, 2),
        "f0_mediana_hz": round(float(np.median(f0[sonoro])) if sonoro.sum() else 0, 1),
        "f0_iqr_hz": round(float(np.subtract(*np.percentile(f0[sonoro], [75, 25]))) if sonoro.sum() else 0, 1),
        "tonos_red_s": round(tonos_s, 2),
        "tonos_red_tipo": tono_lbl,
    }


def main(root: Path, out: Path):
    filas = []
    for carpeta, grupo in [("audios_humanos_censurados", "humano"), ("audios_ia_censurados", "ia")]:
        for f in sorted((root / carpeta).glob("*.wav")):
            r = analyse(f, grupo)
            if r:
                filas.append(r)
                print(f"{grupo:7s} {f.name[:8]} voz={r['pct_voz']:5.1f}% turnos/min={r['turnos_por_min']:5.1f} "
                      f"solape={r['pct_solape_sobre_voz']:5.1f}% F0={r['f0_mediana_hz']:5.1f}Hz", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{len(filas)} audios -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
