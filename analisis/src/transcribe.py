"""
Etapa 2 - Transcripcion completa con ElevenLabs.

Transcribe los 100 audios (50 humanos / 50 IA) con el motor elegido tras el
benchmark de la Etapa 1 (etapa1.ipynb, seccion 4): ElevenLabs gano en
confianza media, tasa de confianza baja y segmentacion de turnos frente a
Deepgram en los 6 casos comparados.

La respuesta cruda de la API se cachea por audio fuera de salidas/ (no es un
entregable) para no volver a pagar la transcripcion si el proceso se corta a
medias. El entregable es un .txt por audio, con turnos de hablante y los
huecos de censura marcados usando salidas/censura.json -- el que genera
etapa1.ipynb (seccion 2) tras calibrar y validar el detector de pitidos por
escucha. No data/eventos_censura.json: ese sale de audio_profile.py, un
script aparte con los mismos umbrales pero sin esa calibracion/validacion
documentada dentro del notebook.
"""

from __future__ import annotations

import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests

MODEL_ID = "scribe_v2"
LANGUAGE_CODE = "spa"
CARPETAS = {"humano": "audios_humanos_censurados", "ia": "audios_ia_censurados"}


def leer(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x, w.getframerate()


def elevenlabs(path: Path, api_key: str) -> dict:
    data = [("model_id", MODEL_ID), ("language_code", LANGUAGE_CODE), ("diarize", "true"),
            ("num_speakers", "2"), ("timestamps_granularity", "word")]
    audio = {"file": (path.name, path.read_bytes(), "audio/wav")}
    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                       headers={"xi-api-key": api_key}, data=data, files=audio, timeout=1800)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text[:500]}")
    return r.json()


def norm_elevenlabs(j: dict) -> list[dict]:
    return [{"text": w["text"], "start": w["start"], "end": w["end"],
             "speaker": w.get("speaker_id", "S0"),
             "conf": math.exp(w["logprob"]) if w.get("logprob") is not None else None}
            for w in j["words"] if w.get("type") == "word"]


def marcar_censura(words: list[dict], beeps: list, etiqueta: str = "[CENSURADO]") -> list[dict]:
    """Intercala marcadores de censura y elimina lo que el motor alucino sobre el pitido.

    `beeps` acepta tanto tuplas (ini, fin, hz, dbfs) -- formato de etapa1.ipynb -- como los
    dicts {inicio_s, fin_s, freq_hz, nivel_dbfs} de data/eventos_censura.json.
    """
    def _rango(e):
        return (e["inicio_s"], e["fin_s"]) if isinstance(e, dict) else (e[0], e[1])

    ev, out, k = [_rango(e) for e in beeps], [], 0
    for w in words:
        while k < len(ev) and ev[k][1] <= w["start"]:            # pitidos que ya quedaron atras
            out.append({**w, "text": etiqueta, "start": ev[k][0], "end": ev[k][1]}); k += 1
        if not any(a < w["end"] and w["start"] < b for a, b in ev):
            out.append(w)
    return out + [{"text": etiqueta, "start": a, "end": b, "speaker": ""} for a, b in ev[k:]]


def por_turnos(words: list[dict], max_pausa: float = 1.0) -> str:
    """Agrupa palabras en turnos de hablante -> texto legible con marca de tiempo."""
    turnos = []
    for w in words:
        if turnos and w["speaker"] == turnos[-1]["speaker"] and w["start"] - turnos[-1]["fin"] < max_pausa:
            turnos[-1]["texto"] += " " + w["text"]; turnos[-1]["fin"] = w["end"]
        else:
            turnos.append({"ini": w["start"], "fin": w["end"], "speaker": w["speaker"], "texto": w["text"]})
    return "\n".join(f"[{int(t['ini'])//60:02d}:{int(t['ini'])%60:02d}] {t['speaker']}: {t['texto']}"
                      for t in turnos)


def main(root: Path, censura_path: Path, out_dir: Path, cache_dir: Path, api_key: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    censura = json.loads(censura_path.read_text(encoding="utf-8"))

    audios = [(g, p) for g, c in CARPETAS.items() for p in sorted((root / c).glob("*.wav"))]
    print(f"{len(audios)} audios a transcribir")

    for i, (grupo, path) in enumerate(audios, 1):
        stem = path.stem
        txt_path = out_dir / grupo / f"{stem}.txt"
        cache_path = cache_dir / f"{grupo}__{stem}.json"
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            words = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            try:
                j = elevenlabs(path, api_key)
            except Exception as e:
                print(f"  [{i}/{len(audios)}] x {grupo}/{stem}: {e}")
                continue
            words = norm_elevenlabs(j)
            cache_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i}/{len(audios)}] {grupo}/{stem}: {len(words)} palabras ({time.time()-t0:.0f}s)")

        texto = por_turnos(marcar_censura(words, censura.get(path.name, [])))
        txt_path.write_text(texto, encoding="utf-8")

    print(f"\nListo -> {out_dir}")


if __name__ == "__main__":
    import os
    root_ = Path(sys.argv[1])
    censura_ = Path(sys.argv[2])
    out_ = Path(sys.argv[3])
    cache_ = Path(sys.argv[4]) if len(sys.argv) > 4 else out_.parent / "cache" / "elevenlabs"
    key_ = os.getenv("ELEVENLABS_API_KEY", "")
    if not key_:
        sys.exit("Falta ELEVENLABS_API_KEY en el entorno.")
    main(root_, censura_, out_, cache_, key_)
