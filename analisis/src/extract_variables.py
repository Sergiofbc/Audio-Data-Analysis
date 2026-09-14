"""
Etapa 3 - Construccion de variables de negociacion desde las transcripciones.

Convierte cada .txt de salidas/transcripciones/{humano,ia}/ (turnos de hablante +
[CENSURADO], generados en la Etapa 2 con ElevenLabs) en una fila de
data/gestion_variables.csv: tipo de gestion, monto, descuento, resultado de la
llamada, objeciones, quien corta y por que. Son variables de lenguaje libre
(montos dichos de palabra, resultados implicitos en como termina la conversacion),
no patrones acusticos como los pitidos de la Etapa 1: no hay una regla determinista
razonable para "leer" un monto en prosa o decidir si una llamada cerro o no, asi
que la extraccion es semantica, no de texto plano.

CONTEXTO IMPORTANTE - como se corrio de verdad esta vez
---------------------------------------------------------
El plazo de entrega no daba tiempo para levantar y probar un pipeline nuevo contra
una API de LLM (no habia ELEVENLABS_API_KEY/DEEPGRAM_API_KEY para esto: son motores
de voz-a-texto, no LLMs de texto). La extraccion real se hizo con Claude Code leyendo
directamente las 100 transcripciones -- repartidas en 4 lotes de 25 corridos en
paralelo -- aplicando el ESQUEMA y el PROMPT que estan abajo, palabra por palabra.
Ese es el metodo documentado en `etapa3.ipynb`, junto con la validacion manual sobre
25/100 filas (releyendo el .txt original contra lo extraido) y las 2 correcciones que
salieron de ahi.

Este archivo dejaria el mismo esquema y el mismo prompt listos para correrse como
pipeline real contra un LLM de texto (Claude, GPT, etc.), con cache por archivo igual
que `transcribe.py` -- si alguien quiere reproducir la extraccion con su propia API
key en vez de confiar en la lectura de Claude Code documentada en el notebook.
`main()` no se ejecuto en esta entrega: `llamar_llm()` esta escrita pero no probada
contra una API real (no habia key de LLM de texto disponible).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CARPETAS = {"humano": "audios_humanos_censurados", "ia": "audios_ia_censurados"}

# --- Esquema ------------------------------------------------------------
# 24 campos por llamada. Enums documentados aqui para que sean el mismo criterio
# en cualquier corrida (con Claude Code leyendo directo, o via API).
CAMPOS = [
    "file",                    # UUID del audio, sin extension
    "grupo",                   # "humano" | "ia" (ya se sabe por la carpeta)
    "tipo_gestion",            # nueva_oferta | seguimiento_o_renegociacion | otro -- ver nota abajo
    "contacto_efectivo",       # bool -- ¿se habla con el titular o alguien que puede decidir?
    "de_donde_llaman",         # string o null, tal como se menciona
    "origen_deuda",            # string o null
    "monto_total_actual_cop",  # numero o null
    "descuento_pct",           # numero o null
    "descuento_monto_cop",     # numero o null
    "valor_a_pagar_cop",       # numero o null
    "n_cuotas",                # entero o null
    "fecha_max_pago",          # string tal cual se dice, o null
    "fecha_pago_acordada",     # string tal cual se dice, o null
    "comparte_medios_pago",    # bool o null
    "cliente_puede_pagar",     # si | no | tal_vez | no_queda_claro
    "objecion_cliente",        # bool
    "tipo_objecion",           # string breve o null
    "alternativa_propuesta",   # string breve o null
    "resultado_llamada",       # cierre_1ra_oferta | cierre_2da_oferta_o_alternativa |
                                # fecha_acordada_sin_descuento | sin_acuerdo | llamada_cortada |
                                # no_es_titular | otro
    "caso_exito",              # bool -- SE RECALCULA de forma determinista a partir de
                                # resultado_llamada al consolidar (ver notebook), no se confia
                                # en el booleano que devuelva el extractor
    "quien_corta_la_llamada",  # agente | cliente | no_aplica | se_corta_tecnicamente
    "motivo_fin_llamada",      # string breve
    "confianza_extraccion",    # alta | media | baja -- guia la muestra de validacion manual
    "notas",                   # string breve opcional
]

# --- Prompt --------------------------------------------------------------
# Texto real usado en los 4 lotes (salvo la lista de archivos), CON UNA CORRECCION:
# el prompt original pedia distinguir `seguimiento_acuerdo` de `renegociacion_fecha`.
# La validacion post-entrega mostro que esa frontera no se sostiene desde el texto
# (ej. el caso `77d258d0`, con la misma firma -- objecion "no tiene plata" + fecha que
# se mueve -- pero clasificado en el otro grupo), asi que aqui ya aparecen fusionados
# en `seguimiento_o_renegociacion`; ver etapa3.ipynb, seccion 2.2, para el detalle.
# `null` significa "no se dice o cae en un [CENSURADO]" -- nunca se infiere un valor
# que no este en el texto.
PROMPT_TEMPLATE = """\
Lee cada uno de estos archivos completos (texto plano con turnos
`[mm:ss] speaker_N: texto`, y `[CENSURADO]` donde hubo un pitido que tapo el audio):

{lista_archivos}

Para cada archivo, extrae ESTRICTAMENTE del texto (nunca inventes ni infieras un
valor que no este dicho o implicado de forma inequivoca; si el dato cae dentro de
un `[CENSURADO]` o simplemente no se menciona, usa `null`) estos campos:

- `tipo_gestion`: uno de `nueva_oferta` (se ofrece descuento/condonacion nueva sobre
  la deuda, sin referirse a un acuerdo previo ya pactado), `seguimiento_o_renegociacion`
  (la llamada gira en torno a un acuerdo/compromiso YA formalizado antes -- recordatorio,
  confirmacion de medios de pago, o el cliente pide mover la fecha pactada), `otro`.
- `contacto_efectivo`: true/false -- ¿habla con el titular o alguien que puede decidir?
- `de_donde_llaman`, `origen_deuda`: string o null.
- `monto_total_actual_cop`, `descuento_pct`, `descuento_monto_cop`, `valor_a_pagar_cop`:
  numeros o null.
- `n_cuotas`: entero o null.
- `fecha_max_pago`, `fecha_pago_acordada`: string tal cual se dice, o null.
- `comparte_medios_pago`: true/false/null.
- `cliente_puede_pagar`: si | no | tal_vez | no_queda_claro.
- `objecion_cliente`: bool. `tipo_objecion`: string breve o null.
- `alternativa_propuesta`: string breve o null.
- `resultado_llamada`: cierre_1ra_oferta | cierre_2da_oferta_o_alternativa |
  fecha_acordada_sin_descuento | sin_acuerdo | llamada_cortada | no_es_titular | otro.
- `quien_corta_la_llamada`: agente | cliente | no_aplica | se_corta_tecnicamente.
- `motivo_fin_llamada`: string breve.
- `confianza_extraccion`: alta/media/baja -- baja si el audio esta muy censurado, la
  transcripcion es confusa, o hubo que interpretar mucho.
- `notas`: string breve opcional.

Devuelve un array JSON de objetos, uno por archivo, en el mismo orden de la lista,
con `file` (nombre sin ".txt") y `grupo` incluidos en cada uno.
"""

# Prompt liviano de la segunda pasada (tono/empatia), deliberadamente mas simple:
# es la capa 5 (Sentimiento) del analisis, exploratoria, no con el mismo rigor que
# el resto del esquema.
PROMPT_SENTIMIENTO = """\
Para cada uno de estos archivos ya leidos antes, extrae rapido (sin sobre-analizar;
usa "no_aplica" si el archivo es muy corto o censurado para juzgar):

{lista_archivos}

- `tono_inicial_cliente` / `tono_final_cliente`: colaborador | neutral | reacio |
  molesto | desconfiado | no_aplica.
- `cambio_tono`: mejora | empeora | igual | no_aplica.
- `agente_se_adapta`: true/false/null -- ¿el agente cambia su enfoque en respuesta a
  algo especifico que dijo el cliente, o repite el mismo guion sin importar la
  respuesta? null si no aplica (nadie contesto).

Devuelve un array JSON con `file` y estos 4 campos, mismo orden.
"""


def build_prompt(archivos: list[str], sentimiento: bool = False) -> str:
    tpl = PROMPT_SENTIMIENTO if sentimiento else PROMPT_TEMPLATE
    return tpl.format(lista_archivos="\n".join(archivos))


def llamar_llm(prompt: str, api_key: str) -> str:
    """Placeholder de referencia -- NO se ejecuto en esta entrega (no habia API key
    de un LLM de texto en el entorno). Documenta como se conectaria un motor real,
    con el mismo patron de cache-por-archivo de transcribe.py."""
    raise NotImplementedError(
        "No se corrio contra ninguna API en esta entrega. La extraccion real la "
        "hizo Claude Code leyendo los .txt directamente -- ver metodologia y "
        "validacion manual en etapa3.ipynb."
    )


def main(root: Path, out_csv: Path) -> None:
    archivos = [(g, p) for g, c in CARPETAS.items()
                for p in sorted((root / "salidas" / "transcripciones" / g).glob("*.txt"))]
    print(f"{len(archivos)} transcripciones encontradas. Este script documenta el "
          f"esquema y el prompt; no ejecuta la extraccion (ver docstring del modulo).")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."),
         Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/gestion_variables.csv"))
