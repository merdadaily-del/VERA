import json
import os
from typing import Any

from openai import OpenAI


# ============================================================
# VERA - DEEPSEEK AI ENGINE
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Modello DeepSeek utilizzato da VERA.
# Può essere cambiato da Render senza modificare il codice.
DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
)


class AIError(Exception):
    pass


# ============================================================
# CLIENT DEEPSEEK
# ============================================================

def _client() -> OpenAI:

    if not DEEPSEEK_API_KEY:
        raise AIError(
            "DEEPSEEK_API_KEY non configurata su Render"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# ============================================================
# JSON PARSER
# ============================================================

def _json_from_text(
    text: str
) -> dict[str, Any]:

    if not text:
        raise AIError(
            "DeepSeek ha restituito una risposta vuota"
        )

    text = text.strip()

    # Rimuove eventuali blocchi Markdown
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:

        result = json.loads(text)

    except Exception as exc:

        raise AIError(
            f"Risposta DeepSeek non valida come JSON: {exc}"
        ) from exc

    if not isinstance(result, dict):

        raise AIError(
            "DeepSeek ha restituito un JSON non valido"
        )

    return result


# ============================================================
# GENERATORE JSON
# ============================================================

def generate_json(
    prompt: str,
    max_tokens: int = 1400
) -> dict[str, Any]:

    client = _client()

    try:

        response = client.chat.completions.create(

            model=DEEPSEEK_MODEL,

            temperature=0.1,

            max_tokens=max_tokens,

            messages=[

                {
                    "role": "system",

                    "content": """
Sei il motore editoriale di VERA.

Sei un assistente giornalistico rigoroso.

Devi usare esclusivamente le informazioni
contenute nelle fonti che ti vengono fornite.

NON inventare:
- fatti
- nomi
- numeri
- date
- luoghi
- dichiarazioni
- collegamenti tra eventi

Devi distinguere sempre tra:
- fatti;
- dichiarazioni;
- accuse;
- informazioni riportate;
- interpretazioni.

Quando le fonti non consentono una conferma
sufficiente, devi dirlo chiaramente.

Rispondi esclusivamente con JSON valido.
""",
                },

                {
                    "role": "user",
                    "content": prompt,
                },

            ],

            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:

        raise AIError(
            f"Errore API DeepSeek: {exc}"
        ) from exc

    try:

        content = response.choices[0].message.content

    except Exception as exc:

        raise AIError(
            f"Risposta DeepSeek inattesa: {exc}"
        ) from exc

    return _json_from_text(content)


# ============================================================
# RIDUZIONE DELLE FONTI
# ============================================================

def _compact_sources(
    sources: list[dict[str, Any]],
    max_sources: int = 3
) -> list[dict[str, Any]]:

    compact = []

    for article in sources[:max_sources]:

        compact.append({

            "id": article.get(
                "id"
            ),

            "outlet": article.get(
                "outlet"
            ),

            "published": article.get(
                "published"
            ),

            "title": str(
                article.get(
                    "title",
                    ""
                )
            )[:300],

            "summary": str(
                article.get(
                    "summary",
                    ""
                )
            )[:500],

        })

    return compact


# ============================================================
# VERIFICA DI UN EVENTO
# ============================================================

def verify_event(
    event: dict[str, Any]
) -> dict[str, Any]:

    sources = _compact_sources(

        event.get(
            "articles",
            []
        ),

        max_sources=3
    )

    event_data = {

        "id": event.get(
            "id"
        ),

        "title": str(
            event.get(
                "title",
                ""
            )
        )[:400],

        "category": event.get(
            "category",
            ""
        ),

    }

    prompt = f"""
VERIFICA EDITORIALE VERA

Devi verificare un evento giornalistico.

Usa ESCLUSIVAMENTE le fonti fornite.

NON usare conoscenze esterne.

REGOLE:

1. Non considerare automaticamente vera
   un'informazione solo perché compare
   in una fonte.

2. Distingui sempre tra:
   - fatti;
   - dichiarazioni;
   - accuse;
   - interpretazioni.

3. Usa CONFIRMED soltanto quando le fonti
   forniscono elementi sufficienti per
   considerare il fatto confermato.

4. Usa REPORTED quando una o più fonti
   riportano il fatto ma non c'è sufficiente
   conferma indipendente.

5. Usa UNVERIFIED quando le fonti non
   permettono di stabilire adeguatamente
   cosa sia successo.

6. Non inventare informazioni mancanti.

7. Gli ID delle fonti devono essere copiati
   esattamente.

EVENTO:

{json.dumps(
    event_data,
    ensure_ascii=False
)}

FONTI:

{json.dumps(
    sources,
    ensure_ascii=False
)}

Restituisci esclusivamente questo JSON:

{{
  "status": "CONFIRMED|REPORTED|UNVERIFIED",
  "confidence": 0,
  "confirmed_facts": [],
  "reported_claims": [],
  "contradictions": [],
  "important_uncertainties": [],
  "independent_source_ids": [],
  "reason": ""
}}
"""

    return generate_json(
        prompt,
        max_tokens=1400
    )


# ============================================================
# COSTRUZIONE DEL BRIEFING
# ============================================================

def build_briefing(
    events: list[dict[str, Any]],
    interests: list[str]
) -> dict[str, Any]:

    evidence = []

    # Limitiamo il numero di eventi inviati
    # a DeepSeek per evitare prompt inutilmente grandi.
    for event in events[:6]:

        sources = _compact_sources(

            event.get(
                "articles",
                []
            ),

            max_sources=2
        )

        verification = event.get(
            "verification",
            {}
        )

        evidence.append({

            "event_id": event.get(
                "id"
            ),

            "title": str(
                event.get(
                    "title",
                    ""
                )
            )[:300],

            "category": event.get(
                "category",
                ""
            ),

            "verification": {

                "status": verification.get(
                    "status"
                ),

                "confidence": verification.get(
                    "confidence"
                ),

                "reason": str(
                    verification.get(
                        "reason",
                        ""
                    )
                )[:300],

            },

            "sources": sources,

        })

    prompt = f"""
SEI VERA.

Crea una rassegna stampa giornalistica
in italiano.

Usa ESCLUSIVAMENTE gli eventi e le fonti
fornite.

NON usare conoscenze esterne.

INTERESSI DELL'UTENTE:

{json.dumps(
    interests or ["tutti"],
    ensure_ascii=False
)}

REGOLE EDITORIALI:

- Non inventare informazioni.
- Non aggiungere fatti non presenti
  nelle fonti.
- Non trasformare dichiarazioni
  in fatti.
- Mantieni le attribuzioni quando
  una notizia è soltanto riferita.
- Se un evento è REPORTED, deve risultare
  chiaramente nel testo.
- Non utilizzare formule artificiali.
- Scrivi in italiano giornalistico naturale.
- Sii sintetico.
- Dai priorità agli eventi più importanti.

Per ogni evento indica:

1. cosa è successo;
2. perché è importante;
3. cosa cambia;
4. eventuali incertezze.

EVENTI:

{json.dumps(
    evidence,
    ensure_ascii=False
)}

Restituisci esclusivamente questo JSON:

{{
  "headline": "",
  "intro": "",
  "items": [
    {{
      "event_id": "",
      "title": "",
      "what_happened": "",
      "why_it_matters": "",
      "what_changed": "",
      "uncertainty": "",
      "status": "CONFIRMED|REPORTED|UNVERIFIED",
      "sources": []
    }}
  ]
}}
"""

    return generate_json(
        prompt,
        max_tokens=1800
    )
