import json
import os
from typing import Any

from openai import OpenAI


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash"
)


class AIError(Exception):
    pass


def _client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise AIError(
            "DEEPSEEK_API_KEY non configurata"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def _json_from_text(text: str) -> dict[str, Any]:
    if not text:
        raise AIError(
            "DeepSeek ha restituito una risposta vuota"
        )

    text = text.strip()

    # Gestione eventuale markdown ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
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


def generate_json(
    prompt: str,
    max_tokens: int = 1200
) -> dict[str, Any]:

    client = _client()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei il motore editoriale di VERA. "
                        "Sei un assistente giornalistico rigoroso. "
                        "Usa esclusivamente le informazioni contenute "
                        "nelle fonti fornite. "
                        "Non inventare fatti, nomi, numeri o eventi. "
                        "Distingui sempre tra fatti verificati, "
                        "dichiarazioni e informazioni riportate. "
                        "Rispondi esclusivamente con JSON valido."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.1,

            max_tokens=max_tokens,

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


def _compact_sources(
    sources: list[dict[str, Any]],
    max_sources: int = 3
) -> list[dict[str, Any]]:

    compact = []

    for article in sources[:max_sources]:

        compact.append({
            "id": article.get("id"),
            "outlet": article.get("outlet"),
            "published": article.get("published"),
            "title": str(
                article.get("title", "")
            )[:300],
            "summary": str(
                article.get("summary", "")
            )[:500],
        })

    return compact


def verify_event(
    event: dict[str, Any]
) -> dict[str, Any]:

    sources = _compact_sources(
        event.get("articles", []),
        max_sources=3
    )

    prompt = f"""
VERIFICA EDITORIALE VERA

Devi verificare un evento giornalistico.

Usa ESCLUSIVAMENTE le fonti riportate sotto.

NON usare conoscenze esterne.

REGOLE:

1. Non considerare automaticamente vera una notizia
   solo perché compare in una fonte.

2. Distingui:
   - fatti riportati dalle fonti;
   - dichiarazioni di persone o istituzioni;
   - accuse;
   - interpretazioni.

3. Usa CONFIRMED solo se le fonti forniscono
   elementi sufficienti per considerare il fatto
   adeguatamente confermato.

4. Usa REPORTED se una fonte attendibile riporta
   il fatto ma non esiste sufficiente conferma
   indipendente.

5. Usa UNVERIFIED se le fonti non permettono
   di stabilire adeguatamente cosa sia successo.

6. Non inventare informazioni mancanti.

7. Gli ID delle fonti devono essere copiati
   esattamente dalle fonti fornite.

EVENTO:

{json.dumps({
    "id": event.get("id"),
    "title": event.get("title", "")[:400],
    "category": event.get("category", "")
}, ensure_ascii=False)}

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
        max_tokens=1200
    )


def build_briefing(
    events: list[dict[str, Any]],
    interests: list[str]
) -> dict[str, Any]:

    evidence = []

    # Non mandiamo un'enorme quantità di testo a DeepSeek.
    # Il briefing finale utilizza al massimo 6 eventi.
    for event in events[:6]:

        sources = _compact_sources(
            event.get("articles", []),
            max_sources=2
        )

        verification = event.get(
            "verification",
            {}
        )

        evidence.append({
            "event_id": event.get("id"),

            "title": str(
                event.get("title", "")
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
in italiano usando ESCLUSIVAMENTE
gli eventi e le fonti fornite.

INTERESSI DELL'UTENTE:

{json.dumps(
    interests or ["tutti"],
    ensure_ascii=False
)}

REGOLE EDITORIALI:

- Non inventare informazioni.
- Non usare conoscenze esterne.
- Non aggiungere fatti che non compaiono
  nelle fonti.
- Non trasformare dichiarazioni in fatti.
- Mantieni le attribuzioni quando necessario.
- Se un evento è REPORTED, il testo deve
  far capire chiaramente che si tratta
  di una notizia riportata dalle fonti.
- Non usare formule artificiose.
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
