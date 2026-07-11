"""Book generation against a local OpenAI-compatible LLM (LM Studio, etc.).

The book is written in Castilian Spanish in two phases, each a separate set
of API calls, so the structure is easy to extend:

    1. Outline  — one call that plans the chapters. Each chapter is built
       around ONE real fact drawn from the user's Takeout data.
    2. Writing  — one streaming call per chapter that writes it extensively,
       fed the tail of what's been written so far for continuity.

The public entry point is `iter_book_events`, an orchestrator that yields
structured events (`status` / `content`) so callers (the Flask SSE endpoint,
the CLI) can show progress before any prose exists — the outline call takes a
while, and we don't want the browser to look frozen meanwhile.

Low-level helpers (`chat`, `stream_chat`) each take a message list, so you can
compose new phases (a per-chapter summary pass, a rewrite pass, …) by writing
another prompt builder and reusing them.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = Path(os.environ.get('OUTPUT_FOLDER', BASE_DIR / 'outputs'))

# Where the OpenAI-compatible LLM backend (LM Studio, etc.) lives. On a
# server this usually isn't localhost, so make it configurable.
LM_STUDIO_BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', 'http://192.168.100.138:1234')
MODEL_NAME = os.environ.get('LM_STUDIO_MODEL', 'deepseek/deepseek-r1-0528-qwen3-8b')

# (connect, read): fail fast if the backend is unreachable, but allow unlimited
# time between streamed tokens for slow (reasoning) models.
REQUEST_TIMEOUT = (10, None)

# How much of the already-written book to feed back into each chapter call for
# continuity. Keep it modest so we don't blow the context window.
CONTINUITY_CHARS = 1200

# Bounds on the outline so generation time stays predictable.
MIN_CHAPTERS = 4
MAX_CHAPTERS = 8

SYSTEM_PROMPT = (
    "Eres un novelista que escribe en español de España (castellano), con un "
    "estilo cercano al de las novelas de aventuras juveniles tipo \"Harry "
    "Potter\". Escribes de forma narrativa, cálida y con imaginación."
)


# ---------------------------------------------------------------------------
# Low-level LLM access. Two primitives, both taking a chat `messages` list:
#   chat()        -> full response text (for structured/outline calls)
#   stream_chat() -> yields text deltas  (for live prose)
# Both strip <think>…</think> reasoning so it never reaches the book.
# ---------------------------------------------------------------------------

def _endpoint() -> str:
    return f"{LM_STUDIO_BASE_URL}/v1/chat/completions"


def _post(messages: list[dict], *, stream: bool, temperature: float):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }
    return requests.post(
        _endpoint(), json=payload, stream=stream, timeout=REQUEST_TIMEOUT
    )


def chat(messages: list[dict], *, temperature: float = 0.4) -> str:
    """One non-streaming completion; returns the cleaned text response."""
    resp = _post(messages, stream=False, temperature=temperature)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return _strip_think(content).strip()


def stream_chat(messages: list[dict], *, temperature: float = 0.85):
    """Yield successive text deltas from a streaming completion."""
    with _post(messages, stream=True, temperature=temperature) as resp:
        resp.raise_for_status()
        yield from _filter_think(_iter_sse_deltas(resp))


def _iter_sse_deltas(resp):
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode('utf-8')
        if not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        chunk = json.loads(data_str)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            yield content


# --- <think> filtering -----------------------------------------------------
# Reasoning models (deepseek-r1, …) wrap their chain-of-thought in
# <think>…</think>. We must drop it both from whole responses and, trickier,
# from a token stream where the tags can be split across chunks.

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_OPEN, _CLOSE = '<think>', '</think>'


def _strip_think(text: str) -> str:
    # Also drop a dangling "<think> …" with no close (truncated reasoning).
    text = _THINK_RE.sub('', text)
    open_idx = text.lower().find(_OPEN)
    if open_idx != -1 and _CLOSE not in text.lower()[open_idx:]:
        text = text[:open_idx]
    return text


def _tag_overlap(buffer: str, tag: str) -> int:
    """Length of the longest suffix of `buffer` that is a prefix of `tag`.

    Lets us hold back a few chars that might be the start of a split tag
    instead of emitting them into the book prematurely.
    """
    for k in range(min(len(buffer), len(tag) - 1), 0, -1):
        if buffer.endswith(tag[:k]):
            return k
    return 0


def _filter_think(chunks):
    """Stream-filter, removing everything inside <think>…</think> spans."""
    buffer = ''
    in_think = False
    for chunk in chunks:
        buffer += chunk
        emitted = []
        progressed = True
        while progressed:
            progressed = False
            if in_think:
                idx = buffer.lower().find(_CLOSE)
                if idx != -1:
                    buffer = buffer[idx + len(_CLOSE):]
                    in_think = False
                    progressed = True
                else:
                    keep = _tag_overlap(buffer.lower(), _CLOSE)
                    buffer = buffer[len(buffer) - keep:] if keep else ''
            else:
                idx = buffer.lower().find(_OPEN)
                if idx != -1:
                    emitted.append(buffer[:idx])
                    buffer = buffer[idx + len(_OPEN):]
                    in_think = True
                    progressed = True
                else:
                    keep = _tag_overlap(buffer.lower(), _OPEN)
                    safe = buffer[:len(buffer) - keep] if keep else buffer
                    if safe:
                        emitted.append(safe)
                    buffer = buffer[len(buffer) - keep:] if keep else ''
        text = ''.join(emitted)
        if text:
            yield text
    if not in_think and buffer:
        yield buffer


# ---------------------------------------------------------------------------
# Turning Takeout data into a prompt-ready description.
# ---------------------------------------------------------------------------

def build_user_description(data: dict) -> str:
    """Serialize profile + activity items into plain text for the prompt."""
    lines = []

    profile = data.get('profile', {})
    for key, value in profile.items():
        lines.append(f"{key}: {value}")

    services = data.get('services', {})
    for service_name, info in services.items():
        items = info.get('items', [])
        if not items:
            continue
        lines.append(f"\n{service_name}:")
        for item in items:
            lines.append(f"- {json.dumps(item, ensure_ascii=False)}")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Phase 1 — outline. One call that returns chapters, each around a single fact.
# ---------------------------------------------------------------------------

def _outline_messages(user_description: str, user_name: str) -> list[dict]:
    user = (
        f"A partir de la siguiente información real sobre una persona llamada "
        f"{user_name}, planifica los capítulos de un libro de aventuras al "
        f"estilo de \"Harry Potter\" en el que {user_name} es el protagonista.\n\n"
        f"Cada capítulo debe girar en torno a UN único hecho o dato importante "
        f"y real extraído de la información (un interés, un lugar, una "
        f"búsqueda, una afición…). No inventes datos que contradigan la "
        f"información.\n\n"
        f"Devuelve EXCLUSIVAMENTE un array JSON válido, sin texto adicional ni "
        f"explicaciones, con esta forma exacta:\n"
        f'[\n'
        f'  {{"titulo": "Episodio 1: ...", "hecho": "un hecho concreto sobre {user_name}"}},\n'
        f'  ...\n'
        f']\n\n'
        f"Genera entre {MIN_CHAPTERS} y {MAX_CHAPTERS} capítulos. Los títulos "
        f"deben empezar por \"Episodio N:\".\n\n"
        f"Información del usuario:\n{user_description}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json_array(text: str):
    """Best-effort parse of a JSON array possibly wrapped in stray prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def generate_outline(user_description: str, user_name: str) -> list[dict]:
    """Return a list of {'title', 'fact'} chapters, or [] if the model's
    response can't be parsed (the orchestrator then falls back to free-form)."""
    raw = chat(_outline_messages(user_description, user_name), temperature=0.4)
    parsed = _extract_json_array(raw)
    if not isinstance(parsed, list):
        return []

    chapters = []
    for i, entry in enumerate(parsed, start=1):
        if not isinstance(entry, dict):
            continue
        title = str(entry.get('titulo') or entry.get('title') or '').strip()
        fact = str(entry.get('hecho') or entry.get('fact') or '').strip()
        if not title:
            title = f'Episodio {i}'
        chapters.append({'title': title, 'fact': fact})
        if len(chapters) >= MAX_CHAPTERS:
            break
    return chapters


# ---------------------------------------------------------------------------
# Phase 2 — write one chapter. Streams prose; the heading is emitted by us so
# the TOC structure is guaranteed regardless of what the model does.
# ---------------------------------------------------------------------------

def _chapter_messages(chapter: dict, index: int, total: int,
                      user_name: str, written_so_far: str) -> list[dict]:
    context = ''
    if written_so_far:
        tail = written_so_far[-CONTINUITY_CHARS:]
        context = (
            f"\n\nPara mantener la continuidad, esto es lo último que se ha "
            f"escrito del libro (no lo repitas):\n\"\"\"\n{tail}\n\"\"\""
        )
    user = (
        f"Estás escribiendo el libro \"Harry Potter y {user_name}\", donde "
        f"{user_name} es el protagonista. Es el capítulo {index} de {total}.\n\n"
        f"Título del capítulo: {chapter['title']}\n"
        f"Hecho central (real, del usuario) que debe vertebrar el capítulo: "
        f"{chapter.get('fact') or '(usa la información general del usuario)'}\n"
        f"{context}\n\n"
        f"Escribe ÚNICAMENTE este capítulo, en español de España, de forma "
        f"extensa y narrativa, integrando el hecho central en la trama. No "
        f"escribas el título (ya está puesto) ni los capítulos siguientes. "
        f"Empieza directamente con la narración."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def iter_chapter(chapter: dict, index: int, total: int,
                 user_name: str, written_so_far: str):
    """Yield the chapter as text: a markdown heading, then streamed prose."""
    yield f"## {chapter['title']}\n\n"
    messages = _chapter_messages(chapter, index, total, user_name, written_so_far)
    yield from stream_chat(messages, temperature=0.85)


# ---------------------------------------------------------------------------
# Orchestrator. Yields events so callers can show progress:
#   {'type': 'status',  'phase': 'outline'|'writing', 'label': str, ...}
#   {'type': 'content', 'text': str}
# ---------------------------------------------------------------------------

def iter_book_events(user_description: str, user_name: str):
    yield {
        'type': 'status', 'phase': 'outline',
        'label': 'Planificando los capítulos de tu libro…',
    }

    chapters = generate_outline(user_description, user_name)
    if not chapters:
        # Outline unusable — fall back to a single free-form pass so the user
        # still gets a book.
        yield from _iter_freeform_events(user_description, user_name)
        return

    total = len(chapters)
    yield {
        'type': 'status', 'phase': 'writing', 'chapter': 0, 'total': total,
        'label': f'Empezando a escribir ({total} capítulos)…',
    }

    written = ''
    for i, chapter in enumerate(chapters, start=1):
        yield {
            'type': 'status', 'phase': 'writing', 'chapter': i, 'total': total,
            'title': chapter['title'],
            'label': f'Escribiendo capítulo {i} de {total}: {chapter["title"]}',
        }
        chapter_text = ''
        for delta in iter_chapter(chapter, i, total, user_name, written):
            chapter_text += delta
            yield {'type': 'content', 'text': delta}
        # Guarantee separation between chapters.
        if not chapter_text.endswith('\n'):
            yield {'type': 'content', 'text': '\n\n'}
            chapter_text += '\n\n'
        written += chapter_text


def _iter_freeform_events(user_description: str, user_name: str):
    """Legacy single-call path, used when the outline can't be parsed."""
    yield {
        'type': 'status', 'phase': 'writing', 'chapter': 0, 'total': 0,
        'label': 'Escribiendo tu libro…',
    }
    user = (
        f"Con la información del usuario:\n\n{user_description}\n\n"
        f"Escribe un libro en español de España llamado \"Harry Potter y "
        f"{user_name}\". Empieza cada capítulo en su propia línea con un "
        f"encabezado markdown, por ejemplo \"## Episodio 1: Título\", para que "
        f"la estructura sea fácil de seguir."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    for delta in stream_chat(messages, temperature=0.85):
        yield {'type': 'content', 'text': delta}


# ---------------------------------------------------------------------------
# Convenience wrappers for callers that only want the text (CLI, tests).
# ---------------------------------------------------------------------------

def iter_book_chunks(user_description: str, user_name: str):
    """Yield only the book text, discarding status events."""
    for event in iter_book_events(user_description, user_name):
        if event['type'] == 'content':
            yield event['text']


def stream_book(user_description: str, user_name: str) -> None:
    for event in iter_book_events(user_description, user_name):
        if event['type'] == 'status':
            print(f"\n[{event['label']}]\n", file=sys.stderr, flush=True)
        else:
            print(event['text'], end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stream a book draft from a local LM Studio model using a processed Takeout session."
    )
    parser.add_argument(
        "session_id",
        help="Session ID of the processed Takeout output (outputs/<session_id>.json)",
    )
    args = parser.parse_args()

    output_path = OUTPUT_FOLDER / f"{args.session_id}.json"
    if not output_path.exists():
        print(f"No output found for session {args.session_id} at {output_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(output_path.read_text(encoding='utf-8'))
    user_name = data.get('profile', {}).get('name') or 'You'
    user_description = build_user_description(data)

    print(f"Generating \"Harry Potter y {user_name}\" with model {MODEL_NAME}...\n")
    stream_book(user_description, user_name)


if __name__ == '__main__':
    main()
