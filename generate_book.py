"""Book generation against a local OpenAI-compatible LLM (LM Studio, etc.).

The book is written in Castilian Spanish in two phases, which is what the site
shows the reader:

    1. Preparation — a chain of planning calls: profile → synopsis → outline →
       one spec per chapter → cover-art prompt → cover art. None of it produces
       readable prose, so it reports itself as numbered steps (PREP_TOTAL of
       them) rather than leaving the page looking frozen.
    2. Writing — one streaming call per chapter, fed the previous chapter for
       continuity. This is the part the reader watches arrive.

The entry points (`iter_book_events`, `iter_book_chunks`, `stream_book`) all
take a single `source`: the path to a processed-Takeout JSON file, or an
already-loaded data dict. So the whole pipeline is drivable from a file:

    import generate_book
    generate_book.stream_book("outputs/<session_id>.json")      # print it
    text = "".join(generate_book.iter_book_chunks("out.json"))  # collect it

`iter_book_events` yields structured events (`status` / `content`) so callers
(the Flask SSE endpoint, the CLI) can show progress before any prose exists.

Each phase of the chain is a function taking explicit arguments and returning
text, with its prompt right above it — so a phase can be run, inspected or
re-prompted on its own:

    md = generate_book.build_user_description(generate_book.load_session(path))
    perfil = generate_book.extract_profile(md)
    sinopsis = generate_book.generate_synopsis(perfil)

Two models do the work, because they're good at different things: a small
instruct model plans (PLANNING_MODEL) and an uncensored writer model writes
(WRITING_MODEL). Both are overridable by env var, as is either backend URL.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = Path(os.environ.get('OUTPUT_FOLDER', BASE_DIR / 'outputs'))

# Where the OpenAI-compatible LLM backend (LM Studio, etc.) lives. On a
# server this usually isn't localhost, so make it configurable.
LM_STUDIO_BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', 'http://192.168.100.138:1234')
# The Stable Diffusion server that draws the cover art — a different box/port.
SD_BASE_URL = os.environ.get('SD_BASE_URL', 'http://192.168.100.138:8000')

# Planning (profile, synopsis, outline, art prompt) wants an instruct model that
# follows a spec; the prose wants a writer model that won't refuse the brief.
# An unknown model id is NOT an error in LM Studio — it quietly answers with
# whatever is loaded — so a typo here degrades the book silently.
PLANNING_MODEL = os.environ.get('PLANNING_MODEL', 'ibm/granite-4-h-tiny')
WRITING_MODEL = os.environ.get(
    'WRITING_MODEL', 'dirty-muse-writer-v01-uncensored-erotica-nsfw-i1')

# Sampler knobs for the prose only — the planning calls want a model that
# repeats the spec back faithfully, so penalising repetition there would work
# against them.
#
# `frequency_penalty` rather than `presence_penalty`: LM Studio's engine accepts
# presence_penalty and silently ignores it. Measured against this box, sweeping
# it from -2.0 to +2.0 gives byte-identical output, while the two knobs below
# each change it — and an unknown param is never an error here, so a setting
# that does nothing looks exactly like one that works. Re-measure before
# swapping either of these for a knob that reads better on paper.
WRITING_PARAMS = {
    'repeat_penalty': 1.8,
    'frequency_penalty': 1.2,
}

# (connect, read): fail fast if the backend is unreachable, but allow unlimited
# time between streamed tokens for slow (reasoning) models.
REQUEST_TIMEOUT = (10, None)
# The health check blocks a page, so it gets a real read timeout: a backend too
# slow to list its models in 5s is not one we can write a book with.
ENGINE_CHECK_TIMEOUT = (3, 5)
# Same for diffusion: one silent wait ending in the whole PNG, with no tokens
# trickling back to prove it's alive.
SD_TIMEOUT = (10, None)

NUM_CHAPTERS = 10

# Preparation steps, in order: profile, synopsis, outline, one per chapter spec,
# art prompt, art. Derived rather than written down, so it stays true if the
# chain grows a phase.
PREP_TOTAL = 3 + NUM_CHAPTERS + 2

# How much activity data to put in the prompt. A processed session can hold
# hundreds of items per service, which would blow past a local model's context
# window — a representative sample is enough to mine facts for the book. These
# defaults keep the outline prompt near ~2.7k tokens, leaving room in an 8k
# context for a reasoning model to think *and* answer. Raise them if your model
# has a larger context; lower them if it's smaller.
PROMPT_ITEMS_PER_SERVICE = 20
PROMPT_ITEM_CHARS = 120


# ---------------------------------------------------------------------------
# Engine availability. The backend is a box on the LAN running LM Studio, so
# "it's not there right now" is a normal condition, not an exceptional one:
# it gets its own exception type, kept apart from bugs in our own prompts.
# ---------------------------------------------------------------------------

class EngineUnavailable(RuntimeError):
    """The backend couldn't serve the request: unreachable, or its model died.

    Explicitly NOT for a request the backend understood and rejected — a prompt
    over the context window is our bug, and telling the reader "the engine is
    down" would send them to retry something that can never work.
    """


# LM Studio reports a model that died or was unloaded mid-request as
# 400 {"error":"terminated"} — a backend failure wearing a client-error code, so
# the status alone can't classify it. Same for a request naming a model that
# isn't loaded, and for
#   400 {"error":"Engine protocol predict request failed: fetch failed"}
# which is LM Studio's own server failing to reach the inference process it
# fronts: its engine died, and "fetch failed" is it saying so from the outside.
# Anything else 4xx is ours to fix, not the engine's — a prompt over the context
# window lands here too, and must not be mistaken for a backend that's down.
_ENGINE_DEAD_RE = re.compile(
    r'terminated|exit|crash|no model|model.{0,30}(not|un)load|not found'
    r'|engine protocol|fetch failed',
    re.IGNORECASE,
)

# That engine crash is transient in a way the others aren't: it drops the one
# request it's on and serves the next one normally (measured: back-to-back
# planning calls die around the 9th, and an immediate retry succeeds). So it's
# worth retrying rather than reporting — the alternative is throwing away a
# ten-minute book over a hiccup the box recovers from before we can ask again.
ENGINE_RETRIES = 3
# Multiplied by the attempt number: 2s, then 4s. The engine is ready again well
# inside the first wait; the backoff is for the case where it isn't.
ENGINE_RETRY_BACKOFF = 2.0


def _endpoint() -> str:
    return f"{LM_STUDIO_BASE_URL}/v1/chat/completions"


def _post(messages: list[dict], *, stream: bool, temperature, model: str,
          params: dict = None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    # Omitted rather than defaulted: each model's own sampler setting is usually
    # what it was tuned with, and the writer model in particular is picked for
    # its voice.
    if temperature is not None:
        payload["temperature"] = temperature
    # Extra sampler knobs, straight through to the API (see WRITING_PARAMS).
    if params:
        payload.update(params)
    try:
        return requests.post(
            _endpoint(), json=payload, stream=stream, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        # Refused, DNS-less, or timed out on connect: nothing was served.
        raise EngineUnavailable(f'sin respuesta de {_endpoint()}: {e}') from e


def _raise_for_status(resp) -> None:
    """Like resp.raise_for_status() but include the backend's error body, which
    is where LM Studio explains a 400 (e.g. the prompt exceeds the context)."""
    if resp.status_code < 400:
        return
    body = resp.text[:500].strip()
    detail = f"{resp.status_code} from {resp.url}: {body}"
    if resp.status_code >= 500 or _ENGINE_DEAD_RE.search(body):
        raise EngineUnavailable(detail)
    raise requests.HTTPError(detail, response=resp)


def engine_status() -> dict:
    """Whether the LLM backend can write a book right now, and if not, why.

    Checks the models are actually loaded, not just that the port answers: an
    unknown model id is not an error to LM Studio, it just replies with whatever
    else is loaded. So "up but the writer model was ejected" looks like success
    at the HTTP level while quietly producing a book in the wrong voice.
    """
    try:
        resp = requests.get(f'{LM_STUDIO_BASE_URL}/v1/models',
                            timeout=ENGINE_CHECK_TIMEOUT)
        resp.raise_for_status()
        loaded = {m['id'] for m in resp.json().get('data', [])}
    except Exception as e:
        return {'available': False, 'reason': 'unreachable', 'detail': str(e),
                'missing_models': []}

    missing = [m for m in (PLANNING_MODEL, WRITING_MODEL) if m not in loaded]
    if missing:
        return {'available': False, 'reason': 'models_missing',
                'detail': f'modelos no cargados: {", ".join(missing)}',
                'missing_models': missing}
    return {'available': True, 'reason': None, 'detail': None,
            'missing_models': []}


def _post_checked(messages, *, stream, temperature, model, params=None):
    """POST a completion and validate the reply, retrying engine failures.

    Retrying is only safe here, and that's why this stops where it does: it
    returns before a single token has been read, so both failures it swallows
    (nothing served, or a crashed engine answering 400) happened with nothing
    yet on the reader's page. A stream that dies *later* is not retried — those
    tokens are already written, and re-running the prompt would repeat them in
    the middle of a chapter.
    """
    for attempt in range(1, ENGINE_RETRIES + 1):
        resp = None
        try:
            resp = _post(messages, stream=stream, temperature=temperature,
                         model=model, params=params)
            _raise_for_status(resp)
            return resp
        except EngineUnavailable as e:
            if resp is not None:
                resp.close()
            if attempt == ENGINE_RETRIES:
                raise
            # Logged, not silent: a box that needs retries on every call is
            # failing in a way someone should see, even though the book survives.
            print(f'engine hiccup ({attempt}/{ENGINE_RETRIES}), reintentando: {e}',
                  file=sys.stderr, flush=True)
            time.sleep(ENGINE_RETRY_BACKOFF * attempt)


def chat(messages, *, temperature=None, model: str = None) -> str:
    """One non-streaming completion; returns the cleaned text response.

    `messages` is a chat messages list, or a bare string for the common
    single-user-turn case.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    resp = _post_checked(messages, stream=False, temperature=temperature,
                         model=model or PLANNING_MODEL)
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return _strip_think(content).strip()


def stream_chat(messages, *, temperature=None, model: str = None, params=None):
    """Yield successive text deltas from a streaming completion.

    This is the prose path — the only streaming caller is iter_chapter — so it
    carries WRITING_PARAMS by default. Pass `params` to override; pass `{}` for
    a bare call with no sampler knobs at all.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    with _post_checked(messages, stream=True, temperature=temperature,
                       model=model or WRITING_MODEL,
                       params=WRITING_PARAMS if params is None else params) as resp:
        yield from _filter_think(_iter_sse_deltas(resp))


def _iter_sse_deltas(resp):
    try:
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
            # A model that dies mid-stream reports it in the stream — the POST
            # already returned 200, so there's no status left to check.
            if chunk.get("error"):
                raise EngineUnavailable(f'{_endpoint()} interrumpió la '
                                        f'respuesta: {chunk["error"]}')
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                yield content
    except requests.RequestException as e:
        # The connection dropped part-way through the prose.
        raise EngineUnavailable(f'se perdió la conexión con {_endpoint()}: {e}') from e


def generate_image(prompt: str, dest_path: Path, **params) -> Path:
    """Draw `prompt` on the Stable Diffusion server; returns the saved path.

    Unknown kwargs go straight through to the API: negative_prompt, steps,
    guidance_scale, width, height, seed. It answers with raw PNG bytes, so an
    error body is the only thing worth decoding.
    """
    try:
        resp = requests.post(f'{SD_BASE_URL}/generate',
                             json={'prompt': prompt, **params}, timeout=SD_TIMEOUT)
    except requests.RequestException as e:
        raise EngineUnavailable(f'sin respuesta de {SD_BASE_URL}: {e}') from e
    _raise_for_status(resp)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)
    return dest_path


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
# Loading a processed-Takeout session and turning it into a prompt.
# ---------------------------------------------------------------------------

def load_session(source) -> dict:
    """Return the processed-Takeout data dict.

    `source` is either a path (str/Path) to an ``outputs/<session_id>.json``
    file, or an already-loaded dict (returned unchanged, so callers can filter
    services before generating).
    """
    if isinstance(source, (str, Path)):
        return json.loads(Path(source).read_text(encoding='utf-8'))
    return source


def get_user_name(data: dict) -> str:
    """The protagonist's name from the profile, or a neutral default."""
    return data.get('profile', {}).get('name') or 'You'


def build_user_description(data: dict, *,
                           items_per_service: int = PROMPT_ITEMS_PER_SERVICE,
                           item_chars: int = PROMPT_ITEM_CHARS) -> str:
    """Serialize profile + a bounded sample of activity as markdown.

    Only the first `items_per_service` items of each service are included, each
    truncated to `item_chars`, so the prompt stays within the model's context
    window no matter how much data the session holds.

    Items are emitted as their title text alone, under a markdown heading per
    service. Once the Takeout processing has dropped urls and timestamps a title
    is all an item holds, so wrapping each one back up as JSON cost ~5 tokens of
    syntax — a fifth of a truncated item — and told the model nothing.
    """
    lines = []

    profile = data.get('profile', {})
    for key, value in profile.items():
        lines.append(f"{key}: {value}")

    services = data.get('services', {})
    for service_name, info in services.items():
        items = info.get('items', [])
        if not items:
            continue
        lines.append(f"\n## {service_name}")
        for item in items[:items_per_service]:
            title = (item.get('title') or '').strip()
            if not title:
                continue
            if len(title) > item_chars:
                title = title[:item_chars] + '…'
            lines.append(f"- {title}")

    return '\n'.join(lines)


def art_path(session_id: str) -> Path:
    """Where a session's cover art lands. book_builder composites it under the
    cover template; app.py looks here when building the finished cover."""
    return OUTPUT_FOLDER / f'{session_id}_art.png'


# ---------------------------------------------------------------------------
# Preparation, phase 1 — mine the activity dump for narrative material.
# ---------------------------------------------------------------------------

PROFILE_PROMPT = """

The contents of the markdown file are part of the context of this conversation. Use them as your primary source of information throughout the task. Do not ask for the file again. If any information is missing from the markdown, continue using only the available data. Never invent, hallucinate, or infer unsupported facts.

You are a data extraction and narrative research assistant.

# Task

Read the provided markdown file containing Google activity from one user. Each "## " heading is a Google service (Books, Flights, Maps, Search, Shopping, Video Search, YouTube, Chrome, Gemini and others) and each "- " bullet under it is one activity entry.

Extract only information useful for building a fictional character and world. The goal is not to summarize the user's life, but to preserve concrete narrative references that can later be reused in a novel.

# Processing instructions

1. Inspect the entire markdown, including nested gemini and ai conversations, search queries, google map locations, books, videos search, youtube historial any textual field.
2. Focus on AI Mode and Gemini to extract relevant information about the
2. Preserve exact names whenever possible.
3. Prefer concrete references over abstract summaries.
4. Write different data in the different categories, do not repeat.
6. Output only the final information.

# Evidence rule

Every interpretation must be immediately supported by evidence extracted from the markdown.

Use this format:
- **Observation:** interpretative narrative observation.
- **Evidence:** exact searches, conversations, titles, places, books, videos, products or other entries found in the markdown.
# Required output

# 1. User Profile
Include only explicit information when available:
- Name
- Approximate age (only if explicit)
- Gender (only if explicit)
- Occupation or studies
- Fears, Desires and Goals
- Character and personality
Then write a short narrative description based on the data.
---
# 2. People
List all named people found in the data.
Include:
- Family members
- Friends
- Authors
- Artists
- Public figures
- Any other named individual
---
# 3. Places
List every concrete location with narrative value.
Examples include:
- Countries
- Cities
- Neighbourhoods
- Streets
- Hotels
- Restaurants
- Cafés
- Museums
- Shops
- Parks
- Mountains
- Beaches
- Any specifically named place
---
# 5. Interests and Hobbies
Extract specific interests.
Avoid generic labels such as "technology" or "travel".
Prefer concrete topics.
---
# 6. Cultural References
List concrete references including:
- Books
- Authors
- Films
- TV series
- Music
Preserve exact titles whenever possible.
---
# 7. Activities and Experiences
Extract activities actually performed, planned or repeatedly searched.
Examples:
- Trips
- Shopping
- Reading
- Cooking
- Hiking
- Gaming
- Sporting activities
- Museum visits
Focus on concrete activities rather than broad lifestyle descriptions.
---
# 8. Conversations, Searches and Ideas
This is very importante and priority.
Summarise conversations with Gemini and IA Mode together with important searches.
Identify and extract
- Fears
- Desires
- Personal projects
- Ideas
Group related entries into themes.
---
# 9. Narrative Threads
Select the unique and reusable references in the entire dataset.
These details should make the fictional character feel unique rather than generic.
# Final quality checklist
Before producing the answer, internally verify that:
- The compelte JSON section was inspected.
- AI Mode and Gemini received the greatest attention.
- Proper names were preserved.
- Concrete references were prioritised over summaries.
- Technical metadata was omitted.
- All sections were completed whenever data existed.
Return only the final information.


"""


def extract_profile(user_description: str) -> str:
    """Turn the raw activity markdown into a narrative profile of the person.

    The description goes in as its own user message rather than pasted into the
    prompt — that's what "attaching the markdown" means against an API with no
    file upload, and it keeps the instructions and the data apart.
    """
    return chat([
        {"role": "user", "content": PROFILE_PROMPT},
        {"role": "user", "content": user_description},
    ])


# ---------------------------------------------------------------------------
# Preparation, phase 2 — the story: synopsis, then a 10-chapter outline, then
# one spec per chapter pulled back out of it.
# ---------------------------------------------------------------------------

def _synopsis_prompt(profile: str) -> str:
    return f"""

		TEXT:

		{profile}

		Based on the information provided in this previous text, write a synopsis for a personalized Harry Potter novel. Use the information as the only source for the protagonist's profile. Do not invent new biographical details beyond what is explicitly supported.

	Requirements:

	- The protagonist is the person described in the provided information.
	- Create a title with the formula "Harry Potter and [protagonist's name]".
	- Harry Potter meets the protagonist early in the story.
	- At a key moment, Harry reveals that the protagonist is a witch or wizard and invites them into the wizarding world.
	- The central emotional thread of the story should be romance, attraction and unresolved romantic tension with several carachters
	- The protagonist should develop meaningful romantic chemistry with several established Harry Potter characters, including both male and female characters. The relationships may involve mutual attraction, jealousy, longing, rivalry, emotional intimacy, misunderstandings and slow-burn romance.
	- The tone of these relationships should be inspired by the emotional style of popular Harry Potter fanfiction, with bisexual romantic tension, complicated feelings and evolving relationships, while remaining compatible with the tone of the original series.
	- Build the main external conflict around one important narrative thread, interest, activity or goal extracted from the protagonist's profile.
	- Naturally incorporate many concrete references from the profile (people, places, objects, books, hobbies, trips, videos, projects, etc.).
	- Include well-known Harry Potter characters such as Dumbledore, Hagrid, McGonagall and others where appropriate.
	- Write only a synopsis (200–300 words), not the full novel.
	"""


def generate_synopsis(profile: str) -> str:
    """A 200–300 word pitch for the novel, built only from the profile."""
    return chat(_synopsis_prompt(profile))


def _outline_prompt(text: str) -> str:
    return f"""
	TEXT:

		{text}

	The information provided in the text consists of:

	1. A narrative profile of the protagonist extracted from real Google activity.
	2. A story synopsis for a personalized Harry Potter novel.

	Use both as the only source of information.

	Your task is to expand the synopsis into a detailed outline of **{NUM_CHAPTERS} chapters**.

	Requirements:

	- Follow and develop the synopsis faithfully.
	- Each chapter must contain:
	  - A chapter title.
	  - A 200 word summary describing the events.
	- Narrative purposeof the chapter (one short sentence explaining why this chapter exists in the overall story, e.g. introducing a character, raising the stakes, revealing new information, strengthening relationships, creating a setback, preparing the climax, resolving a subplot, etc.)
	 - The Harry Potter characters involved.
	- The profile references incorporated.

	- Every chapter should naturally incorporate **3–5 concrete references** from the protagonist's profile.
	- Across the entire outline, ensure that references from all profile categories (People, Places, Objects, Interests, Cultural references, Activities, Conversations and ideas) are represented.
	- Introduce new references throughout the novel so that most of the protagonist's profile is eventually incorporated.
	- Every reference listed must appear naturally in the chapter summary and contribute to the plot, dialogue, setting, relationships or conflict. Do not list references that are not actually used in the chapter.
	- Prioritize specific references over generic concepts.
	- Spread the references across the entire outline. Introduce new references in every chapter and avoid repeatedly relying on the same small set of references.

	- Romantic relationships should be one of the main driving forces of the narrative. Draw inspiration from the emotional style of popular Harry Potter fanfiction, emphasizing slow-burn attraction, unresolved romantic tension, jealousy, longing, emotional vulnerability, misunderstandings, rivalries, secret affections and evolving relationships.
	- The protagonist should develop meaningful romantic chemistry with several main Harry Potter characters of different genders, creating overlapping attractions, uncertainty, emotional conflicts and shifting loyalties. The overall tone should naturally include both heterosexual and homoerotic romantic tension.
	- Ensure that the romantic plot evolves throughout the ten chapters, with feelings gradually intensifying and influencing the main adventure instead of remaining a secondary subplot.
	- Include private intimate moments that deepen relationships and reveal character development.

	- Build a clear narrative progression, escalating conflict and a satisfying character arc across the ten chapters.

	Return only the outline.

	"""


def generate_outline(text: str) -> str:
    """The whole book's chapter outline, as one blob of text."""
    return chat(_outline_prompt(text))


def _chapter_outline_prompt(outline: str, index: int) -> str:
    return f"""
		Extract the especific text tat describes the info in the chapter #{index}, only that, no further explanation:

		Chapters:
		{outline}
	"""


def extract_chapter_outline(outline: str, index: int) -> str:
    """Pull chapter `index`'s spec back out of the outline blob (1-based).

    A model call rather than a regex on purpose: the outline's shape is whatever
    the model felt like emitting that run, so there's no heading format to split
    on reliably.
    """
    return chat(_chapter_outline_prompt(outline, index))


# ---------------------------------------------------------------------------
# Preparation, phase 3 — the cover art.
# ---------------------------------------------------------------------------

def _art_prompt(profile: str) -> str:
    return f"""
	PROFILE:

	{profile}

	Pick the SINGLE most distinctive object or place from this profile — one
	only, the one that best identifies this person.

	Write a Stable Diffusion prompt for a book cover illustration of it, in the
	style of a magical Harry Potter book cover.

	Rules:
	- One line, in English, under 40 words.
	- Describe only that one object or place. No people, no faces, no text.
	- Return only the prompt, no explanation, no quotes.
	"""

ART_NEGATIVE_PROMPT = 'text, watermark, signature, people, faces, blurry'


def generate_art_prompt(profile: str) -> str:
    """One distinctive object/place from the profile, as an image prompt.

    One subject, because a prompt listing several just averages them into mush.
    """
    return chat(_art_prompt(profile)).strip().strip('"')


def generate_cover_art(prompt: str, dest_path: Path) -> Path:
    """Draw the cover art, shaped to the window it shows through."""
    import book_builder

    width, height = book_builder.cover_art_size()
    return generate_image(prompt, dest_path,
                          negative_prompt=ART_NEGATIVE_PROMPT,
                          width=width, height=height)


# ---------------------------------------------------------------------------
# Writing — one chapter, streamed. The heading is emitted by us so the index
# structure is guaranteed regardless of what the model does.
# ---------------------------------------------------------------------------

def _chapter_prompt(chapter_outline: str, index: int, previous_chapter: str) -> str:
    prompt = f"""
		Tarea
	Eres un escritor de fanfiction y novelista. Escribe el siguiente capítulo de una novela de Harry Potter.
	IMPORTANTE: Escribe todo el texto en español (castellano). No escribas ninguna frase en inglés, salvo títulos de vídeos, libros o nombres propios cuando aparezcan como referencias.
	Instrucciones
	Escribe únicamente el capítulo, sin explicaciones.
	Estilo: novela, no resumen.
	Prosa rica, fluida e inmersiva.
	Mucho diálogo.
	Escribe escenas de romance, sexo, intimidad, emociones fuertes, violencia, pasión o conflicto.
	Inspírate en fan fictions eróticos de harry potter.

	Contenidos que deben aparecer:

	{chapter_outline}

		"""

    if previous_chapter:
        prompt += f"""

			Capítulo anterior para contexto:
			{previous_chapter}
		"""

    prompt += """
		Escribe:
		"""
    return prompt


def iter_chapter(chapter_outline: str, index: int, previous_chapter: str):
    """Yield the chapter as text: a markdown heading, then streamed prose.

    Only the previous chapter is passed for continuity, not the book so far —
    that keeps the prompt bounded no matter how long the book runs.
    """
    yield f"# Capítulo {index}:\n\n"
    yield from stream_chat(_chapter_prompt(chapter_outline, index, previous_chapter))


# ---------------------------------------------------------------------------
# Orchestrator. Yields events so callers can show progress:
#   {'type': 'status', 'phase': 'prep',    'step': int, 'total': int, 'label': str}
#   {'type': 'status', 'phase': 'writing', 'chapter': int, 'total': int, 'label': str}
#   {'type': 'content', 'text': str}
# ---------------------------------------------------------------------------

def _save_prompt(data: dict, user_description: str) -> None:
    """Mirror the description to outputs/<session_id>_prompt.md.

    Purely a debug artifact: the description is built per run and otherwise only
    ever exists inside the request to the model, so this is the one way to read
    back what a given session was actually told. Skipped for data with no
    session_id (a hand-built dict), and never read by anything.
    """
    session_id = data.get('session_id')
    if not session_id:
        return
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / f'{session_id}_prompt.md').write_text(
        user_description, encoding='utf-8')


def _save_planning(session_id: str, name: str, text: str) -> None:
    """Mirror a planning step's output to outputs/<session_id>_<name>.md.

    The chain's middle is invisible otherwise — none of it reaches the book, so
    when a book comes out wrong these files are the only way to see which step
    went wrong. Nothing reads them back.
    """
    if not session_id:
        return
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / f'{session_id}_{name}.md').write_text(text, encoding='utf-8')


def iter_book_events(source):
    """Generate the book from a JSON file path (or data dict); yield events."""
    data = load_session(source)
    session_id = data.get('session_id')
    user_description = build_user_description(data)
    _save_prompt(data, user_description)

    step = 0

    def prep(label: str):
        """The next preparation step's status event.

        Each step says what's happening now, in the reader's terms rather than
        the chain's — several minutes pass before a word of prose exists, and
        watching the story get built is more reassuring than a bar that only
        ever says "preparando". `step`/`total` drive the progress bar, so the
        label never has to carry its own count.
        """
        nonlocal step
        step += 1
        return {
            'type': 'status', 'phase': 'prep', 'step': step, 'total': PREP_TOTAL,
            'label': label,
        }

    yield prep('Leyendo tu actividad')
    perfil = extract_profile(user_description)
    _save_planning(session_id, 'perfil', perfil)

    yield prep('Preparando la trama')
    sinopsis = generate_synopsis(perfil)
    _save_planning(session_id, 'sinopsis', sinopsis)

    yield prep('Repartiendo la historia en capítulos')
    outline = generate_outline(sinopsis)
    _save_planning(session_id, 'outline', outline)

    chapter_outlines = []
    for i in range(1, NUM_CHAPTERS + 1):
        yield prep(f'Perfilando el capítulo {i} de {NUM_CHAPTERS}')
        chapter_outlines.append(extract_chapter_outline(outline, i))
    _save_planning(session_id, 'capitulos',
                   '\n\n---\n\n'.join(chapter_outlines))

    yield prep('Imaginando la portada')
    art_prompt = generate_art_prompt(perfil)
    _save_planning(session_id, 'portada', art_prompt)

    yield prep('Dibujando la portada')
    if session_id:
        try:
            generate_cover_art(art_prompt, art_path(session_id))
        except Exception as e:
            # The art is decoration; the book is the product. A drawing server
            # that's down or slow must not throw away the writing that follows —
            # the cover just falls back to its blank template.
            print(f'cover art failed, continuing without it: {e}',
                  file=sys.stderr, flush=True)

    # --- writing -----------------------------------------------------------
    total = len(chapter_outlines)
    yield {
        'type': 'status', 'phase': 'writing', 'chapter': 0, 'total': total,
        'label': f'Empezando a escribir ({total} capítulos)…',
    }

    previous_chapter = ''
    for i, chapter_outline in enumerate(chapter_outlines, start=1):
        yield {
            'type': 'status', 'phase': 'writing', 'chapter': i, 'total': total,
            'label': f'Escribiendo capítulo {i} de {total}',
        }
        chapter_text = ''
        for delta in iter_chapter(chapter_outline, i, previous_chapter):
            chapter_text += delta
            yield {'type': 'content', 'text': delta}
        # Guarantee separation between chapters.
        if not chapter_text.endswith('\n'):
            yield {'type': 'content', 'text': '\n\n'}
            chapter_text += '\n\n'
        previous_chapter = chapter_text


# ---------------------------------------------------------------------------
# Convenience wrappers for callers that only want the text (CLI, tests).
# ---------------------------------------------------------------------------

def iter_book_chunks(source):
    """Yield only the book text, discarding status events."""
    for event in iter_book_events(source):
        if event['type'] == 'content':
            yield event['text']


def stream_book(source) -> None:
    """Print the book to stdout (status labels go to stderr)."""
    for event in iter_book_events(source):
        if event['type'] == 'status':
            print(f"\n[{event['label']}]\n", file=sys.stderr, flush=True)
        else:
            print(event['text'], end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stream a book draft from a local LLM using a processed-Takeout JSON file."
    )
    parser.add_argument(
        "json_path",
        help="Path to a processed-Takeout JSON file (e.g. outputs/<session_id>.json). "
             "A bare session id is also accepted.",
    )
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.exists():
        # Convenience: allow a bare session id, resolved under OUTPUT_FOLDER.
        fallback = OUTPUT_FOLDER / f"{args.json_path}.json"
        if not fallback.exists():
            print(f"No such file: {path}", file=sys.stderr)
            sys.exit(1)
        path = fallback

    data = load_session(path)
    print(f'Generating "Harry Potter y {get_user_name(data)}" with {PLANNING_MODEL} '
          f'(planning) and {WRITING_MODEL} (writing)...\n',
          file=sys.stderr, flush=True)
    stream_book(data)


if __name__ == '__main__':
    main()
