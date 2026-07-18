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
    data_bank = generate_book.reduce_data_bank(generate_book.mine_data_bank(md))
    outline = generate_book.generate_outline("Ada Lovelace", perfil, data_bank)

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

# Preparation steps, in order: profile, mine details, reduce details, outline,
# one per chapter spec, art prompt, art. Derived rather than written down, so it
# stays true if the chain grows a phase.
PREP_TOTAL = 4 + NUM_CHAPTERS + 2

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
# Preparation, phase 1 — read the person out of the activity dump.
#
# Three writing-model calls, on purpose: the profile and the detail-mining want
# the uncensored writer's voice, not the planning model's. profile() paints who
# they are; mine_data_bank() hunts the specific, weird, quotable details; and
# reduce_data_bank() consolidates those to a single sampleable set for the
# outline. Ported from the chain proven out in lm_dev.py's __main__.
# ---------------------------------------------------------------------------

def _profile_prompt(data_md: str) -> str:
    return f"""

You are a sarcastic, needling profiler-for-hire — think a nosy detective who's seen too much and finds people's digital footprints hilarious, not clinical. You are NOT a marketing analyst and NOT a therapist. Never sound reassuring, clinical, or neutral.

<data>
{data_md}
</data>

STEP 1 — SCAN (internal, do not output):
Skim the ENTIRE file. Note at least 3 surprising or contradictory data points that occur OUTSIDE the first third of the file. You must use at least one of these in the final profile.

STEP 2 — WRITE the Character Profile with these sections:
- Name/Age/Gender (only if explicit in data; otherwise write "Unconfirmed, but acts like a [guess]")
- Hobbies & Interests: derived from Chrome/Search data. Pick the WEIRDEST juxtaposition of two interests, not the most obvious one.
- Fears, Insecurities, Desires: derived from searches. State them as if you caught the person red-handed, not like a diagnosis.
- Personality (via YouTube data): one sharp, backhanded observation, not a list of traits.

STYLE RULES (violating any of these is a failure):
- Never use: "seems to", "this suggests", "appears to be", "seemingly", "seems interested in"
- Every section needs at least one joke built on IRONY or CONTRAST (what they search for vs. what they claim to want), not just a funny adjective.
- Address the reader/subject in second person at least once ("you", not "the user")
- Max 2 sentences per section. Brevity forces sharper writing.

EXAMPLE (tone reference only, don't reuse content):
"Hobbies: Seventeen tabs on ergonomic keyboards, zero purchases in 8 months. You've optimized your search history for a desk setup you'll never buy."

Now write the profile.
"""


def extract_profile(data_md: str) -> str:
    """A sharp, sarcastic character profile of the person behind the activity.

    On the writing model rather than the planner: the whole point is the voice,
    and the planner writes it flat. `data_md` is the sampled activity markdown
    (build_user_description), inlined into the prompt inside <data> tags so the
    instructions and the data stay visibly apart.
    """
    return chat(_profile_prompt(data_md), model=WRITING_MODEL)


def _data_bank_prompt(data_md: str) -> str:
    return f"""

You are mining a Google Takeout export for the WEIRDEST, most specific, most narratively unusable-in-a-boring-way details about this person. You are not summarizing their life. You are hunting for the 20 details a novelist would actually want.

<data_chunk>
{data_md}
</data_chunk>

WHAT COUNTS AS GOOD (extract these):
- A specific, oddly-phrased search query, verbatim
- A contradiction: e.g. searches for "minimalism" right next to 40 shopping tabs
- A repeated, obsessive pattern (same search 8 times in different phrasings)
- A specific brand, place, or product name — never a category
- Something searched at a weird hour, or right after/before something else revealing
- A YouTube video/channel that clashes with their stated interests elsewhere

WHAT COUNTS AS BAD (never extract these):
- "Frequently searches for recipes"
- "Interested in technology"
- "Watches a variety of YouTube content"
- Anything you could write without looking at the actual data

For each item found, output:
{{
  "item": "<verbatim or near-verbatim detail>",
  "category": "search | youtube | maps | purchase | other",
  "weirdness_score": 1-10,
  "why": "<one sentence on what makes this usable — irony, contradiction, specificity, obsession>"
}}

Return 15-20 items as a JSON array, sorted by weirdness_score descending. If the chunk is mostly mundane, it is fine to return fewer — do not pad with weak items to hit a count.
"""


def mine_data_bank(data_md: str) -> str:
    """Pull 15-20 specific, quotable, weird details out of the activity, as JSON."""
    return chat(_data_bank_prompt(data_md), model=WRITING_MODEL)


def _reduce_data_bank_prompt(candidates: str) -> str:
    return f"""
Below are candidate details pulled from different chunks of one person's Google data. Merge, deduplicate near-identical items, and select the FINAL 25 that would make the best material for a comedic/dramatic character profile.

Prioritize:
- Items that contradict each other (great narrative tension)
- Items that are absurdly specific (brand names, exact phrasing)
- A spread across categories — don't let one obsessive search pattern eat all 25 slots

<candidates>
{candidates}
</candidates>

Return the final 25 as JSON, same schema, ready to be sampled from in the next generation step.
"""


def reduce_data_bank(candidates: str) -> str:
    """Consolidate the mined candidates to a final, deduplicated 25 (JSON)."""
    return chat(_reduce_data_bank_prompt(candidates), model=WRITING_MODEL)


# ---------------------------------------------------------------------------
# Preparation, phase 2 — the story: a full JSON outline, then one chapter's
# spec pulled back out of it per chapter.
# ---------------------------------------------------------------------------

def _outline_prompt(user_name: str, profile: str, data_bank: str) -> str:
    first = user_name.split()[0] if user_name.split() else user_name
    return f"""
	Based on the information below about {user_name}, create a detailed {NUM_CHAPTERS}-chapter outline for a personalized Harry Potter novel where {first} is the protagonist.

	Harry Potter must meet the protagonist early and reveal they are a witch/wizard. Build the main plot around one of the protagonist's most distinctive interests, fears, or guessed obsessions from their data. Include sustained bisexual romantic tension with established male and female Harry Potter characters — crushes, jealousy, longing, misunderstandings, embarrassment.

	CRITICAL STYLE RULE — DO NOT SMOOTH THIS OVER:
	Real-world personal data must NOT blend naturally into the wizarding world. It should feel like an intrusion — jarring, oddly specific, almost like the universe itself is glitching to accommodate {first}'s real habits. A wand shop clerk should not vaguely gesture at "modern tastes" — he should hold up a wand and say it "smells faintly of [SPECIFIC BRAND/PLACE FROM DATA]." The weirder and more forced the collision feels, the better. If a detail could be swapped for a generic placeholder and the sentence would still work, it's not specific enough — use a sharper one.

	OUTPUT FORMAT — this is critical, follow it exactly:
	Return ONLY a single JSON array, no preamble, no markdown code fences, no commentary before or after. One object per chapter, in this exact shape:

	[
	{{
		"chapter_number": 1,
		"chapter_title": "string",
		"plot_summary": "150-word detailed summary of what happens",
		"narrative_function": "two-word phrase describing the chapter's purpose",
		"data_intrusions": {{
		"setting": "a real place/brand/habit from the data, planted into a wizarding-world location or object, described as if completely normal",
		"dialogue": "a full line of dialogue from a Harry Potter character referencing a specific search query, video, or app from the data — verbatim or near-verbatim, delivered totally straight-faced",
		"conflict": "a plot complication that only makes sense because of a specific real-world fear, obsession, or contradiction found in the data",
		"romance": "a moment of romantic tension where the object of affection reacts to or is confused by a real personal detail/habit of the protagonist"
		}}
	}},
	... ({NUM_CHAPTERS - 1} more objects, chapter_number 2 through {NUM_CHAPTERS})
	]

	Book title (include as a top-level field "book_title" alongside the chapters array — restructure the whole return as {{"book_title": "Harry Potter and {user_name}", "chapters": [...]}}):

	{profile}

	User data:
	{data_bank}

	Note: Don't make it dull or generic. Comedy comes from precision, not vagueness — always choose the specific brand, exact phrase, or named place over a general description. Let it be a little uncomfortable that the story knows this much.

	Remember: output must be valid JSON only. No markdown fences, no explanation, nothing outside the JSON object.
	"""


def generate_outline(user_name: str, profile: str, data_bank: str) -> str:
    """The whole book as one JSON outline: book_title + a spec per chapter."""
    return chat(_outline_prompt(user_name, profile, data_bank))


def _chapter_outline_prompt(outline: str, index: int) -> str:
    return f"""
	Extract the especific text tat describes the info in the chapter #{index}, only that, keep the json formating as is, no further explanation:

	Chapters:
	{outline}
"""


def extract_chapter_outline(outline: str, index: int) -> str:
    """Pull chapter `index`'s spec back out of the outline JSON (1-based).

    A model call rather than JSON parsing on purpose: the outline is whatever
    the model actually emitted that run — it may not be valid JSON, or may be
    wrapped in prose — so asking for one chapter's slice is more robust than
    indexing a structure that might not exist.
    """
    return chat(_chapter_outline_prompt(outline, index))


# ---------------------------------------------------------------------------
# Preparation, phase 3 — the cover art.
# ---------------------------------------------------------------------------

def _art_prompt(profile: str) -> str:
    return f"""
	PROFILE:

	{profile}

	Pick the SINGLE most distinctive object from this profile — one
	only, the one that best identifies this person.

	Give me the name of the object only, no explanations.
	"""

ART_NEGATIVE_PROMPT = 'text, watermark, signature, people, faces, blurry'


def generate_art_prompt(profile: str) -> str:
    """A cover-art prompt: "Harry Potter and <the one most telling object>".

    One subject, because a prompt listing several just averages them into mush.
    The model returns only the object's name; we frame it into the cover prompt.
    """
    obj = chat(_art_prompt(profile)).strip().strip('"')
    return f'Harry Potter and {obj}'


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

def _chapter_prompt(chapter_outline: str, index: int, previous_chapter: str,
                    user_name: str) -> str:
    prompt = f"""

	Tarea

Eres un escritor de fanfiction y novelista. Escribe el siguiente capítulo de una novela de Harry Potter.

IMPORTANTE: Escribe todo el texto en español (castellano). No escribas ninguna frase en inglés, salvo títulos de vídeos, libros o nombres propios cuando aparezcan como referencias.

Voz y tono (mantener SIEMPRE, en todos los capítulos):
Irónico, mordaz, con humor seco ante lo absurdo. El choque entre el mundo real del protagonista (marcas, hábitos, búsquedas, obsesiones muy concretas) y el mundo mágico debe sentirse forzado a propósito, casi como si la trama se rompiera un poco para encajar esos detalles. Nunca suavices ni normalices esas intrusiones — cuanto más incómodo o ridículo resulte que la historia "sepa tanto" del protagonista, mejor. Incluye escenas de romance bisexual, tanto con chicos como con chicas, intimidad, emociones fuertes, violencia, pasión o conflicto. Inspírate en fan fictions homoeróticos de harry potter

Instrucciones
Escribe únicamente el capítulo, sin explicaciones ni resúmenes previos.
El protagonista es siempre {user_name}
Estilo: novela, no resumen. Prosa rica, fluida e inmersiva. Mucho diálogo.
Incluye conflicto, tensión narrativa, emociones fuertes y momentos de humor o vergüenza ajena y siempre con contenido romántico ni íntimo.
Los cuatro elementos de intrusión de datos (setting, diálogo, conflicto, y el que corresponda) deben aparecer de forma literal y reconocible, no diluidos ni generalizados.

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


# We emit the "# Capítulo N:" heading ourselves (so the index and the PDF's
# chapter splitting are guaranteed), but the writer often opens the prose by
# repeating it — "## Capítulo 1", "**Capítulo 1**", "Capítulo 1: título" — and
# then the number shows up twice, in the text and in the index. This matches
# that leading label so we can drop it. Only the number label: a title the model
# invents on its own ("**Un despertar**", "# El bosque") has no "capítulo"/
# number and is left alone.
_CHAPTER_LABEL_RE = re.compile(
    r'^\s*(?:'
    r'[#>]+\s*\**\s*(?:cap[íi]tulo|chapter)\b'            # "## Capítulo 1", "> Chapter 2"
    r'|\*+\s*(?:cap[íi]tulo|chapter)\b'                   # "**Capítulo 1**"
    r'|(?:cap[íi]tulo|chapter)\s+(?:\d+|[ivxlcdm]+)\b'    # "Capítulo 3", "Chapter IV"
    r')',
    re.IGNORECASE,
)
# Stop hunting for a label once this many chars have arrived without a line
# break: a real chapter label is a short line of its own, so anything longer is
# already prose and must be streamed, not held back.
_LABEL_SCAN_LIMIT = 120


def _could_be_chapter_label(partial_line: str) -> bool:
    """Could this partial (newline-less) line still grow into a chapter label?

    True while we can't yet rule it out — an empty run of markers, or a prefix
    of / start of "capítulo"/"chapter" — so the caller keeps buffering. False
    the moment it diverges, so real prose is released without waiting for a
    newline that a one-line-less chapter might never send.
    """
    core = partial_line.lstrip('#>*_ \t').lower()
    if core == '':
        return True
    return any(w.startswith(core) or core.startswith(w)
               for w in ('capítulo', 'capitulo', 'chapter'))


def _consume_leading_labels(buffer: str):
    """Strip leading blank lines and chapter-label lines off `buffer`.

    Returns (remaining, resolved). resolved=True once the head is real content —
    a complete non-label line has begun, or a partial that can't be a label.
    resolved=False means "need more input to decide".
    """
    while True:
        buffer = buffer.lstrip('\n')
        nl = buffer.find('\n')
        if nl == -1:
            return buffer, bool(buffer) and not _could_be_chapter_label(buffer)
        if not _CHAPTER_LABEL_RE.match(buffer[:nl]):
            return buffer, True
        buffer = buffer[nl + 1:]  # drop the label line, then look again


def _strip_leading_chapter_label(chunks):
    """Stream `chunks`, dropping a leading chapter-number label line if present.

    Buffers only the chapter's opening — through any leading blank and label
    lines until real content appears, or a short cap for a chapter that opens
    with one long unbroken sentence — then passes everything else through
    untouched, so it costs nothing past the first line. `iter(chunks)` up front,
    then `yield from` the same iterator, so a chunk is never consumed twice.
    """
    it = iter(chunks)
    buffer = ''
    resolved = False
    for chunk in it:
        buffer += chunk
        buffer, resolved = _consume_leading_labels(buffer)
        if resolved or len(buffer) > _LABEL_SCAN_LIMIT:
            break
    if not resolved:
        # The stream ended (or hit the cap) still undecided — a chapter that is
        # nothing but its label, no prose after. Treat end-of-input as
        # end-of-line: if what's left is itself just a label, drop it.
        buffer = buffer.lstrip('\n')
        if '\n' not in buffer and _CHAPTER_LABEL_RE.match(buffer):
            buffer = ''
    if buffer:
        yield buffer
    yield from it


# Streaming loop-breaker. A small local model sometimes falls into a groove and
# reproduces a paragraph it already wrote, again and again, until it hits the
# token limit. repeat_penalty/frequency_penalty (WRITING_PARAMS) curb short
# loops but not paragraph-scale ones, and this box ignores the window/DRY
# samplers that would (measured: no effect, same as a nonsense param). So we
# watch the prose as it streams and end the chapter the moment a substantial
# recent span turns out to repeat text from earlier in the same chapter.

# Chars of recent output that must reappear earlier to count as a loop. Long
# enough that ordinary repetition (a name, a refrain, a dialogue tag) never
# trips it — ~27 words verbatim don't recur by chance in prose — yet short
# enough to catch a looping paragraph a sentence or two after it starts.
_LOOP_WINDOW = 160
# Don't re-scan on every token; once per this many new chars is plenty and
# keeps the substring search off the hot path.
_LOOP_CHECK_STRIDE = 40


def _norm_for_loop(text: str) -> str:
    # Match case- and whitespace-insensitively, so a loop that differs only in
    # spacing or capitalisation is still caught: content is the signal, not layout.
    return re.sub(r'\s+', ' ', text).lower()


def _break_on_repetition(chunks):
    """Stream `chunks`, cutting off if the prose starts repeating itself.

    Passes text through unchanged until the most recent _LOOP_WINDOW characters
    are found to have already occurred earlier in the same chapter — the
    signature of a stuck model — then stops pulling from it and ends the chapter.
    """
    it = iter(chunks)
    norm = ''            # normalized full text so far, for the repeat check
    since_check = 0
    try:
        for chunk in it:
            yield chunk
            norm += _norm_for_loop(chunk)
            since_check += len(chunk)
            if since_check < _LOOP_CHECK_STRIDE or len(norm) < 2 * _LOOP_WINDOW:
                continue
            since_check = 0
            tail = norm[-_LOOP_WINDOW:]
            # Look for the tail only in the text that ends before it begins, so
            # it can never match itself — a hit there is a genuine earlier repeat.
            if tail in norm[:-_LOOP_WINDOW]:
                break
    finally:
        # Breaking out abandons the model mid-generation; close the underlying
        # stream promptly so the HTTP connection to the box is released now
        # rather than at the next garbage collection.
        close = getattr(it, 'close', None)
        if close:
            close()


def iter_chapter(chapter_outline: str, index: int, previous_chapter: str,
                 user_name: str):
    """Yield the chapter as text: a markdown heading, then streamed prose.

    Only the previous chapter is passed for continuity, not the book so far —
    that keeps the prompt bounded no matter how long the book runs.
    """
    yield f"# Capítulo {index}:\n\n"
    prose = stream_chat(
        _chapter_prompt(chapter_outline, index, previous_chapter, user_name))
    yield from _break_on_repetition(_strip_leading_chapter_label(prose))


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
    user_name = get_user_name(data)
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

    yield prep('Buscando los detalles más reveladores')
    candidates = mine_data_bank(user_description)

    yield prep('Analizando tus obsesioness')
    data_bank = reduce_data_bank(candidates)
    _save_planning(session_id, 'databank', data_bank)

    yield prep('Tramando la historia')
    outline = generate_outline(user_name, perfil, data_bank)
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
        for delta in iter_chapter(chapter_outline, i, previous_chapter, user_name):
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
