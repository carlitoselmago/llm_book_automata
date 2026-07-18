"""Dev-only scratch client for the local LM Studio backend.

Not imported by the app — it's a playground for poking at the API and trying
out chains of calls before any of it becomes real code in generate_book.py.

Every function takes `model=`, so a chain can mix models call by call, and
passes unknown keyword args straight through to the API (temperature,
max_tokens, top_p, …), so nothing here has to change when you want a new knob.

	import lm_dev

	lm_dev.models()                          # what's loaded in LM Studio
	lm_dev.chat("hola, quien eres?")         # -> str
	lm_dev.chat("resume esto", model="openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix")
	lm_dev.complete("Erase una vez")         # raw completion, no chat template
	for tok in lm_dev.chat("cuenta hasta 10", stream=True):
		print(tok, end="")                   # stream=True -> generator of deltas

Chaining is just Python — feed one result into the next call:

	outline = lm_dev.chat("dame 3 titulos de capitulo", temperature=0.4)
	prose   = lm_dev.chat(f"escribe el primero:\n{outline}", temperature=0.9)

To put a session's activity into the conversation (the API has no file upload —
it's just text in a message), pass `attach=` the markdown prompt file:

	lm_dev.chat("resume este perfil", attach="outputs/<id>_prompt.md")

Attach the .md, never the .json. The markdown is what generate_book feeds the
model: sampled to PROMPT_ITEMS_PER_SERVICE items per service, truncated, URL-
filtered and deduplicated — ~1.6k tokens, and it fits. The raw session .json is
~1.4MB / ~350k tokens against an 8k context and cannot fit by any margin.

outputs/<id>_prompt.md is written every time a book is generated (site or CLI).
For a session that hasn't been generated yet, render one in a line:

	import generate_book, json
	md = generate_book.build_user_description(json.load(open("outputs/<id>.json")))

Two things LM Studio does that will waste your time otherwise:

  - An unknown `model=` id is NOT an error: it quietly answers with whatever
	model is loaded. Typos silently test the wrong model — copy ids from
	models(), don't hand-write them.
  - Reasoning models spend `max_tokens` on their thinking first, so a low cap
	returns an empty string with the whole budget burned. Leave max_tokens off
	unless you're deliberately testing truncation.

Reasoning models (deepseek-r1) wrap their thinking in <think>…</think> and it
comes back verbatim here — that's deliberate, since when you're testing chains
you usually want to see it. To drop it from the text after the fact:

	from generate_book import _strip_think
	_strip_think(lm_dev.chat("..."))

To stop the model thinking in the first place there's no universal switch — it
depends on the model, and every knob below is just a **params passthrough, so
nothing here needs changing to use them:

	# Qwen3-family templates: a real off switch, either side of the wire
	lm_dev.chat("...", chat_template_kwargs={"enable_thinking": False})
	lm_dev.chat("... /no_think")

	# gpt-oss: always reasons (it's the harmony format), only turnable down
	lm_dev.chat("...", reasoning_effort="low")      # low | medium | high

	# deepseek-r1: trained to always think, and its template hard-codes the
	# opening <think>. Don't fight it — strip it, or pick another model.

Passing a knob to a model that doesn't know it is harmless: LM Studio ignores
unknown params rather than rejecting them (a junk param and a real one behave
identically). So the failure mode is "nothing happens", not an error — check
usage.reasoning_tokens to see whether it actually took.
"""
import json
import os
import re
from pathlib import Path

import requests

import book_builder
import generate_book

BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', 'http://192.168.100.138:1234')
MODEL = os.environ.get('LM_STUDIO_MODEL', 'deepseek/deepseek-r1-0528-qwen3-8b')
OUTPUT_DIR = Path(os.environ.get('OUTPUT_FOLDER', Path(__file__).parent / 'outputs'))

# The Stable Diffusion server — a different box/port from LM Studio.
SD_BASE_URL = os.environ.get('SD_BASE_URL', 'http://192.168.100.138:8000')

# (connect, read): fail fast if the backend is down, wait forever for tokens.
TIMEOUT = (10, None)
# Same, for diffusion: one silent wait ending in the whole PNG, so there are no
# tokens trickling back to prove it's alive — just let it take as long as it takes.
SD_TIMEOUT = (10, None)


def models() -> list[str]:
	"""The ids LM Studio currently serves — use one as a `model=` argument."""
	resp = requests.get(f'{BASE_URL}/v1/models', timeout=TIMEOUT)
	resp.raise_for_status()
	return [m['id'] for m in resp.json()['data']]


def load_md(path) -> str:
	"""Read a markdown (or any text) file into a string. The plain version of
	md_message() for when you just want the text, not a chat message."""
	return Path(path).read_text(encoding='utf-8')


def md_message(path, *, role: str = 'user', note: str = '',
			   max_chars: int = None) -> dict:
	"""A chat message carrying a markdown file — this is what "uploading" means
	here, since the API only ever takes text.

	Meant for outputs/<session_id>_prompt.md: the sampled, deduplicated activity
	summary that generate_book actually feeds the model. It goes in as-is, with
	no fence around it — markdown is already the model-facing format, and its
	own ## headings delimit the services.

	`max_chars` truncates, for deliberately testing what a smaller context does.
	"""
	text = Path(path).read_text(encoding='utf-8')
	if max_chars is not None and len(text) > max_chars:
		text = text[:max_chars] + '\n…(truncado)'
	return {'role': role, 'content': f'{note}\n\n{text}' if note else text}


def chat(prompt, *, model: str = MODEL, system: str = None, attach=None,
		 stream: bool = False, **params):
	"""Chat completion. `prompt` is a string or a full messages list.

	`attach` adds a markdown file (a path) as an extra message — the quick
	version of md_message(); build the message yourself when you need max_chars
	or a different role.

	Returns the response text, or a generator of text deltas if stream=True.
	"""
	if isinstance(prompt, str):
		messages = [{'role': 'user', 'content': prompt}]
	else:
		messages = list(prompt)
	if system:
		messages = [{'role': 'system', 'content': system}] + messages
	if attach is not None:
		messages = messages + [md_message(attach)]

	return _call('/v1/chat/completions',
				 {'messages': messages, 'model': model, **params},
				 stream=stream,
				 pick=lambda choice: choice['message']['content'],
				 pick_delta=lambda choice: choice['delta'].get('content'))


def complete(prompt: str, *, model: str = MODEL, stream: bool = False, **params):
	"""Raw text completion — no chat template applied to the prompt.

	Returns the completion text, or a generator of text deltas if stream=True.
	"""
	return _call('/v1/completions',
				 {'prompt': prompt, 'model': model, **params},
				 stream=stream,
				 pick=lambda choice: choice['text'],
				 pick_delta=lambda choice: choice.get('text'))


def image(prompt: str, path, **params):
	"""Generate a PNG on the Stable Diffusion server; returns the saved Path.

	Like chat(), unknown kwargs go straight to the API — negative_prompt, steps,
	guidance_scale, width, height, seed. It answers with the raw PNG bytes, not
	JSON, so an error body is the only thing worth decoding:

		lm_dev.image("a red banjo on a desk", "outputs/banjo.png")
		lm_dev.image("...", "out.png", width=512, height=768, seed=7)

	seed= makes a run repeatable; without it every call redraws from scratch.
	"""
	resp = requests.post(f'{SD_BASE_URL}/generate',
						 json={'prompt': prompt, **params}, timeout=SD_TIMEOUT)
	if resp.status_code >= 400:
		raise requests.HTTPError(f'{resp.status_code} from {resp.url}: '
								 f'{resp.text[:500].strip()}')
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(resp.content)
	return path


def _call(path: str, payload: dict, *, stream: bool, pick, pick_delta):
	resp = requests.post(f'{BASE_URL}{path}', json={**payload, 'stream': stream},
						 stream=stream, timeout=TIMEOUT)
	if resp.status_code >= 400:
		# LM Studio explains its 400s in the body (bad model id, context
		# overflow…), and the bare status alone is useless while poking around.
		raise requests.HTTPError(f'{resp.status_code} from {resp.url}: '
								 f'{resp.text[:500].strip()}')
	if stream:
		return _iter_deltas(resp, pick_delta)
	return pick(resp.json()['choices'][0]) or ''


def _iter_deltas(resp, pick_delta):
	with resp:
		for raw in resp.iter_lines():
			if not raw or not raw.startswith(b'data: '):
				continue
			data = raw[len(b'data: '):]
			if data.strip() == b'[DONE]':
				break
			text = pick_delta(json.loads(data)['choices'][0])
			if text:
				yield text


def _rtf_escape(text: str) -> str:
	"""One line of text -> RTF. RTF is a 7-bit format: everything else escapes.

	This is why saving isn't just write_text(): Spanish prose is full of á/ñ/¿/—
	and model output is full of emoji, and a raw byte of any of them corrupts
	the file for Word.
	"""
	out = []
	for ch in text:
		if ch in '\\{}':
			out.append('\\' + ch)
		elif ch == '\n':
			out.append('\\par\n')
		elif ch == '\t':
			out.append('\\tab ')
		elif ord(ch) < 128:
			out.append(ch)
		elif ord(ch) <= 0xFFFF:
			# \uN takes a *signed* 16-bit int, so anything above 32767 wraps.
			n = ord(ch)
			out.append(f'\\u{n - 65536 if n > 32767 else n}?')
		else:
			# Beyond the BMP (emoji): RTF wants the UTF-16 surrogate pair.
			c = ord(ch) - 0x10000
			for half in (0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF)):
				out.append(f'\\u{half - 65536}?')
	return ''.join(out)


# Markdown headings ("# Capítulo 1:", "## ..."), the only markdown we render.
_MD_HEADING = re.compile(r'^\s*(#{1,6})\s+(.*?)\s*#*\s*$')
# Sizes are RTF half-points: \fs36 = 18pt against the \fs24 = 12pt body.
_HEADING_FS = {1: 36, 2: 30, 3: 26}


def _rtf_body(text: str) -> str:
	"""Text -> RTF paragraphs, turning markdown headings into real headings.

	Each line becomes its own \\pard paragraph, which also resets formatting —
	without that, a heading's bold would bleed into the prose after it.
	"""
	out = []
	for line in text.split('\n'):
		heading = _MD_HEADING.match(line)
		if heading:
			fs = _HEADING_FS.get(len(heading.group(1)), 24)
			# keepn: don't let a heading get orphaned at the foot of a page.
			out.append(f'\\pard\\keepn\\sb240\\sa120\\b\\fs{fs} '
					   f'{_rtf_escape(heading.group(2))}\\b0\\fs24\\par\n')
		else:
			out.append(f'\\pard\\fs24 {_rtf_escape(line)}\\par\n')
	return ''.join(out)


def save_rtf(chapters, path) -> Path:
	"""Save text to a .rtf that Word / Pages / LibreOffice will open.

	`chapters` is a string, or a list of them — one chapter per page. Markdown
	headings (#, ##, ###) render as bold headings; everything else goes in as
	prose, verbatim.
	"""
	if isinstance(chapters, str):
		chapters = [chapters]
	body = '\\page\n'.join(_rtf_body(c) for c in chapters)
	rtf = ('{\\rtf1\\ansi\\ansicpg1252\\deff0'
		   '{\\fonttbl{\\f0\\froman Times New Roman;}}\n'
		   '\\fs24\n' + body + '\n}')
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	# ascii is the point, not an accident: if anything slipped past the escaper
	# this raises here rather than writing a file Word renders as mojibake.
	path.write_text(rtf, encoding='ascii')
	return path


if __name__ == '__main__':

	import time
	start_time = time.time()

	"""
	user_name="Fulanito Martínez"
	capitulos=["Hola","Adios"]
	stamp="___"
	cover_path = book_builder.build_cover_image(user_name, OUTPUT_DIR / f'libro_{stamp}_cover.png')
	pdf_path = book_builder.build_pdf(user_name, '\n\n'.join(capitulos), cover_path,
									  OUTPUT_DIR / f'libro_{stamp}.pdf')
	print(f'pdf guardado: {pdf_path}  (portada: {cover_path})')
	sys.exit()
	"""

	# The session this run writes about: its _prompt.md feeds the first call, and
	# its profile name goes on the cover.
	SESSION_ID = 'ea82752f-aecf-44cf-9ef0-3b80525f9fe4'
	user_name = generate_book.get_user_name(
		generate_book.load_session(OUTPUT_DIR / f'{SESSION_ID}.json'))
	data_md=load_md(OUTPUT_DIR / f'{SESSION_ID}_prompt.md')
	#test=chat('Escribe un chiste guarro', model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")
	#print(test)
	#print("")

	print('models:', *models(), sep='\n  ')
	print("")
	print("")
	prompt=f"""
	
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

	{data_md}
"""
	#text = chat(prompt, model="ibm/granite-4-h-tiny", attach=OUTPUT_DIR / f'{SESSION_ID}_prompt.md')
	text = chat(prompt, model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")
	print(text)
	print("")
	print("--- %s seconds ---" % (time.time() - start_time))
	print("###########################################################################################")
	print("")
	# Keep the profile: `text` is reassigned by the synopsis call below, and the
	# cover art prompt at the end needs the user's real references, not the plot.
	perfil = text
	


	prompt=f"""

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

	data_bank = chat(prompt, model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")
	print(data_bank)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("")


	reduce_prompt = f"""
Below are candidate details pulled from different chunks of one person's Google data. Merge, deduplicate near-identical items, and select the FINAL 25 that would make the best material for a comedic/dramatic character profile.

Prioritize:
- Items that contradict each other (great narrative tension)
- Items that are absurdly specific (brand names, exact phrasing)
- A spread across categories — don't let one obsessive search pattern eat all 25 slots

<candidates>
{data_bank}
</candidates>

Return the final 25 as JSON, same schema, ready to be sampled from in the next generation step.
"""

	data_bank = chat(prompt, model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")
	print(data_bank)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("")

prompt = f"""
	Based on the information below about {user_name}, create a detailed 10-chapter outline for a personalized Harry Potter novel where {user_name.split()[0]} is the protagonist.

	Harry Potter must meet the protagonist early and reveal they are a witch/wizard. Build the main plot around one of the protagonist's most distinctive interests, fears, or guessed obsessions from their data. Include sustained bisexual romantic tension with established male and female Harry Potter characters — crushes, jealousy, longing, misunderstandings, embarrassment.

	CRITICAL STYLE RULE — DO NOT SMOOTH THIS OVER:
	Real-world personal data must NOT blend naturally into the wizarding world. It should feel like an intrusion — jarring, oddly specific, almost like the universe itself is glitching to accommodate {user_name.split()[0]}'s real habits. A wand shop clerk should not vaguely gesture at "modern tastes" — he should hold up a wand and say it "smells faintly of [SPECIFIC BRAND/PLACE FROM DATA]." The weirder and more forced the collision feels, the better. If a detail could be swapped for a generic placeholder and the sentence would still work, it's not specific enough — use a sharper one.

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
	... (9 more objects, chapter_number 2 through 10)
	]

	Book title (include as a top-level field "book_title" alongside the chapters array — restructure the whole return as {{"book_title": "Harry Potter and {user_name}", "chapters": [...]}}):

	{perfil}

	User data:
	{data_bank}

	Note: Don't make it dull or generic. Comedy comes from precision, not vagueness — always choose the specific brand, exact phrase, or named place over a general description. Let it be a little uncomfortable that the story knows this much.

	Remember: output must be valid JSON only. No markdown fences, no explanation, nothing outside the JSON object.
	"""

outline = chat(prompt, model="ibm/granite-4-h-tiny")
print(outline)
print("--- %s seconds ---" % (time.time() - start_time))
print("")



chapter_outlines=[]

for i in range(0,10):

	prompt=f"""
	Extract the especific text tat describes the info in the chapter #{i+1}, only that, keep the json formating as is, no further explanation:

	Chapters:
	{outline}
"""

	chapter_outline=chat(prompt, model="ibm/granite-4-h-tiny")
	chapter_outlines.append(chapter_outline)
	print(chapter_outline)
	print("")
	print("#########################################################")
	print("")

print("--- %s seconds ---" % (time.time() - start_time))
print("")

# Escritura del libro en sí

capitulos=[]
capitulo=""
for i, e in enumerate(chapter_outlines):
	# f-string: without the f, "{e}" reaches the model as those three literal
	# characters and every chapter gets written from an empty spec.
	prompt=f"""

	Tarea

Eres un escritor de fanfiction y novelista. Escribe el siguiente capítulo de una novela de Harry Potter.

IMPORTANTE: Escribe todo el texto en español (castellano). No escribas ninguna frase en inglés, salvo títulos de vídeos, libros o nombres propios cuando aparezcan como referencias.

Voz y tono (mantener SIEMPRE, en todos los capítulos):
Irónico, mordaz, con humor seco ante lo absurdo. El choque entre el mundo real del protagonista (marcas, hábitos, búsquedas, obsesiones muy concretas) y el mundo mágico debe sentirse forzado a propósito, casi como si la trama se rompiera un poco para encajar esos detalles. Nunca suavices ni normalices esas intrusiones — cuanto más incómodo o ridículo resulte que la historia "sepa tanto" del protagonista, mejor. Incluye escenas de romance bisexual, tanto con chicos como con chicas, intimidad , emociones fuertes, violencia, pasión o conflicto. Inspírate en fan fictions homoeróticos de harry potter

Instrucciones
Escribe únicamente el capítulo, sin explicaciones ni resúmenes previos.
El protagonista es siempre {user_name}
Estilo: novela, no resumen. Prosa rica, fluida e inmersiva. Mucho diálogo.
Incluye conflicto, tensión narrativa, emociones fuertes y momentos de humor o vergüenza ajena — pero sin contenido romántico ni íntimo.
Los cuatro elementos de intrusión de datos (setting, diálogo, conflicto, y el que corresponda) deben aparecer de forma literal y reconocible, no diluidos ni generalizados.


"""

	if (i>0):
		prompt+=f"""

	Capítulo anterior para contexto:
	{capitulo}
	"""

	prompt+="""
Escribe:
"""

	capitulo=chat(prompt, model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")

	#Añadimos el título
	capitulo=f"# Capítulo {i+1}:\n\n{capitulo}"

	capitulos.append(capitulo)
	print(capitulo)
	print("")
	print("#########################################################")
	print("")

# Timestamped: a run takes long enough that clobbering the previous one hurts.
stamp = time.strftime("%Y%m%d_%H%M%S")
rtf_path = save_rtf(capitulos, OUTPUT_DIR / f'libro_{stamp}.rtf')
print(f'libro guardado: {rtf_path}  ({len(capitulos)} capitulos, {sum(len(c) for c in capitulos)} chars)')

# Cover art: ask for ONE distinctive thing from the profile, then draw it.
# One subject, because a prompt listing several just averages them into mush.
prompt=f"""
PROFILE:

{perfil}

Pick the SINGLE most distinctive object from this profile — one
only, the one that best identifies this person.
.	
Give me the name of the object only, no explanations.

"""
""""""
sd_prompt = "Harry Potter and "+chat(prompt, model="ibm/granite-4-h-tiny").strip().strip('"')
print(f'prompt de portada: {sd_prompt}')

# Shaped like the cover's art window, so nothing worth seeing gets cropped
# away or hidden behind the title band.
art_w, art_h = book_builder.cover_art_size()
art_path = image(sd_prompt, OUTPUT_DIR / f'libro_{stamp}_art.png',
				negative_prompt='text, watermark, signature, people, faces, blurry',
				width=art_w, height=art_h)
print(f'ilustracion guardada: {art_path}')

# Same chapters through the real book pipeline: cover PNG + A5 PDF with
# intro, index and page numbers. parse_chapters keys off the "# Capítulo N:"
# headings added above, so the text goes in as-is.
cover_path = book_builder.build_cover_image(user_name, OUTPUT_DIR / f'libro_{stamp}_cover.png',
											background_path=art_path)
pdf_path = book_builder.build_pdf(user_name, '\n\n'.join(capitulos), cover_path,
								OUTPUT_DIR / f'libro_{stamp}.pdf')
print(f'pdf guardado: {pdf_path}  (portada: {cover_path})')
print("--- %s seconds ---" % (time.time() - start_time))
print("")
""""""