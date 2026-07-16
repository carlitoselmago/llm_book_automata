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
	SESSION_ID = '3901aa49-f142-4408-ad37-11d740eff2b3'#'ff53bf70-fc82-4837-b7db-ecf9efbc734b'
	user_name = generate_book.get_user_name(
		generate_book.load_session(OUTPUT_DIR / f'{SESSION_ID}.json'))

	#test=chat('Escribe un chiste guarro', model="dirty-muse-writer-v01-uncensored-erotica-nsfw-i1")
	#print(test)
	#print("")

	print('models:', *models(), sep='\n  ')
	print("")
	print("")
	prompt="""
	
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
	text = chat(prompt, model="ibm/granite-4-h-tiny", attach=OUTPUT_DIR / f'{SESSION_ID}_prompt.md')
	print(text)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("")

	# Keep the profile: `text` is reassigned by the synopsis call below, and the
	# cover art prompt at the end needs the user's real references, not the plot.
	perfil = text


	# f-string: without the f, "{text}" goes to the model as those six literal
	# characters and the chain is silently broken.
	prompt=f"""

		TEXT:

		{text}

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

	text = chat(prompt, model="ibm/granite-4-h-tiny")
	print(text)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("")

	prompt=f"""
	TEXT:

		{text}

	The information provided in the text consists of:

	1. A narrative profile of the protagonist extracted from real Google activity.
	2. A story synopsis for a personalized Harry Potter novel.

	Use both as the only source of information.

	Your task is to expand the synopsis into a detailed outline of **10 chapters**.

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

	outline = chat(prompt, model="ibm/granite-4-h-tiny")
	print(text)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("")

	chapter_outlines=[]

	for i in range(0,10):

		prompt=f"""
		Extract the especific text tat describes the info in the chapter #{i+1}, only that, no further explanation:

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

	for i, e in enumerate(chapter_outlines):
		# f-string: without the f, "{e}" reaches the model as those three literal
		# characters and every chapter gets written from an empty spec.
		prompt=f"""
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

	{e}

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