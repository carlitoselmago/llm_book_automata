import os
import re
import threading
import time
import uuid
import zipfile
import json
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename

import book_builder
import generate_book
from generate_book import (
    EngineUnavailable, art_path, get_user_name, iter_book_events,
)

# What the reader is told when the writing box is down or its model died. The
# underlying detail (a 400 "terminated", a refused connection) is logged, not
# shown: it's for us, and it isn't something they can act on.
ENGINE_ERROR_MESSAGE = (
    'El motor de escritura no está disponible en este momento. '
    'Inténtalo de nuevo en unos minutos.'
)

# The writing box fits one book at a time. A second reader isn't broken, just
# early, so they get their own wording rather than the generic engine failure.
BUSY_MESSAGE = (
    'El sistema está ocupado generando otro libro, '
    'reinténtalo en unos minutos.'
)

app = Flask(__name__)
# Keep the secret key stable across restarts (via SECRET_KEY) so the
# book-session cookie survives a redeploy; fall back to a random key locally.
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)
# Max upload size — overridable so ops can tune it per deployment (MB).
_max_mb = int(os.environ.get('MAX_UPLOAD_MB', '500'))
app.config['MAX_CONTENT_LENGTH'] = _max_mb * 1024 * 1024



BASE_DIR = Path(__file__).parent
# Data dirs default to the project folder but can point at a mounted volume
# on a server (outputs/ holds the persistent library + generated PDFs).
UPLOAD_FOLDER = Path(os.environ.get('UPLOAD_FOLDER', BASE_DIR / 'uploads'))
OUTPUT_FOLDER = Path(os.environ.get('OUTPUT_FOLDER', BASE_DIR / 'outputs'))
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


MAX_ITEMS_PER_SERVICE = 500

# How many items to skip after each one we keep. Takeout lists activity newest
# first, so taking the first N would only ever see the last few days; skipping
# spreads the sample further back. 0 keeps everything, 1 keeps every other item,
# 2 keeps one in three, and so on.
SKIP_ITEMS = 1

# We find the profile by parsing JSON files rather than by name (names are
# localized), so cap what we're willing to read: a profile is a few KB, while
# a Takeout can carry a 200MB Location History.json we must not load to check.
PROFILE_MAX_BYTES = 1_000_000

# Services to ignore — they hold no useful personal narrative data. Takeout
# localizes folder names, so each service is listed in English plus its Spanish
# translation; brand names Takeout keeps in English everywhere need one entry.
BLACKLISTED_SERVICES = {
    'Ads', 'Anuncios',
    'Assistant', 'Asistente',
    'Android',
    'Android TV',
    'Discover',
    'Drive',
    'Developers', 'Desarrolladores',
    'Flights', 'Vuelos',
    'Google Arts & Culture', 'Google Arts and Culture',
    'Google Business Profile', 'Perfil de Empresa de Google',
    'Google Lens',
    'Google News', 'Google Noticias', 'Noticias',
    'Google Play Console',
    'Google Play Games', 'Google Play Juegos',
    'Google Play Movies & TV', 'Google Play Películas y TV',
    'Google Play Store',
    'Google Translate', 'Traductor de Google',
    'Google TV',
    'Help', 'Ayuda',
    'Hotels', 'Hoteles',
    'Image Search', 'Búsqueda de imágenes',
    'Song Search', 'Búsqueda de canciones',
    'Takeout',
    'Voice Match', 'Coincidencia de voz',
}


# ---------------------------------------------------------------------------
# Activity HTML parsing
# One generic parser handles every service in every language. We never key off
# English service names or English text ("Searched for", "Prompted", month
# abbreviations…): the MyActivity HTML layout is identical across locales, so
# we read the cell structure instead of its words.
# ---------------------------------------------------------------------------

_RE_OUTER_CELL = re.compile(
    rb'<div class="outer-cell[^"]*">.*?</div>\s*</div>\s*</div>',
    re.DOTALL,
)
_RE_CONTENT_CELL = re.compile(
    rb'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>',
    re.DOTALL,
)
_RE_ANCHOR = re.compile(rb'<a href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
_RE_BR = re.compile(rb'<br\s*/?>', re.IGNORECASE)
_RE_TAG = re.compile(rb'<[^>]+>')
_RE_ENTITY = {b'&amp;': b'&', b'&lt;': b'<', b'&gt;': b'>', b'&quot;': b'"',
              b'&#39;': b"'", b'&emsp;': b' ', b'&nbsp;': b' '}
_RE_HTML_ENTITY = re.compile(b'|'.join(re.escape(k) for k in _RE_ENTITY))
# A timestamp line always carries a clock time (HH:MM:SS) whatever the locale,
# e.g. "Sep 30, 2025, 5:32:10 PM CEST" / "11 jul 2026, 21:16:09 CEST" /
# "30 set 2025, 17:32:10 CET". That lets us find it without knowing the
# language's month names or date order.
_RE_TIME = re.compile(r'\d{1,2}:\d{2}:\d{2}')


def _strip_html(raw: bytes) -> str:
    text = _RE_BR.sub(b'\n', raw)
    text = _RE_TAG.sub(b'', text)
    text = _RE_HTML_ENTITY.sub(lambda m: _RE_ENTITY[m.group(0)], text)
    return text.decode('utf-8', errors='replace').strip()


def parse_activity_html(html_bytes: bytes, max_items: int,
                        skip: int = SKIP_ITEMS) -> list[dict]:
    """Extract activity items from a Google MyActivity HTML file using regex.

    Language-agnostic: it keeps each cell's full text as the item content and
    locates the timestamp by its clock time, rather than matching English
    words. This makes one parser work for every service and every Takeout
    locale (English, Spanish, Catalan, French, Italian, …).

    Keeps one item then skips `skip` of them, up to `max_items`, so the sample
    is spread over the export's whole date range rather than its newest corner.

    Uses regex instead of a DOM parser so it stays fast on 100MB+ files.
    """
    items = []
    stride = skip + 1
    for i, m in enumerate(_RE_OUTER_CELL.finditer(html_bytes)):
        if len(items) >= max_items:
            break
        if i % stride:
            continue

        cc = _RE_CONTENT_CELL.search(m.group(0))
        if not cc:
            continue
        inner = cc.group(1)

        anchor = _RE_ANCHOR.search(inner)
        url = anchor.group(1).decode('utf-8', errors='replace') if anchor else None

        # Split the cell into text lines. The line carrying a clock time is the
        # timestamp; every other line is activity content (the action verb plus
        # the query / title / prompt / app name — whatever the service logged).
        lines = [l.strip() for l in _strip_html(inner).split('\n') if l.strip()]
        timestamp = next((l for l in lines if _RE_TIME.search(l)), None)
        title = ' '.join(l for l in lines if not _RE_TIME.search(l)).strip() or None

        if title or timestamp:
            items.append({'title': title, 'url': url, 'timestamp': timestamp})

    return items


_STRIP_FIELDS = {'url', 'timestamp'}
_RE_URL = re.compile(r'https?://|www\.', re.IGNORECASE)


def _clean_items(items: list[dict]) -> list[dict]:
    """Keep only fields worth narrating: no url/timestamp, and nothing holding a
    URL anywhere in its value (a raw link tells the model nothing about the
    person). Items left with no field at all are dropped.

    Repeats are dropped too, keeping the first: people search the same thing
    over and over, and the model learns nothing from the 30th "Buscaste renfe"
    that it didn't learn from the first. Called once per service, so this
    de-duplicates within a service and leaves cross-service repeats alone.
    """
    result = []
    seen = set()
    for item in items:
        cleaned = {k: v for k, v in item.items()
                   if k not in _STRIP_FIELDS and v is not None and v != ''
                   and not _RE_URL.search(str(v))}
        if not cleaned:
            continue
        key = tuple(sorted(cleaned.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def parse_profile_json(raw: bytes) -> dict:
    """Parse Takeout/Profile/Profile.json into a flat profile dict."""
    try:
        data = json.loads(raw.decode('utf-8'))
    except Exception:
        return {}

    profile = {}

    # Name — Google uses several possible shapes
    name_block = data.get('name', {})
    profile['name'] = (
        name_block.get('formattedName')
        or f"{name_block.get('givenName', '')} {name_block.get('familyName', '')}".strip()
        or data.get('displayName')
    ) or None

    # Birthday — may be "YYYY-MM-DD" string or {"year":…,"month":…,"day":…}
    bday = data.get('birthday')
    if isinstance(bday, str):
        profile['birthday'] = bday
    elif isinstance(bday, dict):
        y = bday.get('year', '')
        m = str(bday.get('month', '')).zfill(2)
        d = str(bday.get('day', '')).zfill(2)
        profile['birthday'] = f'{y}-{m}-{d}' if y else None
    else:
        profile['birthday'] = None

    # Gender
    gender_block = data.get('gender', {})
    profile['gender'] = (
        gender_block.get('type') if isinstance(gender_block, dict) else gender_block
    ) or None

    # Primary email
    emails = data.get('emails', [])
    profile['email'] = emails[0].get('value') if emails else None

    return {k: v for k, v in profile.items() if v is not None}


def process_takeout_zip(zip_path: Path, session_id: str) -> dict:
    """Parse the profile + every activity HTML in the zip.

    Folder names are ignored entirely: any .html file is treated as an activity
    export whose service is the folder containing it, and any .json that parses
    into a profile is the profile. That way the zip works whatever language
    Takeout used, without a table of translated folder names to maintain.
    """
    result = {
        'session_id': session_id,
        'profile': {},
        'services': {},
        'skipped_services': [],
    }

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            name = info.filename
            parts = name.split('/')
            if len(parts) < 2:
                continue
            # Takeout sometimes puts non-breaking spaces (\xa0) in folder names
            # (e.g. "Google\xa0Play\xa0Games"); normalize them to plain spaces.
            service_name = parts[-2].replace('\xa0', ' ')
            lower = parts[-1].lower()

            if lower.endswith('.json'):
                if not result['profile'] and info.file_size <= PROFILE_MAX_BYTES:
                    result['profile'] = parse_profile_json(zf.read(name))
                continue

            if not lower.endswith('.html'):
                continue

            if service_name in BLACKLISTED_SERVICES:
                if service_name not in result['skipped_services']:
                    result['skipped_services'].append(service_name)
                continue

            items = _clean_items(parse_activity_html(zf.read(name), MAX_ITEMS_PER_SERVICE))
            if items:
                result['services'][service_name] = {
                    'count': len(items),
                    'items': items,
                }

    return result


@app.route('/')
def home():
    library = list(reversed(_load_library()))
    return render_template('home.html', library=library)


@app.route('/crear')
def crear():
    return render_template('index.html', step=1)


@app.route('/privacidad')
def privacidad():
    return render_template('privacidad.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        return _handle_upload()
    except Exception as e:
        app.logger.exception('Upload failed')
        return jsonify({'error': f'Error interno del servidor: {e}'}), 500


def _handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo.'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Nombre de archivo vacío.'}), 400

    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Solo se aceptan archivos .zip de Google Takeout.'}), 400

    session_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    zip_path = UPLOAD_FOLDER / f'{session_id}_{filename}'
    file.save(zip_path)

    try:
        if not zipfile.is_zipfile(zip_path):
            return jsonify({'error': 'El archivo no es un ZIP válido.'}), 400

        data = process_takeout_zip(zip_path, session_id)
        output_path = OUTPUT_FOLDER / f'{session_id}.json'
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    finally:
        zip_path.unlink(missing_ok=True)

    services_summary = {
        name: info['count']
        for name, info in data['services'].items()
    }

    resp = jsonify({
        'session_id': session_id,
        'services': services_summary,
        'skipped_services': data['skipped_services'],
        'total_items': sum(services_summary.values()),
    })
    # Remembers which session this browser belongs to, so the book-writing
    # area can resume after the tab is closed/reopened — no login involved.
    resp.set_cookie(
        BOOK_COOKIE, session_id,
        max_age=BOOK_COOKIE_MAX_AGE, httponly=True, samesite='Lax',
    )
    return resp


# ---------------------------------------------------------------------------
# Book generation — runs in a background thread per session, independent of
# any single HTTP request, so it keeps going after the browser disconnects.
# Progress is kept in memory and mirrored to disk so a reconnecting client
# (or a fresh request after a server restart) can see what was written.
# ---------------------------------------------------------------------------

BOOK_COOKIE = 'book_session'
BOOK_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
BOOK_POLL_INTERVAL = 0.3  # seconds between checks for new content in the SSE loop

_book_lock = threading.Lock()
_book_sessions: dict[str, dict] = {}  # session_id -> {'status', 'text', 'error'}

# Session id of the book being written right now, or None. One at a time: the
# LLM box serves a single generation, and letting a second one in would make
# both crawl (or fail) rather than making anyone's book arrive sooner.
#
# In-process on purpose — same reason the app runs a single gunicorn worker
# (see gunicorn.conf.py). It also means a restart clears it, which is what we
# want: the thread it was guarding died with the old process.
_active_generation: str | None = None


def _busy_for(session_id: str) -> bool:
    """Is another session's book being written right now?

    False for the session that owns the running generation, so reconnecting to
    your own book never looks like someone else's traffic jam.
    """
    with _book_lock:
        return _active_generation not in (None, session_id)


# ---------------------------------------------------------------------------
# Retention. A Takeout export is the most personal thing a reader will ever
# hand us — years of searches, locations, messages. It is raw material for one
# book and nothing else, so it lives exactly as long as it's being used:
#
#   the .zip          deleted the moment it's parsed (see _handle_upload)
#   the activity      deleted as soon as the book it fed is finished
#   the book itself   deleted with the session; only the library keeps a copy
#
# Anything left behind by a reader who never finished is swept after
# SESSION_TTL. The rule to keep in mind when adding files: everything named
# <session_id>_* is disposable, and every disposable file belongs in
# _session_files() below so the sweeper can find it.
# ---------------------------------------------------------------------------

# How long an unfinished session's data survives. Long enough to retry a book
# that died at chapter nine, or to come back after lunch to a Takeout you
# already uploaded — not long enough to be a store of anyone's activity.
SESSION_TTL = float(os.environ.get('SESSION_TTL_HOURS', '24')) * 3600
SWEEP_INTERVAL = float(os.environ.get('SWEEP_INTERVAL_MINUTES', '60')) * 60


def _book_file(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_book.txt'


def _book_done_marker(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_book.done'


def _session_data_path(session_id: str) -> Path:
    """The parsed Takeout: every activity item we kept. The sensitive one."""
    return OUTPUT_FOLDER / f'{session_id}.json'


def _meta_path(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_meta.json'


def _planning_paths(session_id: str) -> list[Path]:
    """The chain's debug mirrors. Write-only — nothing reads them back — but
    they are not harmless: _prompt.md is the activity re-serialised and
    _perfil.md is a profile of the person written from it."""
    return [
        OUTPUT_FOLDER / f'{session_id}_prompt.md',
        *(OUTPUT_FOLDER / f'{session_id}_{name}.md' for name in
          ('perfil', 'sinopsis', 'outline', 'capitulos', 'portada')),
        art_path(session_id),
    ]


def _save_session_meta(session_id: str, data: dict) -> None:
    """Leave behind the little that has to outlive the Takeout data.

    Only the name, and only because the finished book already carries it — it's
    printed on the cover, baked into the PDF and stored in library.json. Keeping
    it here lets the activity be deleted the moment the book is written while
    /book/download and /library/add still know whose book this is.
    """
    _meta_path(session_id).write_text(
        json.dumps({'user_name': get_user_name(data)}, ensure_ascii=False))


def _purge_session_source(session_id: str) -> None:
    """Delete the Takeout data and everything derived from it.

    Called the moment a book is finished: the activity has done its only job,
    and the book doesn't need it to be read, downloaded or kept. What survives
    is the book, and a meta record holding the reader's name.

    Deliberately NOT called when generation fails — the reader can still press
    "Generar libro" again, and that needs the activity. The sweeper takes it
    instead, once the session has gone cold.
    """
    for path in (_session_data_path(session_id), *_planning_paths(session_id)):
        path.unlink(missing_ok=True)


def _load_book_state(session_id: str) -> dict:
    """Get (or lazily rebuild from disk) the in-memory state for a session."""
    with _book_lock:
        state = _book_sessions.get(session_id)
        if state is not None:
            return state

        book_path = _book_file(session_id)
        if book_path.exists():
            text = book_path.read_text(encoding='utf-8')
            # A process restart loses the in-flight thread; if there's no
            # "done" marker the generation was interrupted mid-way. We can't
            # resume the LLM call itself, so we just surface what was saved.
            status = 'done' if _book_done_marker(session_id).exists() else 'interrupted'
            state = {'status': status, 'text': text, 'error': None,
                     'error_kind': None, 'progress': None}
        else:
            state = {'status': 'none', 'text': '', 'error': None,
                     'error_kind': None, 'progress': None}
        _book_sessions[session_id] = state
        return state


# A run that ended badly leaves the session pointing at a half-written book.
# Neither status can improve on its own, and neither has anything worth keeping,
# so both are a fresh start waiting for the reader to ask for one.
_RETRYABLE_STATUSES = ('error', 'interrupted')


def _reset_for_retry(session_id: str, state: dict) -> None:
    """Put a failed/interrupted session back to square one.

    Without this a reader whose book died — the engine dropped, the process
    restarted mid-chapter — could never press "Generar libro" again: the status
    stays 'error' forever and every later attempt is turned away as already
    started. The partial text goes too; a book resumed from half a chapter that
    no model remembers writing isn't worth keeping.
    """
    with _book_lock:
        state.update({'status': 'none', 'text': '', 'error': None,
                      'error_kind': None, 'progress': None})
    _book_file(session_id).unlink(missing_ok=True)
    _book_done_marker(session_id).unlink(missing_ok=True)


def _append_book_text(session_id: str, state: dict, chunk: str) -> None:
    with _book_lock:
        state['text'] += chunk
    with open(_book_file(session_id), 'a', encoding='utf-8') as f:
        f.write(chunk)


def _run_book_generation(session_id: str, state: dict, data: dict) -> None:
    global _active_generation
    try:
        for event in iter_book_events(data):
            if event['type'] == 'content':
                _append_book_text(session_id, state, event['text'])
            elif event['type'] == 'status':
                # Progress for the UI (e.g. the outline phase, before any prose
                # exists) — kept in memory only; the SSE loop forwards changes.
                with _book_lock:
                    state['progress'] = event
        with _book_lock:
            state['status'] = 'done'
            state['progress'] = None
        _book_done_marker(session_id).touch()
        # The book exists, so the activity that fed it has no further use.
        # Meta first: it's what lets the rest go without breaking download.
        _save_session_meta(session_id, data)
        _purge_session_source(session_id)
    except EngineUnavailable as e:
        # Expected enough to not be worth a stack trace: the box is off, busy,
        # or its model got ejected. Logged as a warning so it stays greppable.
        app.logger.warning('LLM engine unavailable for session %s: %s', session_id, e)
        with _book_lock:
            state['status'] = 'error'
            state['error'] = ENGINE_ERROR_MESSAGE
            state['error_kind'] = 'engine'
    except Exception as e:
        app.logger.exception('Book generation failed for session %s', session_id)
        with _book_lock:
            state['status'] = 'error'
            state['error'] = str(e)
            state['error_kind'] = 'unknown'
    finally:
        # However this ended, the box is free again — release it here rather
        # than on each exit path, so a new failure mode can't wedge the queue.
        with _book_lock:
            if _active_generation == session_id:
                _active_generation = None


def _ensure_book_started(session_id: str) -> dict | None:
    """Return the session's state, starting the background writer if needed.

    Returns None if another session already holds the writing box — the caller
    should tell this reader to come back later. Reconnecting to an already
    running book is never blocked; only claiming a free box is.
    """
    global _active_generation
    state = _load_book_state(session_id)
    with _book_lock:
        should_start = state['status'] == 'none'
        if should_start:
            if _active_generation not in (None, session_id):
                return None
            state['status'] = 'running'
            _active_generation = session_id
    if should_start:
        try:
            data_path = OUTPUT_FOLDER / f'{session_id}.json'
            data = json.loads(data_path.read_text(encoding='utf-8'))
            disabled_services = set(data.get('disabled_services', []))
            if disabled_services:
                data = {
                    **data,
                    'services': {
                        name: info for name, info in data['services'].items()
                        if name not in disabled_services
                    },
                }
            threading.Thread(
                target=_run_book_generation,
                args=(session_id, state, data),
                daemon=True,
            ).start()
        except Exception:
            # We claimed the box but never got a thread onto it (unreadable
            # session file, thread refused to start). Hand it back, or nobody
            # else can generate until a restart.
            with _book_lock:
                state['status'] = 'none'
                if _active_generation == session_id:
                    _active_generation = None
            raise
    return state


@app.route('/generate/configure', methods=['POST'])
def generate_configure():
    """Let the user disable specific services before generation starts."""
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'error': 'No hay una sesión activa.'}), 400

    if _busy_for(session_id):
        # Not the real gate (that's _ensure_book_started, which does the
        # claiming) — this just stops the page from moving on to the writing
        # view when we already know there's no box to write on. Checked before
        # the retry reset below, so a reader turned away here still has their
        # last attempt on screen to come back to.
        return jsonify({'error': BUSY_MESSAGE, 'error_kind': 'busy'}), 409

    # Book state before file state: a finished session has had its activity
    # purged, so "the book is already written" is the true answer there, not
    # the "session not found" the missing file would otherwise produce.
    state = _load_book_state(session_id)
    # Pressing "Generar libro" after a failure means "try again", so honour it.
    if state['status'] in _RETRYABLE_STATUSES:
        _reset_for_retry(session_id, state)
    if state['status'] != 'none':
        return jsonify({'error': 'La generación ya ha comenzado.'}), 409

    output_path = _session_data_path(session_id)
    if not output_path.exists():
        return jsonify({'error': 'Sesión no encontrada.'}), 404

    payload = request.get_json(silent=True) or {}
    disabled_services = payload.get('disabled_services', [])
    if not isinstance(disabled_services, list):
        return jsonify({'error': 'Formato inválido.'}), 400

    data = json.loads(output_path.read_text(encoding='utf-8'))
    data['disabled_services'] = [
        name for name in disabled_services
        if isinstance(name, str) and name in data.get('services', {})
    ]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return jsonify({'ok': True})


@app.route('/engine/status')
def engine_status():
    """Can this reader start a book right now?

    Its own route rather than a field on /generate/status: this one costs a
    round-trip to another machine, and the page asks it at different moments
    (before generating, after a failure) than it asks about the session.

    "Engine down" and "engine taken" are different causes with the same shape
    for the reader — can't start, wait a few minutes, press retry — so they
    share this answer and differ only in `reason` and wording.
    """
    session_id = request.cookies.get(BOOK_COOKIE, '')
    if _busy_for(session_id):
        # Checked before the round-trip: the answer is no either way, and
        # there's no point poking the box while it's mid-book.
        return jsonify({
            'available': False,
            'reason': 'busy',
            'message': BUSY_MESSAGE,
        })

    status = generate_book.engine_status()
    if not status['available']:
        app.logger.warning('Engine check failed (%s): %s',
                           status['reason'], status['detail'])
    return jsonify({
        'available': status['available'],
        'reason': status['reason'],
        'message': None if status['available'] else ENGINE_ERROR_MESSAGE,
    })


@app.route('/generate/status')
def generate_status():
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'has_session': False})

    output_path = _session_data_path(session_id)
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding='utf-8'))
        services_summary = {name: info['count']
                            for name, info in data['services'].items()}
        skipped = data.get('skipped_services', [])
        disabled = data.get('disabled_services', [])
    elif _meta_path(session_id).exists():
        # Finished: the activity has been purged and the service lists went
        # with it. The session is still real — its book is right there — so
        # this must not report "no session", or the page would drop the reader
        # back at step 1 with their finished book still on screen. The empty
        # lists only feed the step-2 picker, which is behind them by now.
        services_summary, skipped, disabled = {}, [], []
    else:
        return jsonify({'has_session': False})

    state = _load_book_state(session_id)

    return jsonify({
        'has_session': True,
        'session_id': session_id,
        'book_status': state['status'],
        'services': services_summary,
        'skipped_services': skipped,
        'disabled_services': disabled,
        'total_items': sum(services_summary.values()),
    })


@app.route('/generate/stream')
def generate_stream():
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'error': 'No hay una sesión activa.'}), 400

    # Either is a real session: the activity (still to be written) or the meta
    # left after the purge (a finished book the reader is reconnecting to).
    if not (_session_data_path(session_id).exists()
            or _meta_path(session_id).exists()):
        return jsonify({'error': 'Sesión no encontrada.'}), 404

    state = _ensure_book_started(session_id)
    if state is None:
        return jsonify({'error': BUSY_MESSAGE, 'error_kind': 'busy'}), 409

    def event_stream():
        yield ': connected\n\n'  # flushes headers immediately, before any content exists
        with _book_lock:
            sent = state['text']
            last_progress = state['progress']
        # Replay current progress + text so a (re)connecting client is in sync.
        if last_progress is not None:
            yield f'data: {json.dumps({"progress": last_progress}, ensure_ascii=False)}\n\n'
        if sent:
            yield f'data: {json.dumps({"content": sent}, ensure_ascii=False)}\n\n'

        while True:
            with _book_lock:
                status = state['status']
                full_text = state['text']
                error = state['error']
                error_kind = state['error_kind']
                progress = state['progress']
            if progress != last_progress:
                last_progress = progress
                if progress is not None:
                    yield f'data: {json.dumps({"progress": progress}, ensure_ascii=False)}\n\n'
            if len(full_text) > len(sent):
                new_part = full_text[len(sent):]
                sent = full_text
                yield f'data: {json.dumps({"content": new_part}, ensure_ascii=False)}\n\n'
            if status in ('done', 'error', 'interrupted'):
                if status == 'error':
                    yield (f'data: {json.dumps({"error": error or "Error desconocido", "error_kind": error_kind}, ensure_ascii=False)}\n\n')
                break
            time.sleep(BOOK_POLL_INTERVAL)
        yield 'event: done\ndata: {}\n\n'

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ---------------------------------------------------------------------------
# Finished-book artifacts (cover + PDF) and the public library.
# ---------------------------------------------------------------------------

LIBRARY_PATH = OUTPUT_FOLDER / 'library.json'
_library_lock = threading.Lock()


def _cover_path(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_cover.png'


def _pdf_path(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_book.pdf'


def _load_library() -> list[dict]:
    if not LIBRARY_PATH.exists():
        return []
    return json.loads(LIBRARY_PATH.read_text(encoding='utf-8'))


def _save_library(entries: list[dict]) -> None:
    LIBRARY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def _session_user_name(session_id: str) -> str:
    """The reader's name, from the meta record if the activity is already gone.

    Meta first: after a book is finished its activity is deleted, and the name
    is the one thing the cover and the library still need from it.
    """
    meta = _meta_path(session_id)
    if meta.exists():
        return json.loads(meta.read_text(encoding='utf-8'))['user_name']
    data = json.loads(_session_data_path(session_id).read_text(encoding='utf-8'))
    return get_user_name(data)


def _ensure_book_artifacts(session_id: str, user_name: str) -> tuple[Path, Path]:
    """Build (once) the cover PNG and PDF for a finished session."""
    cover_path = _cover_path(session_id)
    pdf_path = _pdf_path(session_id)
    if not cover_path.exists():
        # The art is drawn during generation; it's missing only if the drawing
        # server was down, in which case the cover keeps its blank template.
        art = art_path(session_id)
        book_builder.build_cover_image(
            user_name, cover_path,
            background_path=art if art.exists() else None)
    if not pdf_path.exists():
        state = _load_book_state(session_id)
        with _book_lock:
            book_text = state['text']
        book_builder.build_pdf(user_name, book_text, cover_path, pdf_path)
    return cover_path, pdf_path


def _require_finished_session():
    """Shared guard for the download/add-to-library routes.

    Returns (session_id, user_name, None) on success, or
    (None, None, <flask error response>) on failure.
    """
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return None, None, (jsonify({'error': 'No hay una sesión activa.'}), 400)
    # A finished session has only its meta left — and finished is exactly the
    # state these routes exist for, so meta alone is enough to proceed.
    if not (_session_data_path(session_id).exists()
            or _meta_path(session_id).exists()):
        return None, None, (jsonify({'error': 'Sesión no encontrada.'}), 404)
    state = _load_book_state(session_id)
    if state['status'] != 'done':
        return None, None, (jsonify({'error': 'El libro todavía se está escribiendo.'}), 409)
    return session_id, _session_user_name(session_id), None


@app.route('/book/download')
def book_download():
    session_id, user_name, error = _require_finished_session()
    if error:
        return error

    _, pdf_path = _ensure_book_artifacts(session_id, user_name)
    return send_file(
        pdf_path, mimetype='application/pdf', as_attachment=True,
        download_name=f'Harry Potter y {user_name}.pdf',
    )


@app.route('/library/add', methods=['POST'])
def library_add():
    session_id, user_name, error = _require_finished_session()
    if error:
        return error

    _ensure_book_artifacts(session_id, user_name)

    with _library_lock:
        entries = _load_library()
        if not any(e['id'] == session_id for e in entries):
            entries.append({
                'id': session_id,
                'title': f'Harry Potter y {user_name}',
                'user_name': user_name,
                'created_at': datetime.now(timezone.utc).isoformat(),
            })
            _save_library(entries)

    return jsonify({'ok': True})


def _delete_session(session_id: str) -> None:
    """Erase a session completely: activity, mirrors, meta, and the book.

    The cover and PDF are the one exception, and only when the reader put the
    book in the library: those files *are* the library's copy — library.json
    points straight at them — so deleting them would blank out a shelf the
    reader deliberately kept. Everything else goes either way.
    """
    with _book_lock:
        _book_sessions.pop(session_id, None)

    for path in (
        _session_data_path(session_id),
        _meta_path(session_id),
        *_planning_paths(session_id),
        _book_file(session_id),
        _book_done_marker(session_id),
    ):
        path.unlink(missing_ok=True)

    if not any(e['id'] == session_id for e in _load_library()):
        _cover_path(session_id).unlink(missing_ok=True)
        _pdf_path(session_id).unlink(missing_ok=True)


@app.route('/session/reset', methods=['POST'])
def session_reset():
    """Clear the current book session so the browser can start a new adventure.

    This is the reader asking for their data to be gone now, rather than at the
    sweep — so it erases the session outright. Anything already added to the
    library (cover/PDF + its library.json entry) is what they chose to keep.
    """
    session_id = request.cookies.get(BOOK_COOKIE)
    if session_id:
        state = _load_book_state(session_id)
        if state['status'] == 'running':
            return jsonify({'error': 'El libro todavía se está escribiendo.'}), 409
        _delete_session(session_id)

    resp = jsonify({'ok': True})
    resp.delete_cookie(BOOK_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# The sweeper: what catches everything the paths above don't.
#
# A reader who uploads a Takeout and closes the tab never reaches the purge on
# success, never presses "nueva aventura", and leaves their activity sitting on
# disk. Same for a book that failed and was never retried — that data is kept
# on purpose so the retry can work, which only makes a deadline more necessary.
# ---------------------------------------------------------------------------

# Session ids are uuid4. Matching the shape, rather than sweeping everything in
# the folder, keeps the sweeper to files this app created: an operator's own
# notes or a zip dropped in by hand are not ours to delete.
_SESSION_ID_RE = re.compile(
    r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')


def _sessions_on_disk() -> dict:
    """session_id -> when any of its files was last touched."""
    sessions = {}
    for folder in (OUTPUT_FOLDER, UPLOAD_FOLDER):
        for path in folder.iterdir():
            match = _SESSION_ID_RE.match(path.name)
            if not path.is_file() or not match:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue  # vanished under us; nothing left to sweep
            sessions[match.group(1)] = max(sessions.get(match.group(1), 0), mtime)
    return sessions


def _sweep_sessions() -> int:
    """Delete every session that has gone cold. Returns how many."""
    cutoff = time.time() - SESSION_TTL
    swept = 0
    for session_id, touched in _sessions_on_disk().items():
        if touched > cutoff:
            continue
        # A book being written keeps touching its files, so a live session
        # can't look cold — but check anyway rather than rely on that, because
        # the whole prep phase writes nothing at all for minutes at a time.
        with _book_lock:
            state = _book_sessions.get(session_id)
            busy = (session_id == _active_generation
                    or (state is not None and state['status'] == 'running'))
        if busy:
            continue
        _delete_session(session_id)
        # The upload is normally deleted the moment it's parsed; one surviving
        # here means the process died mid-request, and it's the rawest copy of
        # everything the reader gave us.
        for leftover in UPLOAD_FOLDER.glob(f'{session_id}_*'):
            leftover.unlink(missing_ok=True)
        swept += 1
    return swept


def _start_sweeper() -> None:
    """Run the sweep now and every SWEEP_INTERVAL after.

    Started on import, so it covers gunicorn too, not just `python app.py`. The
    sweep on startup is the point of "now": a process that died holding a book
    lost its threads but not its files.
    """
    def loop():
        while True:
            try:
                swept = _sweep_sessions()
                if swept:
                    app.logger.info('sweeper: deleted %d expired session(s)', swept)
            except Exception:
                # Never let a bad file take the sweeper down with it — it would
                # stop deleting everyone's data, silently, until a restart.
                app.logger.exception('session sweep failed')
            time.sleep(SWEEP_INTERVAL)

    threading.Thread(target=loop, daemon=True, name='session-sweeper').start()


_start_sweeper()


@app.route('/library/<session_id>/cover')
def library_cover(session_id):
    cover_path = _cover_path(session_id)
    if not cover_path.exists():
        return jsonify({'error': 'No encontrado.'}), 404
    return send_file(cover_path, mimetype='image/png')


@app.route('/library/<session_id>/pdf')
def library_pdf(session_id):
    pdf_path = _pdf_path(session_id)
    if not pdf_path.exists():
        return jsonify({'error': 'No encontrado.'}), 404
    entries = _load_library()
    entry = next((e for e in entries if e['id'] == session_id), None)
    title = entry['title'] if entry else 'libro'
    return send_file(
        pdf_path, mimetype='application/pdf', as_attachment=True,
        download_name=f'{title}.pdf',
    )


if __name__ == '__main__':
    # Local development only. In production the app is served by a WSGI server
    # (gunicorn) importing the `app` object above — this block never runs there,
    # so `livereload` stays a dev-only dependency.
    port = int(os.environ.get('PORT', 5000))

    if os.environ.get('LIVERELOAD'):
        # Opt-in, and only for style work: livereload serves the app through
        # Tornado's WSGIContainer, which drains the whole response generator and
        # joins it into one body before writing a single byte. /generate/stream
        # therefore delivers the entire book in one lump the moment it finishes,
        # with no live progress and no prose appearing as it's written. The
        # writing area cannot be tested under it — use the default server below.
        from livereload import Server

        server = Server(app.wsgi_app)
        server.watch('static/**/*.less')
        server.watch('static/**/*.js')
        server.watch('templates/*.html')
        server.serve(debug=True, port=port)
    else:
        # threaded: a book holds its SSE thread for ~10 minutes, and the page
        # still has to be able to ask /engine/status while that runs.
        # use_reloader=False: a stray save would otherwise restart the process
        # and kill the generation thread mid-chapter.
        app.run(debug=True, port=port, threaded=True, use_reloader=False)
