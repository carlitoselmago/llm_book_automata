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
from livereload import Server

import book_builder
from generate_book import build_user_description, iter_book_chunks

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max



BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'outputs'
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


MAX_ITEMS_PER_SERVICE = 10

# Google services to ignore — contain no useful personal narrative data
BLACKLISTED_SERVICES = {
    'Ads',
    'Assistant',
    'Android',
    'Discover',
    'Drive',
    'Google Translate',
    'Hotels',
    'Google Lens',
    'Google Play Movies & TV',
    'Voice Match',
    'Developers',
    'Google Play Console',
    'Google Business Profile',
    'Takeout',
    'Help',
    'Android TV',
    'Google TV',
    'Google Arts & Culture',
    'Google News',
    'Google Play Games'
}


# ---------------------------------------------------------------------------
# Per-service filters
# Each function receives a list[dict] and returns a filtered list[dict].
# ---------------------------------------------------------------------------

def _filter_books(items: list[dict]) -> list[dict]:
    result = []
    for item in items:
        title = item.get('title') or ''
        if title and 'http' not in title:
            result.append(item)
    return result

def _filter_news(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get('title')]

def _filter_youtube(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get('title')]


def _filter_maps(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get('title')]


SERVICE_FILTERS: dict[str, callable] = {
    'Books': _filter_books,
    'YouTube': _filter_youtube,
    'Maps': _filter_maps,
    'Google News': _filter_news,
    'Search': _filter_news
}


# ---------------------------------------------------------------------------
# Per-service custom parsers
# These replace parse_activity_html entirely for services whose HTML structure
# doesn't fit the generic outer-cell pattern.
# Each receives (html_bytes, max_items) and returns list[dict].
# ---------------------------------------------------------------------------

_RE_TIMESTAMP_LINE = re.compile(
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4},'
)


def _parse_gemini(html_bytes: bytes, max_items: int) -> list[dict]:
    """Keep only user prompts (cells starting with 'Prompted').

    Each outer-cell is either a user prompt or a Gemini response.
    We only keep the user side and capture: prompt, timestamp, answer.
    The answer is the content of the very next outer-cell (Gemini's reply).
    """
    # Collect all outer-cell text blocks first
    blocks = []
    for m in _RE_OUTER_CELL.finditer(html_bytes):
        cc = _RE_CONTENT_CELL.search(m.group(0))
        if not cc:
            continue
        text = _strip_html(cc.group(1))
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        blocks.append(lines)

    items = []
    i = 0
    while i < len(blocks) and len(items) < max_items:
        lines = blocks[i]
        first = lines[0] if lines else ''

        if first.startswith('Prompted '):
            prompt = first[len('Prompted '):].strip()
            timestamp = next(
                (l for l in lines if _RE_TIMESTAMP_LINE.search(l)), None
            )
            # Grab the next block as the answer if it isn't another prompt
            answer = None
            if i + 1 < len(blocks):
                next_lines = blocks[i + 1]
                next_first = next_lines[0] if next_lines else ''
                if not next_first.startswith('Prompted '):
                    answer = ' '.join(
                        l for l in next_lines
                        if not _RE_TIMESTAMP_LINE.search(l)
                    ) or None

            items.append({'prompt': prompt, 'timestamp': timestamp, 'answer': answer})

        i += 1

    return items


def _parse_google_play_games(html_bytes: bytes, max_items: int) -> list[dict]:
    """Extract game title and timestamp from Play Games activity.

    Each cell looks like:  'Played Fallout Shelter\\nSep 30, 2025, ...'
    We return: game (str), timestamp (str).
    """
    items = []
    for m in _RE_OUTER_CELL.finditer(html_bytes):
        if len(items) >= max_items:
            break
        cc = _RE_CONTENT_CELL.search(m.group(0))
        if not cc:
            continue
        text = _strip_html(cc.group(1))
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        game = None
        timestamp = None
        for line in lines:
            if _RE_TIMESTAMP_LINE.search(line):
                timestamp = line
            elif line.startswith('Played '):
                game = line[len('Played '):].strip()

        if game:
            items.append({'game': game, 'timestamp': timestamp})

    return items


_RE_PRE_ANCHOR = re.compile(rb'^(.*?)<a\s', re.DOTALL)


def _parse_google_play_store(html_bytes: bytes, max_items: int) -> list[dict]:
    """Extract action + app name + timestamp from Play Store activity.

    Cells look like: 'Used <a href="...">App Name</a>\\nTimestamp'
    We return: action (str), app (str), timestamp (str).
    """
    items = []
    for m in _RE_OUTER_CELL.finditer(html_bytes):
        if len(items) >= max_items:
            break
        cc = _RE_CONTENT_CELL.search(m.group(0))
        if not cc:
            continue
        inner = cc.group(1)

        anchor = _RE_ANCHOR.search(inner)
        if not anchor:
            continue

        app = _strip_html(anchor.group(2))
        if not app or app == 'Google Play Store':
            continue

        # Text before the anchor tag is the action verb
        pre = _RE_PRE_ANCHOR.match(inner)
        action = _strip_html(pre.group(1)).rstrip('\xa0 ') if pre else None

        plain = _strip_html(_RE_ANCHOR.sub(b'', inner))
        lines = [l.strip() for l in plain.split('\n') if l.strip()]
        timestamp = next(
            (l for l in lines if _RE_TIMESTAMP_LINE.search(l)), None
        )

        items.append({'action': action or None, 'app': app, 'timestamp': timestamp})

    return items


def _parse_image_search(html_bytes: bytes, max_items: int) -> list[dict]:
    """Extract action + content + timestamp from Image Search activity.

    Search cells:  'Searched for <a href="google.com/search?q=...">query</a>'
    Visited cells: 'Visited <a href="google.com/url?q=https://...">domain</a>'
    We return: action (str), content (str), timestamp (str).
    """
    items = []
    for m in _RE_OUTER_CELL.finditer(html_bytes):
        if len(items) >= max_items:
            break
        cc = _RE_CONTENT_CELL.search(m.group(0))
        if not cc:
            continue
        inner = cc.group(1)

        anchor = _RE_ANCHOR.search(inner)
        if not anchor:
            continue

        content = _strip_html(anchor.group(2))
        if not content:
            continue

        pre = _RE_PRE_ANCHOR.match(inner)
        action = _strip_html(pre.group(1)).rstrip('\xa0 ') if pre else None

        plain = _strip_html(_RE_ANCHOR.sub(b'', inner))
        lines = [l.strip() for l in plain.split('\n') if l.strip()]
        timestamp = next(
            (l for l in lines if _RE_TIMESTAMP_LINE.search(l)), None
        )

        items.append({'action': action or None, 'content': content, 'timestamp': timestamp})

    return items


SERVICE_PARSERS: dict[str, callable] = {
    'Gemini Apps': _parse_gemini,
    'Google Play Games': _parse_google_play_games,
    'Google Play Store': _parse_google_play_store,
    'Image Search': _parse_image_search
}


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


def _strip_html(raw: bytes) -> str:
    text = _RE_BR.sub(b'\n', raw)
    text = _RE_TAG.sub(b'', text)
    text = _RE_HTML_ENTITY.sub(lambda m: _RE_ENTITY[m.group(0)], text)
    return text.decode('utf-8', errors='replace').strip()


def parse_activity_html(html_bytes: bytes, max_items: int) -> list[dict]:
    """Extract activity items from a Google MyActivity HTML file using regex.

    Uses regex instead of a DOM parser so it stays fast on 100MB+ files.
    """
    items = []
    for m in _RE_OUTER_CELL.finditer(html_bytes):
        if len(items) >= max_items:
            break

        cell_bytes = m.group(0)
        cc = _RE_CONTENT_CELL.search(cell_bytes)
        if not cc:
            continue
        inner = cc.group(1)

        anchor = _RE_ANCHOR.search(inner)
        url = anchor.group(1).decode('utf-8', errors='replace') if anchor else None
        title = _strip_html(anchor.group(2)) if anchor else None

        # Timestamp: last text segment after stripping tags
        plain = _strip_html(_RE_ANCHOR.sub(b'', inner))
        # The timestamp is usually the last non-empty line
        lines = [l.strip() for l in plain.split('\n') if l.strip()]
        timestamp = lines[-1] if lines else None

        if title or timestamp:
            items.append({'title': title, 'url': url, 'timestamp': timestamp})

    return items


_STRIP_FIELDS = {'url', 'timestamp'}


def _clean_items(items: list[dict]) -> list[dict]:
    """Remove url/timestamp from every item; drop items with no content left."""
    result = []
    for item in items:
        cleaned = {k: v for k, v in item.items()
                   if k not in _STRIP_FIELDS and v is not None and v != ''}
        # Drop items that have only URL-valued fields as their sole content
        values = list(cleaned.values())
        if values and not all(str(v).startswith('http') for v in values):
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
    """Extract zip and parse profile + each non-blacklisted MyActivity.html."""
    result = {
        'session_id': session_id,
        'profile': {},
        'services': {},
        'skipped_services': [],
    }

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()

        # Profile — Takeout/Profile/Profile.json
        profile_path = next(
            (n for n in names if n.endswith('Profile/Profile.json')), None
        )
        if profile_path:
            result['profile'] = parse_profile_json(zf.read(profile_path))

        # Collect service paths: Takeout/My Activity/<Service>/MyActivity.html
        service_files = {}
        for name in names:
            parts = name.split('/')
            if (len(parts) >= 4
                    and parts[1] == 'My Activity'
                    and parts[-1] == 'MyActivity.html'):
                service_files[parts[2]] = name

        for service_name, file_path in sorted(service_files.items()):
            if service_name in BLACKLISTED_SERVICES:
                result['skipped_services'].append(service_name)
                continue

            html_bytes = zf.read(file_path)
            if service_name in SERVICE_PARSERS:
                items = SERVICE_PARSERS[service_name](html_bytes, MAX_ITEMS_PER_SERVICE)
            else:
                items = parse_activity_html(html_bytes, MAX_ITEMS_PER_SERVICE)
                if service_name in SERVICE_FILTERS:
                    items = SERVICE_FILTERS[service_name](items)
            items = _clean_items(items)
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


def _book_file(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_book.txt'


def _book_done_marker(session_id: str) -> Path:
    return OUTPUT_FOLDER / f'{session_id}_book.done'


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
            state = {'status': status, 'text': text, 'error': None}
        else:
            state = {'status': 'none', 'text': '', 'error': None}
        _book_sessions[session_id] = state
        return state


def _append_book_text(session_id: str, state: dict, chunk: str) -> None:
    with _book_lock:
        state['text'] += chunk
    with open(_book_file(session_id), 'a', encoding='utf-8') as f:
        f.write(chunk)


def _run_book_generation(session_id: str, state: dict, user_description: str, user_name: str) -> None:
    try:
        for chunk in iter_book_chunks(user_description, user_name):
            _append_book_text(session_id, state, chunk)
        with _book_lock:
            state['status'] = 'done'
        _book_done_marker(session_id).touch()
    except Exception as e:
        app.logger.exception('Book generation failed for session %s', session_id)
        with _book_lock:
            state['status'] = 'error'
            state['error'] = str(e)


def _ensure_book_started(session_id: str) -> dict:
    """Return the session's state, starting the background writer if needed."""
    state = _load_book_state(session_id)
    with _book_lock:
        should_start = state['status'] == 'none'
        if should_start:
            state['status'] = 'running'
    if should_start:
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
        user_name = data.get('profile', {}).get('name') or 'You'
        user_description = build_user_description(data)
        threading.Thread(
            target=_run_book_generation,
            args=(session_id, state, user_description, user_name),
            daemon=True,
        ).start()
    return state


@app.route('/generate/configure', methods=['POST'])
def generate_configure():
    """Let the user disable specific services before generation starts."""
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'error': 'No hay una sesión activa.'}), 400

    output_path = OUTPUT_FOLDER / f'{session_id}.json'
    if not output_path.exists():
        return jsonify({'error': 'Sesión no encontrada.'}), 404

    state = _load_book_state(session_id)
    if state['status'] != 'none':
        return jsonify({'error': 'La generación ya ha comenzado.'}), 409

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


@app.route('/generate/status')
def generate_status():
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'has_session': False})

    output_path = OUTPUT_FOLDER / f'{session_id}.json'
    if not output_path.exists():
        return jsonify({'has_session': False})

    data = json.loads(output_path.read_text(encoding='utf-8'))
    services_summary = {name: info['count'] for name, info in data['services'].items()}
    state = _load_book_state(session_id)

    return jsonify({
        'has_session': True,
        'session_id': session_id,
        'book_status': state['status'],
        'services': services_summary,
        'skipped_services': data.get('skipped_services', []),
        'disabled_services': data.get('disabled_services', []),
        'total_items': sum(services_summary.values()),
    })


@app.route('/generate/stream')
def generate_stream():
    session_id = request.cookies.get(BOOK_COOKIE)
    if not session_id:
        return jsonify({'error': 'No hay una sesión activa.'}), 400

    if not (OUTPUT_FOLDER / f'{session_id}.json').exists():
        return jsonify({'error': 'Sesión no encontrada.'}), 404

    state = _ensure_book_started(session_id)

    def event_stream():
        yield ': connected\n\n'  # flushes headers immediately, before any content exists
        with _book_lock:
            sent = state['text']
        if sent:
            yield f'data: {json.dumps({"content": sent}, ensure_ascii=False)}\n\n'

        while True:
            with _book_lock:
                status = state['status']
                full_text = state['text']
                error = state['error']
            if len(full_text) > len(sent):
                new_part = full_text[len(sent):]
                sent = full_text
                yield f'data: {json.dumps({"content": new_part}, ensure_ascii=False)}\n\n'
            if status in ('done', 'error', 'interrupted'):
                if status == 'error':
                    yield f'data: {json.dumps({"error": error or "Error desconocido"})}\n\n'
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
    data = json.loads((OUTPUT_FOLDER / f'{session_id}.json').read_text(encoding='utf-8'))
    return data.get('profile', {}).get('name') or 'You'


def _ensure_book_artifacts(session_id: str, user_name: str) -> tuple[Path, Path]:
    """Build (once) the cover PNG and PDF for a finished session."""
    cover_path = _cover_path(session_id)
    pdf_path = _pdf_path(session_id)
    if not cover_path.exists():
        book_builder.build_cover_image(user_name, cover_path)
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
    if not (OUTPUT_FOLDER / f'{session_id}.json').exists():
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


@app.route('/session/reset', methods=['POST'])
def session_reset():
    """Clear the current book session so the browser can start a new adventure.

    Only removes what isn't needed anymore: the raw session data plus any
    book artifacts that were never saved to the library. Anything already
    added to the library (cover/PDF + its library.json entry) is untouched.
    """
    session_id = request.cookies.get(BOOK_COOKIE)
    if session_id:
        state = _load_book_state(session_id)
        if state['status'] == 'running':
            return jsonify({'error': 'El libro todavía se está escribiendo.'}), 409

        with _book_lock:
            _book_sessions.pop(session_id, None)

        in_library = any(e['id'] == session_id for e in _load_library())

        for path in (
            OUTPUT_FOLDER / f'{session_id}.json',
            _book_file(session_id),
            _book_done_marker(session_id),
        ):
            path.unlink(missing_ok=True)

        if not in_library:
            _cover_path(session_id).unlink(missing_ok=True)
            _pdf_path(session_id).unlink(missing_ok=True)

    resp = jsonify({'ok': True})
    resp.delete_cookie(BOOK_COOKIE)
    return resp


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
    server = Server(app.wsgi_app)
    server.watch('static/**/*.less')
    server.watch('static/**/*.js')
    server.watch('templates/*-html')
    server.serve(debug=True, port=5000)
    #app.run(debug=True, port=5000, use_reloader=False, threaded=True)
