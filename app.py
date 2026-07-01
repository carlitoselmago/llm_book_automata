import os
import re
import uuid
import zipfile
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

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
def index():
    return render_template('index.html', step=1)


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

    return jsonify({
        'session_id': session_id,
        'services': services_summary,
        'skipped_services': data['skipped_services'],
        'total_items': sum(services_summary.values()),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
