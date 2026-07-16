"""Turns a raw streamed book (markdown-ish text) into the final artifacts:
a cover PNG (template + the user's name) and an A5 PDF (cover, intro,
index, chapters with page numbers).
"""
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A5
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image as RLImage, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / 'book_template'
COVER_TEMPLATE_PATH = TEMPLATE_DIR / 'cover_template.png'
INTRO_TXT_PATH = TEMPLATE_DIR / 'intro.txt'

FONT_DIR = Path('/usr/share/fonts/truetype/liberation2')

# ---------------------------------------------------------------------------
# Cover text placement — tune these to line up with cover_template.png.
# (X, Y) is measured in pixels from the top-left of the template image;
# X=None keeps the name horizontally centered on the page.
# ---------------------------------------------------------------------------
COVER_NAME_X = None
COVER_NAME_Y = 230
COVER_NAME_FONT_SIZE = 140
COVER_NAME_FONT_PATH = str(TEMPLATE_DIR / 'font.ttf')
COVER_NAME_COLOR = (255, 255, 255)
# Drop shadow behind the name. OFFSET is (dx, dy) in pixels, BLUR is the
# gaussian radius, and the shadow colour's alpha sets how strong it reads.
COVER_NAME_SHADOW_COLOR = (0, 0, 0, 130)
COVER_NAME_SHADOW_OFFSET = (0, 5)
COVER_NAME_SHADOW_BLUR = 1

# The author line, sat inside the green box on the template (which spans
# y=429..554). Same font and shadow as the name, just smaller.
COVER_AUTHOR_X = 822
COVER_AUTHOR_Y = 455
COVER_AUTHOR_FONT_SIZE = 70
COVER_AUTHOR_COLOR = (255, 255, 255)

# The window the cover art shows through: full width, this tall, flush to the
# bottom of the template. Everything above it is the opaque title band, so art
# drawn up there is simply lost — including anything the model put near the top
# of its frame.
COVER_ART_WIDTH = 1748
COVER_ART_HEIGHT = 2008

PAGE_SIZE = A5
MARGIN = 18 * mm

fake_names = [
    "A.K. Rowling",
    "B.K. Rowling",
    "C.K. Rowling",
    "D.K. Rowling",
    "E.K. Rowling",
    "F.K. Rowling",
    "G.K. Rowling",
    "H.K. Rowling",
    "I.K. Rowling",
    "K.K. Rowling",
    "L.K. Rowling",
    "M.K. Rowling",
    "N.K. Rowling",
    "O.K. Rowling",
    "P.K. Rowling",
    "Q.K. Rowling",
    "R.K. Rowling",
    "S.K. Rowling",
    "T.K. Rowling",
    "U.K. Rowling",
    "V.K. Rowling",
    "W.K. Rowling",
    "X.K. Rowling",
    "Y.K. Rowling",
    "Z.K. Rowling",
]


def _draw_cover_text(cover: Image.Image, text: str, x, y: int, size: int,
                     color) -> Image.Image:
    """Draw one centered line, over its drop shadow. Returns the new image.

    `x=None` centres on the page. Compositing the shadow replaces the image
    rather than mutating it, hence the return.
    """
    font = ImageFont.truetype(COVER_NAME_FONT_PATH, size)
    if x is None:
        x = cover.width // 2

    # The shadow goes on its own layer: blurring it in place would smear the
    # cover art under it too. anchor="ma": horizontally centered, Y is the top
    # of the text — same for both passes, so they line up.
    dx, dy = COVER_NAME_SHADOW_OFFSET
    shadow = Image.new('RGBA', cover.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((x + dx, y + dy), text, font=font,
                                fill=COVER_NAME_SHADOW_COLOR, anchor='ma')
    shadow = shadow.filter(ImageFilter.GaussianBlur(COVER_NAME_SHADOW_BLUR))
    cover = Image.alpha_composite(cover.convert('RGBA'), shadow).convert('RGB')

    ImageDraw.Draw(cover).text((x, y), text, font=font, fill=color, anchor='ma')
    return cover


def _fill(img: Image.Image, size) -> Image.Image:
    """Scale-and-center-crop `img` so it covers `size` exactly, undistorted.

    The generated art is square-ish and the cover is A5-tall, so something has
    to give: cropping the overflow keeps the art's proportions, stretching
    wouldn't.
    """
    target_w, target_h = size
    scale = max(target_w / img.width, target_h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)),
                     Image.LANCZOS)
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def cover_art_size(width: int = 512) -> tuple[int, int]:
    """A generation size shaped like the art window, for the image server.

    Small on purpose: the art is upscaled to fill, so generating it at the
    window's true 1748px width would only cost time. Matching the *shape* is
    what matters — it's what stops the subject being cropped away. The height
    is rounded to a multiple of 8, which diffusers requires.
    """
    height = round(width * COVER_ART_HEIGHT / COVER_ART_WIDTH / 8) * 8
    return width, height


def build_cover_image(user_name: str, dest_path: Path,
                      author_name: str = None,
                      background_path: Path = None) -> Path:
    """Overlay the centered [ME] name and an author line onto the cover template.

    `author_name` defaults to a random entry from `fake_names` — pass one
    explicitly to pin it (a caller that rebuilds a cover and wants the same
    author as last time, a test).

    `background_path` is artwork to sit under the template, showing through its
    transparent middle. Without one the blank stays white.
    """
    if author_name is None:
        author_name = random.choice(fake_names)

    # The template's "blank" area is fully transparent (alpha=0), not white —
    # a plain .convert('RGB') drops the alpha and leaves black. Composite it
    # over the artwork (or plain white) instead.
    template = Image.open(COVER_TEMPLATE_PATH).convert('RGBA')
    cover = Image.new('RGB', template.size, (255, 255, 255))
    if background_path is not None:
        # Fit the art to the window alone, not the whole page: filling the page
        # pushes the middle of the art up behind the title band, which is where
        # a subject usually sits.
        art = _fill(Image.open(background_path).convert('RGB'),
                    (COVER_ART_WIDTH, COVER_ART_HEIGHT))
        cover.paste(art, (0, cover.height - COVER_ART_HEIGHT))
    cover.paste(template, mask=template.split()[3])

    cover = _draw_cover_text(cover, user_name, COVER_NAME_X, COVER_NAME_Y,
                             COVER_NAME_FONT_SIZE, COVER_NAME_COLOR)
    cover = _draw_cover_text(cover, author_name, COVER_AUTHOR_X, COVER_AUTHOR_Y,
                             COVER_AUTHOR_FONT_SIZE, COVER_AUTHOR_COLOR)
    cover.save(dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Parse the streamed markdown-ish text into chapters, mirroring the same
# "## heading" / blank-line-ends-paragraph logic used by the live JS index
# in templates/index.html, so the PDF matches what the user watched being
# written.
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r'^#{1,6}\s*(.+)$')


def parse_chapters(text: str) -> list[dict]:
    chapters = []
    preamble: list[str] = []
    current = None
    para_lines: list[str] = []

    def flush(target: list[str]):
        if para_lines:
            target.append(' '.join(para_lines).strip())
            para_lines.clear()

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        target = current['paragraphs'] if current is not None else preamble
        if heading:
            flush(target)
            current = {'title': heading.group(1).strip(), 'paragraphs': []}
            chapters.append(current)
        elif line == '':
            flush(target)
        else:
            para_lines.append(line)

    flush(current['paragraphs'] if current is not None else preamble)

    if preamble:
        chapters.insert(0, {'title': None, 'paragraphs': preamble})

    return chapters


def _escape(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _format_inline(text: str) -> str:
    """Escape for reportlab markup, then map markdown **bold**/*italic*."""
    text = _escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    return text


# --- intro.txt rendering ---------------------------------------------------
# The intro is hand-written markdown (unlike the chapters, which are model
# output), so it gets a renderer that honours what it actually uses: headings,
# a <sub> block, and paragraphs whose line breaks are meant to be kept.

_INTRO_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
_SUB_OPEN_RE = re.compile(r'<sub>', re.IGNORECASE)
_SUB_CLOSE_RE = re.compile(r'</sub>', re.IGNORECASE)
# Zero-width spaces ride along in text pasted from rich editors, and Liberation
# Serif has no glyph for them — they'd surface as tofu boxes mid-sentence.
_ZERO_WIDTH_RE = re.compile(r'[​‌‍﻿]')

_INTRO_HEADING_SIZES = {1: 18, 2: 15, 3: 13}


def _intro_flowables(styles: dict) -> list:
    """Render book_template/intro.txt's markdown into flowables."""
    if not INTRO_TXT_PATH.exists():
        return []
    text = _ZERO_WIDTH_RE.sub('', INTRO_TXT_PATH.read_text(encoding='utf-8'))

    flowables: list = []
    para: list[str] = []
    small = False

    def flush():
        if para:
            # Single newlines become <br/> rather than spaces: the disclaimer's
            # clauses are separate lines on purpose, and joining them into one
            # blob is exactly what this is fixing.
            flowables.append(Paragraph(
                '<br/>'.join(_format_inline(line) for line in para),
                styles['IntroSmall'] if small else styles['Body']))
            para.clear()

    for raw_line in text.split('\n'):
        line = raw_line.strip()

        # <sub>…</sub> isn't markdown, it's the inline HTML markdown allows, and
        # here it means small print. It can't be passed through to reportlab,
        # whose own <sub> is subscript — that would set the whole disclaimer as
        # tiny raised text.
        if _SUB_OPEN_RE.search(line):
            flush()
            small = True
            line = _SUB_OPEN_RE.sub('', line).strip()
        closes = bool(_SUB_CLOSE_RE.search(line))
        if closes:
            line = _SUB_CLOSE_RE.sub('', line).strip()

        heading = _INTRO_HEADING_RE.match(line)
        if not line:
            flush()
        elif heading:
            flush()
            level = len(heading.group(1))
            style = styles['IntroHeading']
            size = _INTRO_HEADING_SIZES.get(level, 12)
            if size != style.fontSize:
                style = ParagraphStyle(f'IntroHeading{level}', parent=style,
                                       fontSize=size, leading=round(size * 1.35))
            flowables.append(Paragraph(_format_inline(heading.group(2)), style))
        else:
            para.append(line)

        if closes:
            flush()
            small = False

    flush()
    return flowables


_FONTS_REGISTERED = False


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    pdfmetrics.registerFont(TTFont('LiberationSerif', str(FONT_DIR / 'LiberationSerif-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('LiberationSerif-Bold', str(FONT_DIR / 'LiberationSerif-Bold.ttf')))
    pdfmetrics.registerFont(TTFont('LiberationSerif-Italic', str(FONT_DIR / 'LiberationSerif-Italic.ttf')))
    pdfmetrics.registerFont(TTFont('LiberationSerif-BoldItalic', str(FONT_DIR / 'LiberationSerif-BoldItalic.ttf')))
    pdfmetrics.registerFontFamily(
        'LiberationSerif',
        normal='LiberationSerif', bold='LiberationSerif-Bold',
        italic='LiberationSerif-Italic', boldItalic='LiberationSerif-BoldItalic',
    )
    _FONTS_REGISTERED = True


def _book_styles() -> dict:
    return {
        'Body': ParagraphStyle(
            'Body', fontName='LiberationSerif', fontSize=11, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        'ChapterTitle': ParagraphStyle(
            'ChapterTitle', fontName='LiberationSerif-Bold', fontSize=16,
            leading=20, alignment=TA_CENTER, spaceBefore=6, spaceAfter=18,
        ),
        'TOCHeading': ParagraphStyle(
            'TOCHeading', fontName='LiberationSerif-Bold', fontSize=18,
            alignment=TA_CENTER, spaceAfter=18,
        ),
        'TOCEntry': ParagraphStyle(
            'TOCEntry', fontName='LiberationSerif', fontSize=11, leading=16,
        ),
        # Intro headings are deliberately NOT called 'ChapterTitle': that name is
        # what afterFlowable watches to build the index, and the intro's heading
        # is an epigraph, not a chapter.
        'IntroHeading': ParagraphStyle(
            'IntroHeading', fontName='LiberationSerif-Bold', fontSize=13,
            leading=18, alignment=TA_CENTER, spaceBefore=6, spaceAfter=16,
        ),
        # The <sub>…</sub> small print.
        'IntroSmall': ParagraphStyle(
            'IntroSmall', fontName='LiberationSerif', fontSize=7.5, leading=10,
            alignment=TA_JUSTIFY, spaceAfter=5,
        ),
    }


class _BookDocTemplate(BaseDocTemplate):
    """Feeds every chapter-title paragraph into the TOC as it's laid out."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'ChapterTitle':
            self.notify('TOCEntry', (0, flowable.getPlainText(), self.page))


def _draw_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('LiberationSerif', 9)
    canvas.drawCentredString(PAGE_SIZE[0] / 2, 12 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def build_pdf(user_name: str, book_text: str, cover_path: Path, dest_path: Path) -> Path:
    _register_fonts()
    styles = _book_styles()
    chapters = parse_chapters(book_text)

    doc = _BookDocTemplate(
        str(dest_path), pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title=f'Harry Potter y {user_name}',
    )

    cover_frame = Frame(0, 0, PAGE_SIZE[0], PAGE_SIZE[1],
                         leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                         id='cover')
    content_frame = Frame(MARGIN, MARGIN, PAGE_SIZE[0] - 2 * MARGIN, PAGE_SIZE[1] - 2 * MARGIN,
                           id='content')

    doc.addPageTemplates([
        PageTemplate(id='cover', frames=[cover_frame]),
        PageTemplate(id='content', frames=[content_frame], onPage=_draw_page_number),
    ])

    story = [
        NextPageTemplate('content'),
        RLImage(str(cover_path), width=PAGE_SIZE[0], height=PAGE_SIZE[1]),
        PageBreak(),
    ]

    story.extend(_intro_flowables(styles))
    story.append(PageBreak())

    story.append(Paragraph('Índice', styles['TOCHeading']))
    toc = TableOfContents()
    toc.levelStyles = [styles['TOCEntry']]
    story.append(toc)
    story.append(PageBreak())

    for i, chapter in enumerate(chapters):
        if chapter['title']:
            story.append(Paragraph(_format_inline(chapter['title']), styles['ChapterTitle']))
        for para in chapter['paragraphs']:
            if para:
                story.append(Paragraph(_format_inline(para), styles['Body']))
        if i < len(chapters) - 1:
            story.append(PageBreak())

    doc.multiBuild(story)
    return dest_path
