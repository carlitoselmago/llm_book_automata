"""Turns a raw streamed book (markdown-ish text) into the final artifacts:
a cover PNG (template + the user's name) and an A5 PDF (cover, intro,
index, chapters with page numbers).
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
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
COVER_NAME_Y = 320
COVER_NAME_FONT_SIZE = 130
COVER_NAME_FONT_PATH = str(FONT_DIR / 'LiberationSans-Bold.ttf')
COVER_NAME_COLOR = (0, 0, 0)

PAGE_SIZE = A5
MARGIN = 18 * mm


def build_cover_image(user_name: str, dest_path: Path) -> Path:
    """Overlay the centered [ME] name onto the fixed cover template."""
    # The template's "blank" area is fully transparent (alpha=0), not white —
    # a plain .convert('RGB') drops the alpha and leaves black. Composite
    # over a white background instead.
    template = Image.open(COVER_TEMPLATE_PATH).convert('RGBA')
    cover = Image.new('RGB', template.size, (255, 255, 255))
    cover.paste(template, mask=template.split()[3])
    draw = ImageDraw.Draw(cover)
    font = ImageFont.truetype(COVER_NAME_FONT_PATH, COVER_NAME_FONT_SIZE)
    x = COVER_NAME_X if COVER_NAME_X is not None else cover.width // 2
    # anchor="ma": horizontally centered, Y is the top of the text.
    draw.text((x, COVER_NAME_Y), user_name, font=font, fill=COVER_NAME_COLOR, anchor='ma')
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

    intro_text = INTRO_TXT_PATH.read_text(encoding='utf-8') if INTRO_TXT_PATH.exists() else ''
    for para in intro_text.split('\n\n'):
        para = para.strip()
        if para:
            story.append(Paragraph(_format_inline(para), styles['Body']))
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
