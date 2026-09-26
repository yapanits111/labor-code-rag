"""Turn the two DOLE PDFs into clean, structured records.

Plain text extraction loses the information that matters here, so this works
from PyMuPDF's font metadata instead:
- superscript spans are footnote markers  -> removed from the text, but the
                                              number is kept to link the note
- rows in a smaller font are footnotes     -> moved out of the body text
- rows are rebuilt from x/y positions      -> table columns stay aligned

Labor Code -> one record per article, with its footnotes attached.
Handbook   -> one record per topic section (e.g. "13 Thirteenth-Month Pay").
"""
import re
from dataclasses import dataclass, field

import pymupdf

ORDINALS = {"st", "nd", "rd", "th"}  # superscripts that belong to the text (13th)
COLUMN_GAP = 12.0  # points of empty space between two spans that mean "new column"


@dataclass
class Row:
    """One visual line of a page."""
    text: str
    size: float                # largest font size in the row
    x0: float
    y: float                   # baseline
    first_size: float          # size of the first span (footnote numbers are tiny)
    bold: bool = False         # every span bold (Handbook subheadings: "A. Definition")
    page: str = ""             # printed page number
    pdf_page: int = 0          # 1-based page in the PDF file, for "open source" links
    new_page: bool = False     # first row of its page
    markers: list[str] = field(default_factory=list)  # footnote markers in the row


def _page_rows(page, skip_rects=()) -> list[Row]:
    """Rebuild visual rows from spans. Superscript markers are dropped from
    the text, and spans inside `skip_rects` (tables) are ignored."""
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                cx = (s["bbox"][0] + s["bbox"][2]) / 2
                cy = (s["bbox"][1] + s["bbox"][3]) / 2
                if any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in skip_rects):
                    continue
                spans.append(s)
    spans.sort(key=lambda s: (s["origin"][1], s["bbox"][0]))

    groups: list[list[dict]] = []
    for s in spans:  # spans on (nearly) the same baseline form one row
        if groups and abs(s["origin"][1] - groups[-1][0]["origin"][1]) <= 3:
            groups[-1].append(s)
        else:
            groups.append([s])

    rows = []
    for group in groups:
        group.sort(key=lambda s: s["bbox"][0])
        text, markers, sizes, bolds, prev_x1 = "", [], [], [], None
        for s in group:
            t = s["text"]
            if s["flags"] & 1 and t.strip() not in ORDINALS:  # superscript marker
                markers.append(t.strip())
                prev_x1 = s["bbox"][2]
                continue
            if prev_x1 is not None:
                gap = s["bbox"][0] - prev_x1
                if gap > COLUMN_GAP:
                    text = text.rstrip() + " | "
                elif not text.endswith(" ") and not t.startswith(" ") and gap > 0.5:
                    text += " "
            text += t
            sizes.append(s["size"])
            bolds.append(bool(s["flags"] & 16))
            prev_x1 = s["bbox"][2]
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            rows.append(Row(text, max(sizes), group[0]["bbox"][0], group[0]["origin"][1],
                            sizes[0], bold=all(bolds), markers=markers))
    return rows


def _label_page(rows: list[Row], page_index: int, max_size: float) -> list[Row]:
    """Remove the printed page number (a digits-only row at the top or bottom)
    and stamp every row with it."""
    label = str(page_index + 1)
    for i in (0, -1):
        if rows and re.fullmatch(r"\d{1,3}", rows[i].text) and rows[i].size <= max_size:
            label = rows.pop(i).text
            break
    for i, r in enumerate(rows):
        r.page, r.pdf_page, r.new_page = label, page_index + 1, i == 0
    return rows


SUBHEADING_RE = re.compile(r"^[A-Z]\. [A-Z]")


def is_subheading(row: Row) -> bool:
    return row.bold and bool(SUBHEADING_RE.match(row.text)) and len(row.text) < 120


def join_rows(rows: list[Row], para_gap: float) -> list[tuple[str, Row]]:
    """Join visual rows into paragraphs, each returned with the row it starts
    on (for its page number). A big vertical jump starts a new paragraph;
    table rows ("a | b") always stand alone; a page break continues the
    paragraph unless the last line ended a sentence."""
    def is_table(row: Row) -> bool:
        return " | " in row.text or row.text.startswith("|")

    paragraphs: list[list] = []
    prev = None
    for r in rows:
        if prev is None:
            new = True
        elif r.new_page:
            new = bool(re.search(r"[.:;]$", paragraphs[-1][0])) or is_table(r)
        else:
            wrapped_heading = is_subheading(prev) and r.bold and not is_table(r)
            new = (is_table(r) or is_table(prev) or r.y - prev.y > para_gap
                   or is_subheading(r) or (is_subheading(prev) and not wrapped_heading))
        if new:
            paragraphs.append([r.text, r])
        elif paragraphs[-1][0].endswith("-") and r.text[:1].islower():
            paragraphs[-1][0] += r.text  # compound split across lines: thirteenth-/month
        else:
            paragraphs[-1][0] += " " + r.text
        prev = r
    return [(text, row) for text, row in paragraphs]


def _page_span(rows: list[Row]) -> str:
    pages = list(dict.fromkeys(r.page for r in rows))
    return pages[0] if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"


# --- Labor Code --------------------------------------------------------------

ART_RE = re.compile(r"^ART\. (\d+)\.\s*(?:\[([^\]]+)\]\s*)?(.*)$", re.S)
# Body headings are title case ("Book Three – ..."), the table of contents is upper case
HEADING_RE = re.compile(r"^(PRELIMINARY TITLE|BOOK [A-Z]+|TITLE [IVXL]+(?:-[A-Z])?|CHAPTER [IVXL]+)\b",
                        re.I)
HEADING_LEVEL = {"PRELIMINARY": "book", "BOOK": "book", "TITLE": "title", "CHAPTER": "chapter"}
LC_NOTE_MAX = 6.5      # footnote text is 5.5pt (body is 9.5pt)
LC_HEADING_MIN = 11.5  # Book/Title/Chapter headings are 12-16pt


def extract_labor_code(path: str) -> list[dict]:
    doc = pymupdf.open(path)
    headings = {"book": None, "title": None, "chapter": None}
    footnotes: dict[str, str] = {}
    articles: list[dict] = []
    current, note_id, last_level = None, None, None

    for page in doc:
        for r in _label_page(_page_rows(page), page.number, max_size=8.5):
            if r.size <= LC_NOTE_MAX:  # footnote: starts with a tiny number
                m = re.match(r"^(\d{1,3}) (.*)$", r.text)
                if m and r.first_size < 4.5:
                    note_id = m.group(1)
                    footnotes[note_id] = m.group(2)
                elif note_id:
                    footnotes[note_id] += " " + r.text
                continue

            if r.size >= LC_HEADING_MIN and not r.text.startswith("ART."):
                m = HEADING_RE.match(r.text)
                if m:
                    last_level = HEADING_LEVEL[m.group(1).split()[0].upper()]
                    headings[last_level] = r.text
                    if last_level == "book":
                        headings["title"] = headings["chapter"] = None
                    elif last_level == "title":
                        headings["chapter"] = None
                elif last_level:  # heading wrapped onto a second line
                    headings[last_level] += " " + r.text
                continue
            last_level = None

            if ART_RE.match(r.text):
                current = {"rows": [], "markers": [],
                           "path": " > ".join(h for h in headings.values() if h)}
                articles.append(current)
            if current is not None:  # skip front matter before Article 1
                current["rows"].append(r)
                current["markers"].extend(r.markers)

    return [_finish_article(a, footnotes) for a in articles]


def _finish_article(article: dict, footnotes: dict[str, str]) -> dict:
    text = "\n".join(p for p, _ in join_rows(article["rows"], para_gap=17))
    number, old_number, rest = ART_RE.match(text).groups()
    # "Right to Holiday Pay. – (a) Every worker..." -> title, body
    m = re.match(r"^(.{3,250}?)\.?\s*–\s*(.*)$", rest, re.S)
    title, body = (m.group(1), m.group(2)) if m else (rest.split(".")[0], rest)
    return {
        "doc": "labor_code",
        "article": number,
        "old_article": old_number,
        "title": title.strip().rstrip("."),
        "path": article["path"],
        "pages": _page_span(article["rows"]),
        "pdf_page": article["rows"][0].pdf_page,
        "text": body.strip(),
        "notes": [footnotes[n] for n in dict.fromkeys(article["markers"]) if n in footnotes],
    }


# --- Handbook ----------------------------------------------------------------

HB_FIRST_PAGE = 10       # 0-based index of the Compliance Guide (printed page 1)
HB_STOP = "DIRECTORY OF DOLE REGIONAL OFFICES"
HB_NOTE_MAX = 10.0       # footnote text is 9.1pt (body is 11.5pt)
HB_HEADING_MIN = 17.0    # topic headings are 18pt


def extract_handbook(path: str) -> list[dict]:
    doc = pymupdf.open(path)
    sections: list[dict] = []
    current, note = None, None

    for page in doc.pages(HB_FIRST_PAGE):
        if HB_STOP in page.get_text():
            break
        tables = page.find_tables().tables
        rows = _page_rows(page, skip_rects=[pymupdf.Rect(t.bbox) for t in tables])
        for t in tables:  # ruled tables: rebuilt as markdown, placed where they sit
            md = clean_table(re.sub(r"<br>", " ", t.to_markdown()))
            rows.append(Row(md, 11.5, t.bbox[0], t.bbox[1], 11.5))
        rows.sort(key=lambda r: r.y)
        rows = _label_page(rows, page.number, max_size=13)

        for r in rows:
            if r.size >= HB_HEADING_MIN:  # "13" / "THIRTEENTH-MONTH PAY" -> one heading
                if current and not current["rows"]:
                    current["section"] += " " + r.text
                else:
                    current = {"section": r.text, "rows": [], "notes": []}
                    sections.append(current)
                continue
            if current is None or r.text in ORDINALS:
                continue
            if r.size < 6.5:  # a footnote's letter ("kk") sits on its own raised row
                current["notes"].append("")
                continue
            if r.size < HB_NOTE_MAX and not r.text.startswith("|"):  # footnote text
                if current["notes"]:
                    current["notes"][-1] = (current["notes"][-1] + " " + r.text).strip()
                else:
                    current["notes"].append(r.text)
                continue
            r.text = re.sub(r"^([A-Z]|\d{1,2})\.(?=[A-Z])", r"\1. ", r.text)
            prev = current["rows"][-1] if current["rows"] else None
            if prev and r.text.startswith("| ") and prev.text.startswith("| "):
                merged = merge_table_continuation(prev.text, r.text)
                if merged:
                    prev.text = merged
                    continue
            if (prev and " | " in prev.text and " | " not in r.text
                    and r.x0 > prev.x0 + 3 and len(r.text) < 60 and not r.new_page):
                # wrapped table label: "…falling on rest" + "day" -> same row
                label, _, value = prev.text.partition(" | ")
                prev.text = f"{label} {r.text} | {value}"
                continue
            current["rows"].append(r)

    return [_finish_section(s) for s in sections if s["rows"]]


SMALL_WORDS = {"and", "or", "of", "for", "the", "a", "an", "to", "in", "on"}
KEEP_CASE = {"PHILHEALTH": "PhilHealth", "PAG-IBIG": "Pag-IBIG", "(VAWC)": "(VAWC)"}


def table_cells(line: str) -> list[str]:
    """"| a |  | c |" -> ["a", "", "c"] (empty cells kept)."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def merge_table_continuation(prev: str, cont: str) -> str | None:
    """A table that runs onto the next page comes back as a second table with
    the same header. Join it to the first; if its first row has an empty first
    cell, that row finishes the previous page's last row."""
    prev_lines, cont_lines = prev.split("\n"), cont.split("\n")
    if prev_lines[0] != cont_lines[0]:
        return None
    rows = cont_lines[2:]
    if rows and table_cells(rows[0])[0] == "":
        last, first = table_cells(prev_lines[-1]), table_cells(rows[0])
        merged = [f"{a} {b}".strip() for a, b in zip(last, first)]
        prev_lines[-1] = "| " + " | ".join(merged) + " |"
        rows = rows[1:]
    return "\n".join(prev_lines + rows)


def clean_table(md: str) -> str:
    """Tidy a markdown table from PyMuPDF's table detector:
    - drop **bold** / _italic_ markup and "Col3"-style placeholder headers
    - a merged header cell spans to the right, so blanks inherit from the left
    - a second header row (all bold) is merged into the first; a sub-header
      cell that just repeats its left neighbour is spill from a merged cell
    """
    lines = [line for line in md.strip().split("\n") if line.strip()]
    rows = [table_cells(line) for line in lines]
    header, body = rows[0], rows[2:]
    sub = None
    if body and all(c.startswith("**") for c in body[0] if c):
        sub, body = body[0], body[1:]

    def plain(cell: str) -> str:
        cell = re.sub(r"\*\*", "", cell)
        cell = re.sub(r"(?<!\w)_|_(?!\w)", "", cell)
        return re.sub(r"\s+", " ", cell).strip()

    header = ["" if re.fullmatch(r"Col\d+", c) else plain(c) for c in header]
    for i in range(1, len(header)):
        header[i] = header[i] or header[i - 1]
    if sub:
        sub = [plain(c) for c in sub]
        sub = [c if i == 0 or c != sub[i - 1] else "" for i, c in enumerate(sub)]
        header = [f"{h}, {s[0].lower()}{s[1:]}" if s and s != h else h
                  for h, s in zip(header, sub)]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(plain(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def _title_case(heading: str) -> str:
    """"13 THIRTEENTH-MONTH PAY" -> "13. Thirteenth-Month Pay"."""
    words = []
    for i, w in enumerate(heading.split()):
        if w.upper() in KEEP_CASE:
            words.append(KEEP_CASE[w.upper()])
        elif w.lower() in SMALL_WORDS and i:
            words.append(w.lower())
        else:
            words.append("-".join(p.capitalize() for p in w.split("-")))
    title = " ".join(words)
    return re.sub(r"^(\d+) ", r"\1. ", title)


def _finish_section(section: dict) -> dict:
    # Handbook lines are 11.5pt apart; paragraphs 17.5pt and list items 14.4pt
    paragraphs = join_rows(section["rows"], para_gap=13.5)
    return {
        "doc": "handbook",
        "section": _title_case(section["section"]),
        "pages": _page_span(section["rows"]),
        "pdf_page": section["rows"][0].pdf_page,
        "text": "\n".join(p for p, _ in paragraphs),
        # (text, printed page, pdf page, is subheading) per paragraph, so each
        # chunk cites its own page and knows which subsection it belongs to
        "paragraphs": [(p, r.page, r.pdf_page, is_subheading(r)) for p, r in paragraphs],
        "notes": [n for n in section["notes"] if n],
    }
