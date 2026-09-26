"""Records -> chunks, split along the documents' own structure.

Labor Code: one chunk per article. The article is the natural unit of the
law, and it is what an answer should cite. Long articles are split at
paragraph boundaries ("(a)", "(b)", ...) into parts; footnotes go into a
separate "notes" chunk that points back to the same article.

Handbook: each topic section is packed paragraph by paragraph (and table row
by table row) into ~220-word chunks, so no paragraph or row is cut in half,
and each chunk cites the page it starts on.
"""
import re

MAX_ARTICLE_WORDS = 350  # an article up to this long stays in one chunk
PART_WORDS = 220         # target size when something has to be split


def _words(text: str) -> int:
    return len(text.split())


def window(text: str, size: int, overlap: int) -> list[str]:
    """Fallback for a single paragraph longer than `size` words: sliding
    word windows that overlap, so no sentence is lost at a boundary."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def split_table(text: str, max_words: int) -> list[str]:
    """A markdown table too big for one chunk is split between rows; every
    piece repeats the header, so each chunk can be read on its own."""
    lines = text.split("\n")
    head, rows = lines[:2], lines[2:]
    pieces, current = [], []
    for row in rows:
        if current and _words("\n".join(head + current + [row])) > max_words:
            pieces.append("\n".join(head + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(head + current))
    return pieces


def _text(p) -> str:
    return p if isinstance(p, str) else p[0]


def _is_heading(p) -> bool:
    return not isinstance(p, str) and len(p) > 3 and p[3]


def pack(paragraphs: list, max_words: int = PART_WORDS) -> list[list]:
    """Greedily group whole paragraphs into chunks of at most `max_words`.
    `paragraphs` are strings, or tuples (text, page, pdf_page, is_subheading).
    A subheading is never left dangling at the end of a chunk, and a
    paragraph too big for any chunk is split into overlapping windows."""
    groups, current = [], []

    def flush():
        nonlocal current
        carry = [current.pop()] if len(current) > 1 and _is_heading(current[-1]) else []
        if current:
            groups.append(current)
        current = carry

    for p in paragraphs:
        n = _words(_text(p))
        if n > max_words:  # too big: split tables between rows, text into windows
            lead = " ".join(_text(q) for q in current) if sum(
                _words(_text(q)) for q in current) < 60 else ""
            if not lead:
                flush()
            text = _text(p)
            pieces = (split_table(text, max_words) if text.startswith("| ")
                      else window(text, max_words, 40))
            for i, piece in enumerate(pieces):
                piece = f"{lead}\n{piece}" if lead and i == 0 else piece
                groups.append([piece if isinstance(p, str) else (piece, *p[1:])])
            current = []
            continue
        size = sum(_words(_text(q)) for q in current)
        # a new subsection starts a new chunk, so one chunk = one topic
        if current and (size + n > max_words or (_is_heading(p) and size >= 60)):
            flush()
        current.append(p)
    if current:
        groups.append(current)
    return groups


def article_label(rec: dict) -> str:
    old = f" (formerly Art. {rec['old_article']})" if rec.get("old_article") else ""
    return f"Labor Code Art. {rec['article']}{old}, {rec['title']}"


def _article_chunks(rec: dict) -> list[dict]:
    base = {"doc": "labor_code", "article": rec["article"],
            "old_article": rec["old_article"], "title": rec["title"],
            "path": rec["path"], "pages": rec["pages"], "pdf_page": rec["pdf_page"]}
    label = article_label(rec)
    if _words(rec["text"]) <= MAX_ARTICLE_WORDS:
        parts = [rec["text"]]
    else:
        parts = ["\n".join(g) for g in pack(rec["text"].split("\n"))]

    chunks = []
    for i, part in enumerate(parts, start=1):
        suffix = f" (part {i} of {len(parts)})" if len(parts) > 1 else ""
        chunks.append({**base, "id": f"art-{rec['article']}" + (f"-{i}" if len(parts) > 1 else ""),
                       "kind": "article", "label": label + suffix, "text": part})
    if rec["notes"]:
        for i, group in enumerate(pack(rec["notes"]), start=1):
            chunks.append({**base, "id": f"art-{rec['article']}-notes" + (f"-{i}" if i > 1 else ""),
                           "kind": "notes", "label": f"Footnotes to {label}",
                           "text": "\n".join(group)})
    return chunks


def _handbook_chunks(rec: dict) -> list[dict]:
    slug = re.sub(r"[^a-z0-9]+", "-", rec["section"].lower()).strip("-")
    label = f"DOLE Handbook 2024, {rec['section']}"

    # stamp each paragraph with the subheading ("A. Definition") it falls under:
    # (text, page, pdf_page, is_subheading, subheading)
    paragraphs, current = [], None
    for text, page, pdf_page, is_head in rec["paragraphs"]:
        current = text if is_head else current
        paragraphs.append((text, page, pdf_page, is_head, current))

    chunks = []
    for i, group in enumerate(pack(paragraphs), start=1):
        pages = list(dict.fromkeys(p[1] for p in group))
        sub = group[0][4]
        text = "\n".join(p[0] for p in group)
        if sub and not _is_heading(group[0]):
            text = f"{sub} (continued)\n{text}"  # a table or list that runs on
        chunks.append({
            "id": f"hb-{slug}-{i}", "doc": "handbook", "kind": "guide",
            "section": rec["section"],
            "label": f"{label}, {sub}" if sub else label,
            "pages": pages[0] if len(pages) == 1 else f"{pages[0]}-{pages[-1]}",
            "pdf_page": group[0][2],
            "text": text,
        })
    if rec["notes"]:
        for i, group in enumerate(pack(rec["notes"]), start=1):
            chunks.append({"id": f"hb-{slug}-notes-{i}", "doc": "handbook", "kind": "notes",
                           "section": rec["section"], "label": f"Footnotes to {label}",
                           "pages": rec["pages"], "pdf_page": rec["pdf_page"],
                           "text": "\n".join(group)})
    return chunks


def chunk_records(records: list[dict]) -> list[dict]:
    chunks = []
    for rec in records:
        chunks.extend(_article_chunks(rec) if rec["doc"] == "labor_code"
                      else _handbook_chunks(rec))
    return chunks


def embedding_input(chunk: dict) -> str:
    """What gets embedded: the citation label and where it sits in the Code
    ("Book Three > Title I > Chapter III – Holidays...") in front of the text.
    That context helps a short article match questions phrased in everyday
    words, e.g. a question about "day off" finding "Right to Weekly Rest Day"."""
    context = chunk.get("path") or chunk.get("section", "")
    return f"{chunk['label']}\n{context}\n{chunk['text']}"
