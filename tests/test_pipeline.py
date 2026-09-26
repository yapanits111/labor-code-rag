"""Offline tests: no embedding model, no LLM calls. They check the parts
that were written by hand, against the real extracted corpus."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from labor_rag.chunk import chunk_records, pack
from labor_rag.evaluate import article_check, contains_all, grade, matches_source
from labor_rag.generate import cited_numbers, tidy_citations
from labor_rag.retrieve import article_refs, lookup_articles
from labor_rag.similarity import cosine_similarity, cosine_similarity_matrix
from labor_rag.store import VectorStore

CORPUS = json.loads(Path("data/corpus.json").read_text(encoding="utf-8"))
ARTICLES = {r["article"]: r for r in CORPUS if r["doc"] == "labor_code"}
SECTIONS = {r["section"]: r for r in CORPUS if r["doc"] == "handbook"}


# --- extraction (checked against the real corpus) ----------------------------

def test_every_article_extracted_once():
    assert sorted(int(n) for n in ARTICLES) == list(range(1, 318))


def test_old_article_numbers_kept():
    assert ARTICLES["294"]["title"] == "Security of Tenure"
    assert ARTICLES["294"]["old_article"] == "279"
    assert ARTICLES["297"]["old_article"] == "282"


def test_footnote_markers_removed_and_notes_attached():
    art = ARTICLES["94"]
    assert art["title"] == "Right to Holiday Pay"  # not "Right to Holiday Pay.78"
    assert any("R.A. No. 9849" in n for n in art["notes"])


def test_book_title_chapter_path():
    assert ARTICLES["94"]["path"].startswith("Book Three")
    assert "Chapter III" in ARTICLES["94"]["path"]


def test_rate_table_rows_stay_aligned():
    # the row PDF text extraction scrambled: label and value must share a line
    text = SECTIONS["Compliance Guide"]["text"]
    assert "Regular holiday falling on rest day | 2.6 or 260%" in text
    assert "Special (non-working) day falling on rest day | 1.5 or 150%" in text


def test_handbook_sections_and_subheadings():
    assert "13. Thirteenth-Month Pay" in SECTIONS
    subs = [p[0] for p in SECTIONS["13. Thirteenth-Month Pay"]["paragraphs"] if p[3]]
    assert "A. Definition" in subs


# --- chunking ----------------------------------------------------------------

CHUNKS = chunk_records(CORPUS)


def test_chunk_ids_unique_and_sizes_bounded():
    ids = [c["id"] for c in CHUNKS]
    assert len(ids) == len(set(ids))
    assert max(len(c["text"].split()) for c in CHUNKS) < 400


def test_short_article_is_one_chunk_with_citation_label():
    art = [c for c in CHUNKS if c["id"] == "art-294"]
    assert len(art) == 1
    assert art[0]["label"] == "Labor Code Art. 294 (formerly Art. 279), Security of Tenure"


def test_subsection_continuation_is_labelled():
    cont = [c for c in CHUNKS if c["doc"] == "handbook" and "(continued)" in c["text"]]
    assert cont and all(c["label"].count(",") >= 2 for c in cont)


def test_big_table_splits_between_rows_with_header():
    table = "| Leave | Days |\n|---|---|\n" + "\n".join(f"| {'word ' * 40}| {i} |" for i in range(8))
    pieces = pack([table], max_words=120)
    assert len(pieces) > 1
    for (piece,) in pieces:
        assert piece.startswith("| Leave | Days |\n|---|---|\n")
        assert all(line.endswith("|") for line in piece.split("\n"))  # no row cut in half


def test_leave_table_is_one_table_across_pages():
    guide = SECTIONS["Compliance Guide"]["text"]
    assert guide.count("| Type of Leave Benefit |") == 1
    assert "suffered a miscarriage. | seven days |" in guide


def test_pack_keeps_paragraphs_whole():
    paras = ["a " * 100, "b " * 100, "c " * 100]
    groups = pack(paras, max_words=220)
    assert [len(g) for g in groups] == [2, 1]


# --- similarity and store ------------------------------------------------------

def test_cosine_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(math.cos(math.pi / 4))


def test_matrix_matches_scalar():
    rng = np.random.default_rng(0)
    q, m = rng.normal(size=8), rng.normal(size=(5, 8))
    assert cosine_similarity_matrix(q, m) == pytest.approx(
        [cosine_similarity(q.tolist(), r.tolist()) for r in m])


# --- retrieval: direct article lookup -----------------------------------------

def test_article_refs():
    assert article_refs("What does Article 279 say?") == ["279"]
    assert article_refs("Compare Art. 94 and art 95") == ["94", "95"]
    assert article_refs("How much is 13th month pay?") == []


def test_old_number_finds_renumbered_article():
    store = VectorStore("toy")
    chunks = [c for c in CHUNKS if c["id"] in ("art-294", "art-279-1", "art-94")]
    store.add(chunks, np.eye(len(chunks), dtype=np.float32))
    found = {c["id"] for c in lookup_articles(["279"], store)}
    assert found == {"art-279-1", "art-294"}  # today's Art. 279 and the old one


# --- generation helpers ----------------------------------------------------------

def test_tidy_citations():
    assert tidy_citations("pay【1†L1-L7】 rule [4](b)") == "pay[1] rule [4] (b)"
    assert tidy_citations("twice [1][1] [b]") == "twice [1] "
    assert tidy_citations("strike-breakers [2(c)].") == "strike-breakers [2] (c)."


def test_cited_numbers():
    assert cited_numbers("A [2][4], B [2, 3], C [1]") == [2, 4, 3, 1]


# --- eval grading --------------------------------------------------------------

def test_keyword_alternatives_and_typography():
    assert contains_all("Pay is 2 × the rate (200%)", [["200%", "twice"]])
    assert contains_all("Row‑Level", [["row level"]])
    assert not contains_all("five days", [["seven", "7"]])


def test_percent_written_out_matches():
    # both systems wrote rates in words; the grader must not penalise that
    assert contains_all("plus twenty-five percent", [["25%", "twenty five%"]])
    assert contains_all("an additional 30 per cent", [["30%"]])
    assert not contains_all("a percentage of the wage", [["30%"]])


def test_matches_source():
    art = {"doc": "labor_code", "article": "94"}
    hb = {"doc": "handbook", "section": "13. Thirteenth-Month Pay"}
    assert matches_source(art, "art:94") and not matches_source(art, "art:95")
    assert matches_source(hb, "hb:13") and not matches_source(hb, "hb:1")


def test_grade_separates_failure_modes():
    q = {"sources": ["art:294"], "keywords": [["reinstatement"]]}
    assert grade("You are entitled to reinstatement.", q, rank=1) == "PASS"
    assert grade("I couldn't find this.", q, rank=None) == "RETRIEVAL"
    assert grade("I couldn't find this.", q, rank=2) == "GENERATION"


def test_article_check_flags_outdated_numbers():
    q = {"sources": ["art:297"]}
    old = {"297": "282"}
    assert article_check("Under Article 297, ...", q, old) == "current"
    assert article_check("Under Article 282, ...", q, old) == "outdated"
    assert article_check("The law says ...", q, old) == "missing"
    assert article_check("...", {"sources": ["hb:13"]}, old) is None


def test_ambiguous_article_number_is_spelled_out():
    from labor_rag.retrieve import renumbering_notes
    store = VectorStore("toy")
    chunks = [c for c in CHUNKS if c["id"] in ("art-294", "art-279-1", "art-94")]
    store.add(chunks, np.eye(len(chunks), dtype=np.float32))
    [note] = renumbering_notes("What does Article 279 say?", store)
    assert "Prohibited activities" in note and "Security of Tenure" in note and "Art. 294" in note
    assert renumbering_notes("What does Article 94 say?", store) == []


def test_relevance_gate():
    from labor_rag.generate import MIN_RELEVANCE, is_relevant
    far = [{"score": MIN_RELEVANCE - 0.07}, {"score": 0.40}]
    near = [{"score": MIN_RELEVANCE + 0.05}]
    named = [{"score": 1.0, "via": "article number"}]
    assert not is_relevant(far)          # "what can i do with this?" scored 0.55
    assert is_relevant(near)
    assert is_relevant(named)            # "What does Article 279 say?" always answered


def test_subsection_letter_citations_removed():
    assert tidy_citations("the only requirement [6][D]. Not cash [6][E].") == \
        "the only requirement [6]. Not cash [6]."
