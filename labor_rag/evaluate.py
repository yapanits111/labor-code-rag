"""Grading helpers for the eval.

Each question lists the `sources` that hold its answer ("art:94" = Labor Code
Article 94, "hb:13" = Handbook section 13) and `keywords` a correct answer
must contain. Keywords are checked twice: in the retrieved passages (was the
answer retrieved?) and in the reply (did the model use it?). That is what
separates the two failure modes.
"""
import re
import unicodedata

ARTICLE_REF = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{1,3})\b", re.I)


def normalise(text: str) -> str:
    """Lowercase and flatten typographic look-alikes (non-breaking hyphens,
    narrow no-break spaces) so keyword checks don't fail on typography;
    "25 percent" and "25 per cent" become "25%"."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch) in ("Pd", "Zs") else ch for ch in text)
    text = re.sub(r"\s*\b(?:percent|per cent)\b", "%", text)
    return re.sub(r"\s+%", "%", text)  # "125 %" (thin space) -> "125%"


def contains_all(text: str, keywords: list[list[str]]) -> bool:
    """Every keyword group must match; a group is a list of accepted alternatives."""
    text = normalise(text)
    return all(any(normalise(alt) in text for alt in group) for group in keywords)


def matches_source(chunk: dict, spec: str) -> bool:
    kind, _, value = spec.partition(":")
    if kind == "art":
        return chunk["doc"] == "labor_code" and chunk.get("article") == value
    section = chunk.get("section", "")
    return chunk["doc"] == "handbook" and (section == value or section.split(".")[0] == value)


def answer_rank(chunks: list[dict], question: dict) -> int | None:
    """Rank of the first retrieved passage that actually holds the answer:
    from an expected source AND containing the keywords."""
    for rank, c in enumerate(chunks, start=1):
        if (any(matches_source(c, s) for s in question["sources"])
                and contains_all(c["text"], question["keywords"])):
            return rank
    return None


def summarise(ranks: list[int | None]) -> tuple[float, float]:
    """hit@k (share of questions with the answer in the top k) and MRR."""
    hit = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    return hit, mrr


def grade(reply: str, question: dict, rank: int | None) -> str:
    """PASS, or which half of the pipeline to blame:
    RETRIEVAL  - the answer never reached the prompt
    GENERATION - it was in the prompt, the reply still missed it"""
    if contains_all(reply, question["keywords"]):
        return "PASS"
    if question["sources"] and rank is None:
        return "RETRIEVAL"
    return "GENERATION"


def summarise_answers(rows: list[dict], who: str) -> dict:
    """Totals for one system ("rag" or "baseline") from graded eval rows.
    - correct: answer contains the expected facts
    - names_article: of the questions answered purely by Labor Code articles,
      how many answers name (or cite) the right article by its current number.
      Questions the Handbook also answers are left out: citing the Handbook
      there is just as valid.
    - outdated_numbers: answers that name a pre-2015 article number instead
    - wrong_numbers: answers that name only articles that don't say this"""
    graded = [r for r in rows if f"{who}_verdict" in r]
    verdicts = [r[f"{who}_verdict"] for r in graded]
    article_only = [r for r in graded if r["sources"]
                    and all(s.startswith("art:") for s in r["sources"])]
    summary = {
        "correct": verdicts.count("PASS"), "of": len(graded),
        "names_article": sum(r[f"{who}_article"] == "current" for r in article_only),
        "article_questions": len(article_only),
        "outdated_numbers": sum(r.get(f"{who}_article") == "outdated" for r in graded),
        "wrong_numbers": sum(r.get(f"{who}_article") == "wrong" for r in graded),
    }
    if who == "rag":
        summary["retrieval_failures"] = verdicts.count("RETRIEVAL")
        summary["generation_failures"] = verdicts.count("GENERATION")
    return summary


def mentioned_articles(text: str) -> set[str]:
    return set(ARTICLE_REF.findall(text))


def article_check(reply: str, question: dict, old_numbers: dict[str, str],
                  cited_chunks: list[dict] = ()) -> str | None:
    """For questions answered by specific articles:
    "current"  - names a correct article by its current number (or, for the
                 RAG answer, cites a passage of that article)
    "outdated" - names it only by its pre-2015 number
    "wrong"    - names only articles that don't say this
    "missing"  - names no article at all
    None for questions that don't depend on an article."""
    expected = [s.split(":")[1] for s in question["sources"] if s.startswith("art:")]
    if not expected:
        return None
    named = mentioned_articles(reply)
    cited = {c.get("article") for c in cited_chunks if c["doc"] == "labor_code"}
    if named & set(expected) or cited & set(expected):
        return "current"
    if named & {old_numbers[n] for n in expected if old_numbers.get(n)}:
        return "outdated"
    return "wrong" if named else "missing"
