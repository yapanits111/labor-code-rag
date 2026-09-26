"""Steps 4-5: find the passages that answer a question.

Two routes, merged:
1. Semantic search: embed the question, rank every chunk by cosine similarity.
2. Direct lookup: if the question names an article ("Article 94", "Art. 279"),
   fetch it by number. Embeddings are weak at exact numbers, and many people
   (and chatbots) still use the pre-2015 numbering, so an old number also
   pulls in the renumbered article: "Article 279" finds Art. 294 (formerly
   279), Security of Tenure, as well as today's Art. 279.
"""
import re

from .embed import embed_query
from .store import VectorStore

ARTICLE_REF = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{1,3}(?:-[A-Z])?)\b", re.I)


def article_refs(question: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in ARTICLE_REF.findall(question)))


def lookup_articles(numbers: list[str], store: VectorStore) -> list[dict]:
    """Article text chunks whose current OR old number matches."""
    hits = []
    for n in numbers:
        for c in store.chunks:
            if c["kind"] == "article" and n in (c.get("article"), c.get("old_article")):
                hits.append({**c, "score": 1.0, "via": "article number"})
    return hits


def renumbering_notes(question: str, store: VectorStore) -> list[str]:
    """An article number the question names can be both a current article and
    another article's pre-2015 number ("Article 279": today, Prohibited
    Activities; before 2015, Security of Tenure, now Art. 294). Spell that
    out for the LLM instead of hoping it notices."""
    notes = []
    for n in article_refs(question):
        articles = [c for c in store.chunks if c["kind"] == "article"]
        now = next((c for c in articles if c["article"] == n), None)
        was = next((c for c in articles if c.get("old_article") == n), None)
        if now and was:
            notes.append(f'"Article {n}" is ambiguous. Today, Art. {n} is "{now["title"]}". '
                         f'Before the 2015 renumbering, Art. {n} was "{was["title"]}", which is '
                         f'now Art. {was["article"]}.')
    return notes


def retrieve(question: str, store: VectorStore, k: int = 6) -> list[dict]:
    direct = lookup_articles(article_refs(question), store)[:k]
    semantic = store.search(embed_query(question), k + len(direct))
    results, seen = [], set()
    for c in direct + semantic:
        if c["id"] not in seen:
            seen.add(c["id"])
            results.append(c)
    return results[:k]
