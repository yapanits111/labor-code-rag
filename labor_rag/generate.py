"""Steps 6-7: build a grounded prompt from the retrieved passages, call the LLM."""
import re

from .llm import DEFAULT_PROVIDER, call_llm
from .retrieve import renumbering_notes, retrieve
from .store import VectorStore

PROMPT_TEMPLATE = """You explain Philippine labor law in plain language to workers and small employers.

Answer the question using ONLY the numbered passages below. They come from the
Labor Code of the Philippines (DOLE 2022 renumbered edition) and DOLE's 2024
Handbook on Workers' Statutory Monetary Benefits.

Rules:
- Give the direct answer in your first sentence (no "Direct answer:" heading),
  then the key conditions or exceptions.
- Write formulas in plain words with ×, ÷ and =. No LaTeX or math markup.
- Cite every claim with its passage number, like [1] or [2][3]. Plain brackets
  only; to point at a paragraph of an article, name it in words: "Art. 94(b) [4]".
- Use the article numbers exactly as the passages show them. The Labor Code
  was renumbered in 2015: if the question names an article number that a
  passage lists as "formerly Art. N", the user most likely means that article.
  Say it is now Art. X (formerly Art. N) and answer from it. If the number
  matches both a current article and another article's former number, cover
  both briefly: the current one first, then the renumbered one.
- For pay questions, use the rates exactly as the passages state them. Prefer
  the Handbook's combined multipliers (e.g. 2.6 for a regular holiday that is
  also a rest day) over building your own, show the arithmetic, and check
  that every line of it adds up to your total.
- Only if you quote a minimum wage amount, note that it is as of the 2024 Handbook.
- If the answer depends on facts the question doesn't give (company size,
  employment status, the kind of holiday), say what it depends on.
- If the passages don't answer the question, reply exactly:
  "I couldn't find this in the Labor Code or the DOLE Handbook." Do not guess.

Passages:
{context}
{notes}
Question: {question}
Answer:"""

# gpt-oss models sometimes cite in their training format, e.g. 【1†L3-L7】,
# despite the instruction above. Rewrite those to the plain [1] style.
_ODD_CITATION = re.compile(r"【(\d+)[^】]*】")
NOT_FOUND = "I couldn't find this in the Labor Code or the DOLE Handbook."

# Retrieval always returns its top k, even for "tell me a joke". If even the
# best passage is this far from the question, don't ask the LLM at all.
# Calibrated in eval/scope_check.py: no on-topic question (45 tried) scores
# below 0.666; 18 of 20 off-topic ones score below 0.62.
MIN_RELEVANCE = 0.62
OUT_OF_SCOPE = (
    "I couldn't find anything in the Labor Code or the DOLE Handbook that matches "
    "this question. I can answer questions about work in the Philippines: pay and "
    "wages, overtime, holidays, leave, 13th-month pay, working hours, and "
    "termination. Try one of the examples below, or rephrase with more detail.")


def is_relevant(chunks: list[dict]) -> bool:
    """True if the question is close enough to the law to be worth answering:
    an article was named directly, or a passage passes the similarity bar."""
    return any(c.get("via") == "article number" or c["score"] >= MIN_RELEVANCE
               for c in chunks)


def tidy_citations(text: str) -> str:
    text = _ODD_CITATION.sub(r"[\1]", text)
    text = re.sub(r"\[(\d+)\]\(([a-z])\)", r"[\1] (\2)", text)  # [4](b) would render as a link
    text = re.sub(r"\[(\d+)\s*\(([a-z])\)\]", r"[\1] (\2)", text)  # [2(b)] -> [2] (b)
    text = re.sub(r"(\[\d+\])\1+", r"\1", text)  # [1][1] -> [1]
    return re.sub(r"\[[A-Za-z]\]", "", text)  # stray subsection cites like [b] or [D]


def cited_numbers(text: str) -> list[int]:
    """Passage numbers the answer cites, in order of first use. Handles [1],
    [2][3] and [2, 3]."""
    found = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        found.extend(int(n) for n in group.split(","))
    return list(dict.fromkeys(found))


def passage_header(chunk: dict) -> str:
    return f"{chunk['label']} (p. {chunk['pages']})"


def build_prompt(question: str, chunks: list[dict], notes: list[str] = ()) -> str:
    context = "\n\n".join(f"[{i}] {passage_header(c)}\n{c['text']}"
                          for i, c in enumerate(chunks, start=1))
    note_text = "".join(f"\nNote: {n} Cover both, the current article first.\n" for n in notes)
    return PROMPT_TEMPLATE.format(context=context, notes=note_text, question=question)


def answer(question: str, store: VectorStore, k: int = 6,
           provider: str = DEFAULT_PROVIDER) -> dict:
    """Full RAG: retrieve -> prompt -> generate. Returns the answer, every
    passage it was given, and which of those it cited."""
    chunks = retrieve(question, store, k)
    if not is_relevant(chunks):  # off topic or too vague: no LLM call
        return {"answer": OUT_OF_SCOPE, "passages": chunks, "cited": [], "declined": True}
    prompt = build_prompt(question, chunks, renumbering_notes(question, store))
    text = tidy_citations(call_llm(prompt, provider=provider, max_tokens=900))
    cited = [n for n in cited_numbers(text) if 1 <= n <= len(chunks)]
    return {"answer": text, "passages": chunks, "cited": cited}


BASELINE_PROMPT = """Answer this question about Philippine labor law in plain language.
Cite the Labor Code article number(s) your answer relies on.

Question: {question}
Answer:"""


def baseline_answer(question: str, provider: str = DEFAULT_PROVIDER) -> str:
    """The same model with no retrieval: what a plain chatbot would say."""
    return tidy_citations(call_llm(BASELINE_PROMPT.format(question=question),
                                   provider=provider, max_tokens=900))
