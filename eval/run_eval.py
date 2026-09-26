"""Measure the system instead of eyeballing it.

    python eval/run_eval.py                          # retrieval only (no API key needed)
    python eval/run_eval.py --generate               # + grade the RAG answers
    python eval/run_eval.py --generate --baseline    # + the same LLM with no retrieval

Every answer is saved to eval/last_run.json, so failures can be read, not just counted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labor_rag.embed import MODEL_NAME  # noqa: E402
from labor_rag.evaluate import (answer_rank, article_check, contains_all,  # noqa: E402
                                grade, summarise, summarise_answers)
from labor_rag.generate import answer, baseline_answer  # noqa: E402
from labor_rag.llm import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS  # noqa: E402
from labor_rag.retrieve import retrieve  # noqa: E402
from labor_rag.store import VectorStore  # noqa: E402


def with_retry(fn, *args, **kwargs):
    """Free-tier LLM APIs rate-limit bursts; wait and retry instead of failing."""
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "limit" not in str(e).lower() or attempt == 4:
                raise
            wait = 20 * (attempt + 1)
            print(f"        (rate limited, retrying in {wait}s)")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--generate", action="store_true", help="grade the RAG answers")
    parser.add_argument("--baseline", action="store_true", help="also ask the LLM with no retrieval")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--baseline-from", metavar="RUN_JSON",
                        help="reuse the plain-LLM answers saved in an earlier run instead of "
                             "asking again (the baseline doesn't change when the RAG side does)")
    args = parser.parse_args()

    store = VectorStore.load(ROOT / "index", expected_model=MODEL_NAME)
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    old_numbers = {c["article"]: c["old_article"] for c in store.chunks
                   if c["doc"] == "labor_code" and c.get("old_article")}

    saved_baseline = {}
    if args.baseline_from:
        saved = json.loads(Path(args.baseline_from).read_text(encoding="utf-8"))
        saved_baseline = {r["question"]: r["baseline_answer"] for r in saved["questions"]
                          if "baseline_answer" in r}
        args.baseline = True

    ranks, rows, stopped = [], [], None
    for q in questions:
        answerable = bool(q["sources"])
        row = {"question": q["q"], "sources": q["sources"]}
        if args.generate:
            try:
                result = with_retry(answer, q["q"], store, k=args.k, provider=args.provider)
            except RuntimeError as e:  # e.g. a daily quota: keep what we have
                stopped = str(e)
                print(f"\nStopped early: {stopped}")
                break
            chunks, reply = result["passages"], result["answer"]
            cited = [chunks[n - 1] for n in result["cited"]]
        else:
            chunks = retrieve(q["q"], store, args.k)

        rank = answer_rank(chunks, q) if answerable else None
        if answerable:
            ranks.append(rank)
        row.update(retrieval_rank=rank, retrieved=[c["id"] for c in chunks])
        line = f"{('rank ' + str(rank)) if rank else ('miss' if answerable else 'n/a'):<7}"

        if args.generate:
            verdict, art = grade(reply, q, rank), article_check(reply, q, old_numbers, cited)
            row.update(rag_verdict=verdict, rag_article=art, cited=[c["id"] for c in cited],
                       rag_answer=reply)
            line += f" RAG {verdict:<10} {art or '':<8}"
        if args.baseline and answerable and q.get("baseline", True):
            base = saved_baseline.get(q["q"]) or with_retry(baseline_answer, q["q"],
                                                            provider=args.provider)
            verdict_b = "PASS" if contains_all(base, q["keywords"]) else "FAIL"
            art_b = article_check(base, q, old_numbers)
            row.update(baseline_verdict=verdict_b, baseline_article=art_b, baseline_answer=base)
            line += f" | plain {verdict_b:<4} {art_b or '':<8}"
        print(f"{line} {q['q'][:64]}")
        rows.append(row)
        time.sleep(args.delay)

    hit, mrr = summarise(ranks) if ranks else (0.0, 0.0)
    summary = {"questions": len(questions), "answerable": len(ranks), "k": args.k,
               f"hit@{args.k}": round(hit, 3), "mrr": round(mrr, 3)}
    print(f"\nRetrieval ({len(ranks)} answerable questions, k={args.k}): "
          f"hit@{args.k} {hit:.0%}, MRR {mrr:.3f}")

    for who, name in (("rag", "RAG"), ("baseline", "Plain LLM")):
        if any(f"{who}_verdict" in r for r in rows):
            summary[who] = t = summarise_answers(rows, who)
            extra = (f" ({t['retrieval_failures']} retrieval, {t['generation_failures']} "
                     "generation failures)" if who == "rag" else "")
            print(f"{name + ':':<11} {t['correct']}/{t['of']} correct{extra}; names the right "
                  f"article {t['names_article']}/{t['article_questions']}; "
                  f"outdated article numbers {t['outdated_numbers']}")

    # an incomplete run never overwrites the last complete one
    out = ROOT / "eval" / ("partial_run.json" if stopped else "last_run.json")
    out.write_text(json.dumps({"provider": args.provider, "summary": summary, "questions": rows,
                               "stopped_early": stopped}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{len(rows)} of {len(questions)} questions saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
