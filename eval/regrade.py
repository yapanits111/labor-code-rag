"""Re-grade the answers saved in eval/last_run.json against the current
eval/questions.json and grader, without calling any LLM.

Use it after widening a keyword that was too literal (an answer wrote
"twenty-five percent" where the keyword was "25%"). Every verdict that
changes is printed, so each one can be checked by reading the answer.

    python eval/regrade.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labor_rag.evaluate import (article_check, contains_all, grade,  # noqa: E402
                                summarise_answers)


def main():
    run_path = ROOT / "eval" / "last_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    questions = {q["q"]: q for q in json.loads(
        (ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))}
    chunks = {c["id"]: c for c in json.loads(
        (ROOT / "index" / "chunks.json").read_text(encoding="utf-8"))["chunks"]}
    old_numbers = {c["article"]: c["old_article"] for c in chunks.values()
                   if c["doc"] == "labor_code" and c.get("old_article")}

    for row in run["questions"]:
        q = questions[row["question"]]
        row["sources"] = q["sources"]
        if "rag_answer" in row:
            cited = [chunks[i] for i in row["cited"] if i in chunks]
            new = (grade(row["rag_answer"], q, row["retrieval_rank"]),
                   article_check(row["rag_answer"], q, old_numbers, cited))
            if new != (row["rag_verdict"], row["rag_article"]):
                print(f"RAG   {row['rag_verdict']}/{row['rag_article']} -> {new[0]}/{new[1]}: {q['q']}")
            row["rag_verdict"], row["rag_article"] = new
        if "baseline_answer" in row:
            new = ("PASS" if contains_all(row["baseline_answer"], q["keywords"]) else "FAIL",
                   article_check(row["baseline_answer"], q, old_numbers))
            if new != (row["baseline_verdict"], row["baseline_article"]):
                print(f"plain {row['baseline_verdict']}/{row['baseline_article']} -> "
                      f"{new[0]}/{new[1]}: {q['q']}")
            row["baseline_verdict"], row["baseline_article"] = new

    for who in ("rag", "baseline"):
        if who in run["summary"]:
            run["summary"][who] = summarise_answers(run["questions"], who)
            print(who, run["summary"][who])
    run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
