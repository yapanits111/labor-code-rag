"""Ask a labor law question from the terminal.

    python ask.py "Am I entitled to 13th month pay if I resigned in June?"
    python ask.py "..." --show-context     # every retrieved passage and its score
    python ask.py "..." --retrieve-only    # no LLM call, no API key needed
    python ask.py                          # interactive mode
"""
import argparse

from labor_rag.embed import MODEL_NAME
from labor_rag.generate import answer, passage_header
from labor_rag.llm import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS
from labor_rag.retrieve import retrieve
from labor_rag.store import VectorStore


def show(chunks: list[dict], cited: list[int] | None = None) -> None:
    for i, c in enumerate(chunks, start=1):
        mark = "*" if cited and i in cited else " "
        print(f" {mark}[{i}] {c['score']:.3f}  {passage_header(c)}")


def run(question: str, store: VectorStore, args) -> None:
    if args.retrieve_only:
        show(retrieve(question, store, args.k))
        return
    try:
        result = answer(question, store, k=args.k, provider=args.provider)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return
    print(f"\n{result['answer']}\n")
    if args.show_context:
        print("Retrieved passages (* = cited):")
        show(result["passages"], result["cited"])
    else:
        for n in result["cited"]:
            print(f"  [{n}] {passage_header(result['passages'][n - 1])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--k", type=int, default=6, help="passages to retrieve")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    store = VectorStore.load("index", expected_model=MODEL_NAME)
    if args.question:
        run(args.question, store, args)
        return

    print(f"Loaded {len(store)} passages. Ask a question (empty line to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        run(question, store, args)
        print()


if __name__ == "__main__":
    main()
