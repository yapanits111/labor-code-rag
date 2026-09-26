"""Step 0: sources/*.pdf -> data/corpus.json (structured, cleaned records).

    python build_corpus.py

The PDFs are not committed (see sources/README.md for where they come from);
the extracted corpus is, so the app can be rebuilt and deployed without them.
"""
import json
from pathlib import Path

from labor_rag.extract import extract_handbook, extract_labor_code

SOURCES = Path("sources")
OUT = Path("data/corpus.json")


def main():
    articles = extract_labor_code(SOURCES / "labor-code-2022.pdf")
    sections = extract_handbook(SOURCES / "wsmb-handbook-2024.pdf")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(articles + sections, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    words = sum(len(r["text"].split()) for r in articles + sections)
    print(f"{len(articles)} Labor Code articles + {len(sections)} Handbook sections "
          f"({words:,} words) -> {OUT}")


if __name__ == "__main__":
    main()
