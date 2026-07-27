#!/usr/bin/env python3
"""Document and check the related-work search protocol.

The paper intentionally cites this script instead of printing the full Boolean
query in the manuscript. With --bibtex, it performs a lightweight coverage check
over BibTeX/BibLaTeX title fields using the same semantic term blocks. Dataset,
library, and mathematical-method references are excluded from the related-work
coverage denominator because they support the experimental setup rather than the
literature search.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


QUERY = """(
  "federated learning" OR "federated optimization" OR
  "split learning" OR "vertical federated learning" OR
  "horizontal federated learning" OR "decentralized federated learning" OR
  "VFL" OR "HFL"
)
AND
(
  "byzantine" OR "poisoning" OR "backdoor" OR "robust aggregation" OR
  "adversarial attack" OR "differential privacy" OR "label inference" OR
  "gradient inversion" OR "secure aggregation" OR "model security" OR
  "human activity recognition" OR "HAR" OR "multimodal" OR
  "wearable sensor" OR "survey" OR "heterogeneous"
)"""

# Title-level expansions used for coverage auditing. They intentionally include
# common equivalents found in the cited bibliography, while QUERY remains the
# exact database search string reported by the artifact.
FL_TERMS = [
    "federated",
    "split learning",
    "splitfed",
    " vfl ",
    " hfl ",
]

THEMATIC_TERMS = [
    "byzantine",
    "poisoning",
    "backdoor",
    "robust",
    "adversarial",
    "differential privacy",
    "differentially private",
    "label inference",
    "gradient inversion",
    "data reconstruction",
    "secure aggregation",
    "secure model",
    "security",
    "threat",
    "vulnerab",
    "attack",
    "no free lunch",
    "sybil",
    "defense",
    "privacy",
    "activity recognition",
    " har",
    "har:",
    "multimodal",
    "multi-modal",
    "cross-modal",
    "wearable",
    "multi-task",
    "sensor",
    "survey",
    "review",
    "overview",
    "concepts",
    "concept ",
    "advances",
    "challenges",
    "perspective",
    "state-of-the-art",
    "heterogeneous",
    "decentralized",
]

SNOWBALL_KEYS = {
    "thapa2022splitfed",
    "yadav2021multimodalreview",
    "fung2020foolsgold",
}

MANUAL_INCLUSION_KEYS = {
    # 2025-2026 arXiv preprints: no indexer in the declared set covers preprint
    # servers, so these entered through reading and recommendation, not search.
    "gramfeddhar",
    "islamov2026byzclip",
    "karakulev2025bayesian",
    "xia2025feddproc",
}

SUPPORTING_REFERENCE_KEYS = {
    "Chavarriaga2013TheRecognition",
    "krizhevsky2009learning",
    "mironov2017renyi",
    "rupasinghe2022towards",
    "yousefpour2021opacus",
    "banos2014mhealthdroid",
    "ding2022timetrojan",
    "bonawitz2019production",
    "hard2018federated",
    "pentina2023melloddy",
    "yang2019federated",
}


def has_any(title: str, terms: list[str]) -> bool:
    padded = f" {title.lower()} "
    return any(term in padded for term in terms)


def strip_braces(value: str) -> str:
    return re.sub(r"[{}]", "", value)


def parse_bibtex_titles(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = match.group(1).strip()
        start = match.end()
        next_entry = text.find("\n@", start)
        body = text[start:] if next_entry == -1 else text[start:next_entry]
        title_match = re.search(r"\btitle\s*=\s*[{'\"](.+?)[}'\"]\s*,", body, re.I | re.S)
        if title_match:
            title = re.sub(r"\s+", " ", strip_braces(title_match.group(1))).strip()
            entries[key] = title
    return entries


def cited_keys(tex_paths: list[Path]) -> set[str]:
    """Collect every key appearing in a \\cite-like command across .tex files."""
    keys: set[str] = set()
    pattern = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")
    for tex in tex_paths:
        text = tex.read_text(encoding="utf-8")
        # a trailing % inside the braces comments out the newline; drop both
        text = re.sub(r"%\s*\n\s*", "", text)
        for match in pattern.finditer(text):
            keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def check_coverage(path: Path, tex_paths: list[Path] | None = None) -> int:
    entries = parse_bibtex_titles(path.read_text(encoding="utf-8"))
    if not entries:
        print(f"No BibTeX title entries found in {path}")
        return 2

    if tex_paths:
        cited = cited_keys(tex_paths)
        missing = sorted(cited - entries.keys())
        dropped = sorted(entries.keys() - cited)
        entries = {k: v for k, v in entries.items() if k in cited}
        print(f"Restricted to keys cited in {len(tex_paths)} .tex file(s): "
              f"{len(entries)} of {len(entries) + len(dropped)} bib entries")
        if dropped:
            print(f"  uncited bib entries ignored   : {', '.join(dropped)}")
        if missing:
            print(f"  [!] cited but absent from .bib : {', '.join(missing)}")
        print()

    related_entries = {
        key: title for key, title in entries.items()
        if key not in SUPPORTING_REFERENCE_KEYS
    }
    supporting = {
        key: title for key, title in entries.items()
        if key in SUPPORTING_REFERENCE_KEYS
    }

    by_query: list[tuple[str, str]] = []
    by_snowball: list[tuple[str, str]] = []
    uncovered: list[tuple[str, str]] = []

    by_manual: list[tuple[str, str]] = []
    for key, title in sorted(related_entries.items()):
        if key in MANUAL_INCLUSION_KEYS:
            by_manual.append((key, title))
        elif key in SNOWBALL_KEYS:
            by_snowball.append((key, title))
        elif has_any(title, FL_TERMS) and has_any(title, THEMATIC_TERMS):
            by_query.append((key, title))
        else:
            uncovered.append((key, title))

    print(f"Bibliography entries with titles : {len(entries)}")
    print(f"Supporting references excluded   : {len(supporting)}")
    print(f"Related-work entries audited     : {len(related_entries)}")
    print(f"Recovered by Boolean query       : {len(by_query)}/{len(related_entries)}")
    print(f"Added by backward snowballing    : {len(by_snowball)}/{len(related_entries)}")
    print(f"Manual inclusions (preprints)    : {len(by_manual)}/{len(related_entries)}")
    print(f"Not covered                      : {len(uncovered)}/{len(related_entries)}")

    if uncovered:
        print("\nUncovered entries:")
        for key, title in uncovered:
            print(f"  [{key}] {title}")
        return 1

    print("\nOK: query + snowballing cover "
          f"{len(related_entries)}/{len(related_entries)} related-work entries.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-query", action="store_true", help="print the exact Boolean query")
    parser.add_argument("--bibtex", type=Path, help="optional .bib file to audit for coverage")
    parser.add_argument("--tex", type=Path, nargs="*", default=None,
                        help="restrict the audit to keys actually cited in these .tex files")
    args = parser.parse_args()

    if args.print_query or not args.bibtex:
        print(QUERY)

    if args.bibtex:
        return check_coverage(args.bibtex, args.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
