#!/usr/bin/env python3
"""
Compute average CodeBLEU over multiple examples.

Input formats
-------------
Input: JSONL with one example per line.
Each line is a JSON object with the following fields:
- "prediction": string (required) - model output code snippet
- "reference": string (optional) - single reference for this example
- "references": [string, ...] (optional) - multiple references; if present,
  the best-scoring reference per example is used (by overall CodeBLEU)
- "lang": string (optional) - language for this example, defaults to CLI --lang

At least one of "reference" or "references" must be provided per example.

Examples (JSONL):
  {"prediction": "def add(a,b):\n return a+b", "reference": "def sum(x,y):\n return x+y"}
  {"prediction": "def f():\n pass", "references": ["def f():\n return None", "def f():\n ..."], "lang": "python"}


Output
------
- Prints a concise human-readable summary of average metrics across all examples.
- With --pretty, also prints a JSON object with averages and counts.

Metrics averaged: codebleu, ngram_match_score, weighted_ngram_match_score,
syntax_match_score, dataflow_match_score.

Notes on multi-reference handling
---------------------------------
When an example provides multiple references ("references"), this script
computes CodeBLEU against each reference separately and takes the result with
the highest overall 'codebleu' as the score for that example. The final
averages are then computed across examples.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional
from codebleu import calc_codebleu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute average CodeBLEU score over examples.")
    p.add_argument("--data", dest="data_path", required=True, help="Path to JSONL examples (one per line)")
    p.add_argument("--lang", default="python", help="Default programming language (used if example has no 'lang')")
    p.add_argument(
        "--weights",
        type=float,
        nargs=4,
        metavar=("NGRAM", "W_NGRAM", "SYNTAX", "DATAFLOW"),
        default=(0.25, 0.25, 0.25, 0.25),
        help="Weights for CodeBLEU components (default: 0.25 0.25 0.25 0.25)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the resulting metrics as JSON.",
    )
    p.add_argument(
        "--show-summary-only",
        action="store_true",
        help="Only print the human-readable summary (omit JSON output).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Helper to compute a single example score given prediction and a single reference
    def score_single(pred: str, ref: str, lang: str) -> Dict[str, float]:
        return calc_codebleu([ref], [pred], lang=lang, weights=tuple(args.weights), tokenizer=None)

    totals: Dict[str, float] = {
        "codebleu": 0.0,
        "ngram_match_score": 0.0,
        "weighted_ngram_match_score": 0.0,
        "syntax_match_score": 0.0,
        "dataflow_match_score": 0.0,
    }
    count = 0

    per_example_results: List[Dict[str, object]] = []

    # JSONL mode (required)
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {line_no}: {e}", file=sys.stderr)
                sys.exit(2)

            pred = obj.get("prediction")
            lang = obj.get("lang") or args.lang
            ref: Optional[str] = obj.get("reference")
            refs: Optional[List[str]] = obj.get("references")

            if not isinstance(pred, str):
                print(f"Line {line_no}: missing or invalid 'prediction' (must be string)", file=sys.stderr)
                sys.exit(2)

            candidate_refs: List[str] = []
            if isinstance(ref, str):
                candidate_refs.append(ref)
            if isinstance(refs, list):
                candidate_refs.extend([r for r in refs if isinstance(r, str)])

            if not candidate_refs:
                print(f"Line {line_no}: provide either 'reference' or 'references'", file=sys.stderr)
                sys.exit(2)

            # Choose best-scoring reference for this example
            best_result: Optional[Dict[str, float]] = None
            best_ref: Optional[str] = None
            for r in candidate_refs:
                res = score_single(pred, r, lang)
                if best_result is None or res["codebleu"] > best_result["codebleu"]:
                    best_result, best_ref = res, r

            assert best_result is not None

            for k in totals:
                totals[k] += float(best_result[k])
            count += 1

            per_example_results.append({
                "prediction": pred,
                "chosen_reference": best_ref,
                "lang": lang,
                **{k: float(best_result[k]) for k in totals.keys()},
            })

    if count == 0:
        print("No examples found.", file=sys.stderr)
        sys.exit(2)

    averages = {k: (v / count) for k, v in totals.items()}

    # Human-readable summary
    print(f"Average CodeBLEU over {count} example(s) (lang default: {args.lang})")
    print(f"- CodeBLEU:                 {averages['codebleu']:.4f}")
    print(f"- N-gram match:             {averages['ngram_match_score']:.4f}")
    print(f"- Weighted n-gram match:    {averages['weighted_ngram_match_score']:.4f}")
    print(f"- Syntax match:             {averages['syntax_match_score']:.4f}")
    print(f"- Dataflow match:           {averages['dataflow_match_score']:.4f}")

    # Optional JSON dump of summary
    if not args.show_summary_only:
        result = {
            "count": count,
            "default_lang": args.lang,
            "weights": tuple(args.weights),
            "averages": averages,
        }
        if args.pretty:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
