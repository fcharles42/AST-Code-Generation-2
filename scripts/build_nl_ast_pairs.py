import os
import sys
import json
import ast
from tqdm import tqdm
from datasets import load_dataset

# ======================================================
# Project setup
# ======================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ast_codec.python_ast import code_to_ast, ast_to_tokens

# ======================================================
# Config
# ======================================================
OUTPUT_PATH = "data/processed/nl_ast_pairs.jsonl"

DATASET_NAME = "code_search_net"
LANGUAGE = "python"
SPLIT = "train"

MAX_SAMPLES = 20_000        
MAX_AST_TOKENS = 4096      
MIN_DOCSTRING_LEN = 1     

# ======================================================
# Utilities
# ======================================================
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def count_existing_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

# ======================================================
# Example processing
# ======================================================
def process_example(example):
    code = example.get("func_code_string", "")
    doc_tokens = example.get("func_documentation_tokens", [])

    if not code or not code.strip():
        return []

    if not doc_tokens:
        return []

    # Reconstruct NL prompt
    prompt = " ".join(doc_tokens)
    if len(prompt) < MIN_DOCSTRING_LEN:
        return []

    try:
        tree = code_to_ast(code)
        tokens = ast_to_tokens(tree)
    except (SyntaxError, ValueError, RecursionError):
        return []

    if len(tokens) > MAX_AST_TOKENS:
        return []

    return [{
        "prompt": prompt,
        "ast_tokens": tokens,
    }]


# ======================================================
# Main
# ======================================================
def main():

    ensure_dir(OUTPUT_PATH)

    already_written = count_existing_lines(OUTPUT_PATH)
    if already_written >= MAX_SAMPLES:
        print(f"[INFO] {already_written} samples already exist. Nothing to do.")
        return

    print(f"[INFO] Resuming from {already_written} samples")

    dataset = load_dataset(
        DATASET_NAME,
        LANGUAGE,
        split=SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    written = already_written
    skipped = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset)):
            if i < already_written:
                continue

            results = process_example(example)

            if not results:
                skipped += 1
                continue

            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1

                if written >= MAX_SAMPLES:
                    break
    if written < MAX_SAMPLES:
        print("[WARN] Dataset exhausted before reaching MAX_SAMPLES")

    print("\n===== NL → AST BUILD SUMMARY =====")
    print(f"Written samples : {written}")
    print(f"Skipped samples : {skipped}")
    print(f"Output file     : {OUTPUT_PATH}")

# ======================================================
# Entry point
# ======================================================
if __name__ == "__main__":
    main()
