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
# HuggingFace stability (important for streaming)
# ======================================================
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_HTTP_RETRIES", "50")

# ======================================================
# Config
# ======================================================
OUTPUT_PATH = "data/processed/nl_ast_pairs.jsonl"

DATASET_NAME = "code-search-net/code_search_net"
LANGUAGE = "python"
SPLIT = "train"

MAX_SAMPLES = 50_000        # change if you want more
MAX_AST_TOKENS = 4096       # must match training MAX_SEQ_LEN
MIN_DOCSTRING_LEN = 10      # filter junk

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
# AST extraction (function-level)
# ======================================================
def extract_function_level_asts(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(
                body=[node],
                type_ignores=[]
            )

# ======================================================
# Example processing
# ======================================================
def process_example(example):
    """
    Returns list of (nl_prompt, ast_tokens)
    """
    docstring = example.get("docstring", "")
    code = example.get("code", "")

    if not docstring or len(docstring.strip()) < MIN_DOCSTRING_LEN:
        return []

    if not code.strip():
        return []

    outputs = []

    try:
        module_tree = code_to_ast(code)

        for fn_tree in extract_function_level_asts(module_tree):
            tokens = ast_to_tokens(fn_tree)

            if len(tokens) > MAX_AST_TOKENS:
                continue

            outputs.append({
                "prompt": docstring.strip(),
                "ast_tokens": tokens,
            })

    except Exception:
        return []

    return outputs

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
    )

    written = already_written
    skipped = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for example in tqdm(dataset, initial=already_written):
            if written >= MAX_SAMPLES:
                break

            results = process_example(example)

            if not results:
                skipped += 1
                continue

            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1

                if written >= MAX_SAMPLES:
                    break

    print("\n===== NL → AST BUILD SUMMARY =====")
    print(f"Written samples : {written}")
    print(f"Skipped samples : {skipped}")
    print(f"Output file     : {OUTPUT_PATH}")

# ======================================================
# Entry point
# ======================================================
if __name__ == "__main__":
    main()
