import os
import sys
import json
import ast
from tqdm import tqdm
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ast_codec.python_ast import code_to_ast, ast_to_tokens

# =====================
# HF streaming stability
# =====================
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_HTTP_RETRIES", "50")

# =====================
# Config
# =====================
OUTPUT_PATH = "data/processed/python_ast.jsonl"
MAX_SAMPLES = 20_000
DATASET_NAME = "bigcode/the-stack"
DATASET_DIR = "data/python"
SPLIT = "train"
MAX_AST_TOKENS = 2048

# =====================
# Utilities
# =====================
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def count_existing_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

# =====================
# Function-level AST extraction
# =====================
def extract_function_level_asts(tree: ast.AST):
    """
    Given a Module AST, yield Module-wrapped
    FunctionDef / AsyncFunctionDef ASTs.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(
                body=[node],
                type_ignores=[]
            )

# =====================
# Example processing
# =====================
def process_example(example):
    code = example.get("content", "")

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
                "ast_tokens": tokens
            })

    except Exception:
        return []

    return outputs

# =====================
# Main
# =====================
def main():
    ensure_dir(OUTPUT_PATH)

    already_written = count_existing_lines(OUTPUT_PATH)
    if already_written >= MAX_SAMPLES:
        print(f"[INFO] {already_written} samples already exist. Nothing to do.")
        return

    print(f"[INFO] Resuming from {already_written} samples")

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATASET_DIR,
        split=SPLIT,
        streaming=True,
    )

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        count = already_written

        for example in tqdm(dataset, initial=already_written):
            if count >= MAX_SAMPLES:
                break

            results = process_example(example)

            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1

                if count >= MAX_SAMPLES:
                    break

    print(f"[INFO] Finished. Total samples written: {count}")

# =====================
# Entry point
# =====================
if __name__ == "__main__":
    main()