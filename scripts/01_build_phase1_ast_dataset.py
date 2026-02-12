# scripts/01_build_phase1_ast_dataset.py
import os, sys, json, ast
from tqdm import tqdm
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.codec import ast_to_tokens

OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase1_ast.jsonl")

DATASET_NAME = "bigcode/the-stack"
DATA_DIR = "data/python"
SPLIT = "train"

MAX_SAMPLES = 20000
MAX_AST_TOKENS = 2048


def extract_function_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(body=[node], type_ignores=[])


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        split=SPLIT,
        streaming=True,
    )

    written = 0
    skipped = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset):
            if written >= MAX_SAMPLES:
                break

            code = ex.get("content", "")
            if not code.strip():
                skipped += 1
                continue

            try:
                mod = ast.parse(code)
            except Exception:
                skipped += 1
                continue

            for fn_mod in extract_function_modules(mod):
                try:
                    toks = ast_to_tokens(fn_mod)
                except Exception:
                    skipped += 1
                    continue

                if len(toks) > MAX_AST_TOKENS:
                    skipped += 1
                    continue

                f.write(json.dumps({"ast_tokens": toks}, ensure_ascii=False) + "\n")
                written += 1

                if written >= MAX_SAMPLES:
                    break

    print(f"[OK] Wrote {written} AST samples to {OUT_PATH}")
    print(f"[INFO] Skipped {skipped}")


if __name__ == "__main__":
    main()
