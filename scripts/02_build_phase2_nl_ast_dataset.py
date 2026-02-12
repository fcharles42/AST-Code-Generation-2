# scripts/02_build_phase2_nl_ast_dataset.py
import os, sys, json, ast
from tqdm import tqdm
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.codec import ast_to_tokens

OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase2_nl_ast.jsonl")

DATASET_NAME = "code_search_net"
LANG = "python"
SPLIT = "train"

MAX_SAMPLES = 10000
MAX_AST_TOKENS = 2048
MIN_PROMPT_LEN = 10


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        LANG,
        split=SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    written = 0
    skipped = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset):
            if written >= MAX_SAMPLES:
                break

            code = ex.get("func_code_string", "")
            doc_tokens = ex.get("func_documentation_tokens", [])

            if not code.strip() or not doc_tokens:
                skipped += 1
                continue

            prompt = " ".join(doc_tokens).strip()
            if len(prompt) < MIN_PROMPT_LEN:
                skipped += 1
                continue

            try:
                tree = ast.parse(code)
                toks = ast_to_tokens(tree)
            except Exception:
                skipped += 1
                continue

            if len(toks) > MAX_AST_TOKENS:
                skipped += 1
                continue

            row = {"prompt": prompt, "ast_tokens": toks}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Wrote {written} NL→AST samples to {OUT_PATH}")
    print(f"[INFO] Skipped {skipped}")


if __name__ == "__main__":
    main()
