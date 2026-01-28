import json
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ast_codec.tokenizer import ASTTokenizer

INPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/python_ast.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/python_ast_ids.jsonl")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data/processed/ast_vocab.json")

tokenizer = ASTTokenizer(VOCAB_PATH)

num_valid = 0
num_skipped = 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        try:
            obj = json.loads(line)
            if "ast_tokens" not in obj:
                num_skipped += 1
                continue

            ast_ids = tokenizer.encode(obj["ast_tokens"])
            fout.write(json.dumps({"ast_ids": ast_ids}) + "\n")
            num_valid += 1

        except Exception as e:
            num_skipped += 1
            print(f"[WARN] Skipping line {i}: {e}")

print(f"[INFO] Encoded dataset: {num_valid} samples (skipped {num_skipped})")
