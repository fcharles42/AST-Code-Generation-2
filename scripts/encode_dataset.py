import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ast_codec.tokenizer import ASTTokenizer

INPUT_PATH = "data/processed/python_ast.jsonl"
OUTPUT_PATH = "data/processed/python_ast_ids.jsonl"
VOCAB_PATH = "data/processed/ast_vocab.json"

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
