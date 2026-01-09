from collections import Counter
import json

INPUT_PATH = "data/processed/python_ast.jsonl"
OUTPUT_PATH = "data/processed/ast_vocab.json"

counter = Counter()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        counter.update(obj["ast_tokens"])

specials = ["<pad>", "<bos>", "<eos>"]
vocab = specials + sorted(counter)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print(f"[OK] Wrote {len(vocab)} AST tokens to {OUTPUT_PATH}")
