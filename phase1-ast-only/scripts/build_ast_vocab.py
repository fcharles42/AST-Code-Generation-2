from collections import Counter
import json

INPUT_PATH = "data/processed/python_ast.jsonl"
OUTPUT_PATH = "data/processed/ast_vocab.json"

counter = Counter()
num_valid = 0
num_skipped = 0

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            obj = json.loads(line)
            if "ast_tokens" not in obj:
                num_skipped += 1
                continue

            counter.update(obj["ast_tokens"])
            num_valid += 1

        except Exception:
            num_skipped += 1

specials = ["<pad>", "<bos>", "<eos>"]
vocab = specials + sorted(counter)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print(
    f"[OK] Wrote {len(vocab)} AST tokens "
    f"(from {num_valid} samples, skipped {num_skipped}) "
    f"to {OUTPUT_PATH}"
)
