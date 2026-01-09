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

with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
    for line in fin:
        obj = json.loads(line)

        ast_tokens = obj["ast_tokens"]
        ast_ids = tokenizer.encode(ast_tokens)

        out = {
            "ast_ids": ast_ids,
        }

        fout.write(json.dumps(out) + "\n")
