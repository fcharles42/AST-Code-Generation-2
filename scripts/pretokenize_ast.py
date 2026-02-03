import json, os, torch, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoTokenizer
from ast_codec.tokenizer import ASTTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "/kaggle/working/data/processed/nl_ast_pairs.jsonl"
OUT_PATH = "/kaggle/working/data/processed/tokenized.pt"
AST_VOCAB_PATH = "/kaggle/input/astcodes/data/processed/ast_vocab.json"

MAX_SEQ_LEN = 512

base_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, use_fast=True
)
base_tokenizer.pad_token = base_tokenizer.eos_token

ast_tokenizer = ASTTokenizer(AST_VOCAB_PATH)

BASE_VOCAB_SIZE = len(base_tokenizer)
AST_OFFSET = BASE_VOCAB_SIZE
AST_BOS = ast_tokenizer.bos_id + AST_OFFSET
AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

vocab = ast_tokenizer.token_to_id
PAD_ID = ast_tokenizer.pad_id
CANONICAL_ID = next(vocab[t] for t in ast_tokenizer.vocab if t.startswith("<id:"))

examples = []

with open(DATA_PATH) as f:
    for line in f:
        ex = json.loads(line)

        prompt_ids = base_tokenizer.encode(
            ex["prompt"], add_special_tokens=False
        )

        ast_ids = []
        for tok in ex["ast_tokens"]:
            if tok in vocab:
                ast_ids.append(vocab[tok])
            elif tok.startswith("<id:"):
                ast_ids.append(CANONICAL_ID)
            else:
                ast_ids.append(PAD_ID)

        ast_ids = [i + AST_OFFSET for i in ast_ids]

        ids = (prompt_ids + [AST_BOS] + ast_ids + [AST_EOS])[:MAX_SEQ_LEN]
        lbl = (
            [-100] * len(prompt_ids)
            + [-100]
            + ast_ids
            + [AST_EOS]
        )[:MAX_SEQ_LEN]

        examples.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
        })

torch.save(examples, OUT_PATH)
print(f"Saved {len(examples)} examples to {OUT_PATH}")
