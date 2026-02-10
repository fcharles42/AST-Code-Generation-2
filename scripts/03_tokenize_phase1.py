# scripts/03_tokenize_phase1.py
import os, sys, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.tokenize_ast_sequence import encode_codec_tokens

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")
IN_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase1_ast.jsonl")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase1_tokenized.pt")

MAX_SEQ_LEN = 2048


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    with open(AST_VOCAB_PATH, "r", encoding="utf-8") as f:
        ast_vocab = json.load(f)

    added = tokenizer.add_tokens(ast_vocab, special_tokens=False)
    print("[INFO] Added AST tokens:", added)
    print("[INFO] Tokenizer size:", len(tokenizer))

    ast_bos_id = tokenizer.convert_tokens_to_ids("<ast_bos>")
    ast_eos_id = tokenizer.convert_tokens_to_ids("<ast_eos>")

    if ast_bos_id == tokenizer.unk_token_id:
        raise ValueError("Missing <ast_bos> in tokenizer")
    if ast_eos_id == tokenizer.unk_token_id:
        raise ValueError("Missing <ast_eos> in tokenizer")

    examples = []
    skipped = 0

    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            toks = obj["ast_tokens"]

            try:
                ast_ids = encode_codec_tokens(toks, tokenizer)
            except Exception:
                skipped += 1
                continue

            # Full unshifted sequence
            full = [ast_bos_id] + ast_ids + [ast_eos_id]

            # Truncate but preserve EOS
            if len(full) > MAX_SEQ_LEN:
                full = full[:MAX_SEQ_LEN]
                full[-1] = ast_eos_id

            input_ids = full
            labels = full 

            examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    torch.save(examples, OUT_PATH)
    print(f"[OK] Saved {len(examples)} examples to {OUT_PATH}")
    print(f"[INFO] Skipped {skipped} invalid examples")


if __name__ == "__main__":
    main()
