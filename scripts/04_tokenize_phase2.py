# scripts/04_tokenize_phase2.py
import os, sys, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.tokenize_ast_sequence import encode_codec_tokens

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")
IN_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase2_nl_ast.jsonl")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase2_tokenized.pt")

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

            prompt = obj["prompt"]
            ast_tokens = obj["ast_tokens"]

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            try:
                ast_ids = encode_codec_tokens(ast_tokens, tokenizer)
            except Exception:
                skipped += 1
                continue

            # Full unshifted sequence
            full = prompt_ids + [ast_bos_id] + ast_ids + [ast_eos_id]

            # Truncate but preserve EOS
            if len(full) > MAX_SEQ_LEN:
                full = full[:MAX_SEQ_LEN]
                full[-1] = ast_eos_id

            # Labels must align with full sequence length
            labels = [-100] * len(prompt_ids)
            labels += [-100]         # mask ast_bos
            labels += ast_ids
            labels += [ast_eos_id]

            # Truncate labels to match full
            labels = labels[:len(full)]

            # If labels ended shorter (rare edge case), pad with -100
            if len(labels) < len(full):
                labels += [-100] * (len(full) - len(labels))

            input_ids = full
            # model.forward() will shift internally
            examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    torch.save(examples, OUT_PATH)
    print(f"[OK] Saved {len(examples)} examples to {OUT_PATH}")
    print(f"[INFO] Skipped {skipped} invalid examples")


if __name__ == "__main__":
    main()
