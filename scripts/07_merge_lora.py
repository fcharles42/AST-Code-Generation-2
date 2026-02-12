import os, sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")

PHASE2_DIR = "/kaggle/input/phase2/kaggle/working/AST-Code-Generation-2/checkpoints/phase2_lora"
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "merged")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PHASE2_DIR):
        raise FileNotFoundError(f"Missing Phase2 LoRA directory: {PHASE2_DIR}")

    if not os.path.exists(AST_VOCAB_PATH):
        raise FileNotFoundError(f"Missing ast_vocab.json: {AST_VOCAB_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    with open(AST_VOCAB_PATH, "r", encoding="utf-8") as f:
        ast_vocab = json.load(f)

    added = tokenizer.add_tokens(ast_vocab, special_tokens=False)
    print("[INFO] Added AST tokens:", added)
    print("[INFO] Tokenizer size:", len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, PHASE2_DIR)
    merged = model.merge_and_unload()

    merged.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[OK] Saved merged model to {OUT_DIR}")


if __name__ == "__main__":
    main()
