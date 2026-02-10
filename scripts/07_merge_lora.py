# scripts/07_merge_lora.py
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PHASE2_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase2_lora")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "merged")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PHASE2_DIR):
        raise FileNotFoundError(f"Missing Phase2 LoRA directory: {PHASE2_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(PHASE2_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
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
