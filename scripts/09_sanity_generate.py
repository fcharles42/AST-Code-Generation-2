import os, sys, json, ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.codec import tokens_to_ast, ast_to_code

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")
PHASE2_LORA = "/kaggle/input/phase2/kaggle/working/AST-Code-Generation-2/checkpoints/phase2_lora"

OUT_PATH = os.path.join(REPO_ROOT, "results", "samples", "sample_generations.json")

AST_BOS = "<ast_bos>"
AST_EOS = "<ast_eos>"

PROMPTS = [
    "Return the sum of two integers.",
    "Check if a number is prime.",
    "Sort a list of integers using bubble sort.",
    "Return the factorial of n using recursion.",
]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if not os.path.exists(PHASE2_LORA):
        raise FileNotFoundError(f"Missing phase2_lora directory: {PHASE2_LORA}")

    if not os.path.exists(AST_VOCAB_PATH):
        raise FileNotFoundError(f"Missing ast_vocab.json: {AST_VOCAB_PATH}")

    # ALWAYS rebuild tokenizer from base model
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
    print("[INFO] Tokenizer vocab size:", len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, PHASE2_LORA, is_trainable=False)
    model.eval()

    ast_bos_id = tokenizer.convert_tokens_to_ids(AST_BOS)
    ast_eos_id = tokenizer.convert_tokens_to_ids(AST_EOS)

    if ast_bos_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer missing <ast_bos>")
    if ast_eos_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer missing <ast_eos>")

    outputs = []

    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + [ast_bos_id]], dtype=torch.long).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=ast_eos_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = out[0].tolist()
        gen_ast_ids = gen_ids[len(prompt_ids) + 1:]

        if ast_eos_id in gen_ast_ids:
            gen_ast_ids = gen_ast_ids[: gen_ast_ids.index(ast_eos_id)]

        gen_tokens = tokenizer.convert_ids_to_tokens(gen_ast_ids)

        row = {"prompt": prompt, "tokens": gen_tokens}

        try:
            tree = tokens_to_ast(gen_tokens)
            ast.fix_missing_locations(tree)
            compile(tree, "<gen>", "exec")

            row["decoded_ok"] = True
            row["code"] = ast_to_code(tree)

        except Exception as e:
            row["decoded_ok"] = False
            row["error"] = str(e)
            row["code"] = None

        outputs.append(row)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved sanity generations to {OUT_PATH}")


if __name__ == "__main__":
    main()
