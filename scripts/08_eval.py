import os, sys, json, math, argparse
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.codec import tokens_to_ast, ast_to_code, ast_to_tokens
from ast_codec.tokenize_ast_sequence import encode_codec_tokens

OPEN = "<open>"
CLOSE = "<close>"


def bracket_balance_ok(tokens):
    return tokens.count(OPEN) == tokens.count(CLOSE)


def node_type_counter(tree):
    import ast
    c = Counter()
    for node in ast.walk(tree):
        c[node.__class__.__name__] += 1
    return c


def node_f1(gen_tree, ref_tree):
    gen = node_type_counter(gen_tree)
    ref = node_type_counter(ref_tree)

    tp = sum((gen & ref).values())
    fp = sum((gen - ref).values())
    fn = sum((ref - gen).values())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


def normalized_ted_approx(gen_tree, ref_tree):
    gen = node_type_counter(gen_tree)
    ref = node_type_counter(ref_tree)

    deletions = sum((ref - gen).values())
    insertions = sum((gen - ref).values())

    dist = deletions + insertions
    norm = dist / (sum(ref.values()) + 1e-9)
    return norm


def compute_ast_ppl(model, input_ids, labels):
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss.item()
    return math.exp(loss), loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--phase2_lora", required=True)

    parser.add_argument("--data", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--save_json", default=None)

    args = parser.parse_args()

    AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")
    if not os.path.exists(AST_VOCAB_PATH):
        raise FileNotFoundError(f"Missing ast_vocab.json: {AST_VOCAB_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
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
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False

    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, args.phase2_lora, is_trainable=False)
    model.eval()

    ast_bos_id = tokenizer.convert_tokens_to_ids("<ast_bos>")
    ast_eos_id = tokenizer.convert_tokens_to_ids("<ast_eos>")

    if ast_bos_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer missing <ast_bos>")
    if ast_eos_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer missing <ast_eos>")

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" not in obj or "ast_tokens" not in obj:
                continue
            data.append(obj)
            if len(data) >= args.num_samples:
                break

    print(f"[INFO] Loaded {len(data)} eval samples")

    total = 0
    decode_ok = 0
    compile_ok = 0
    roundtrip_ok = 0

    ted_scores = []
    node_f1_scores = []
    ppl_scores = []

    saved_rows = []

    for ex in tqdm(data):
        total += 1

        prompt = ex["prompt"]
        ref_ast_tokens = ex["ast_tokens"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + [ast_bos_id]], dtype=torch.long).to(model.device)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=ast_eos_id,
            )

        gen_seq = gen_out[0].tolist()
        gen_ast_ids = gen_seq[len(prompt_ids) + 1:]

        if ast_eos_id in gen_ast_ids:
            eos_idx = gen_ast_ids.index(ast_eos_id)
            gen_ast_ids = gen_ast_ids[:eos_idx]

        gen_ast_tokens = tokenizer.convert_ids_to_tokens(gen_ast_ids)

        if len(gen_ast_tokens) == 0:
            continue

        decoded_ok_flag = False
        decoded_tree = None

        if bracket_balance_ok(gen_ast_tokens):
            try:
                decoded_tree = tokens_to_ast(gen_ast_tokens)
                decoded_ok_flag = True
            except Exception:
                decoded_ok_flag = False

        if decoded_ok_flag:
            decode_ok += 1

        rt_ok = False
        if decoded_ok_flag:
            try:
                rt_tokens = ast_to_tokens(decoded_tree)
                rt_ok = (rt_tokens == gen_ast_tokens)
            except Exception:
                rt_ok = False

        if rt_ok:
            roundtrip_ok += 1

        comp_ok = False
        if decoded_ok_flag:
            try:
                import ast
                ast.fix_missing_locations(decoded_tree)
                compile(decoded_tree, "<gen>", "exec")
                comp_ok = True
            except Exception:
                comp_ok = False

        if comp_ok:
            compile_ok += 1

        ted = None
        f1 = None

        try:
            ref_tree = tokens_to_ast(ref_ast_tokens)
        except Exception:
            ref_tree = None

        if decoded_ok_flag and ref_tree is not None:
            ted = normalized_ted_approx(decoded_tree, ref_tree)
            _, _, f1 = node_f1(decoded_tree, ref_tree)

            ted_scores.append(ted)
            node_f1_scores.append(f1)

        try:
            ref_ast_ids = encode_codec_tokens(ref_ast_tokens, tokenizer)
        except Exception:
            ref_ast_ids = []

        full_ids = prompt_ids + [ast_bos_id] + ref_ast_ids + [ast_eos_id]

        labels = [-100] * len(prompt_ids)
        labels += [-100]
        labels += ref_ast_ids
        labels += [ast_eos_id]

        full_ids = torch.tensor([full_ids], dtype=torch.long).to(model.device)
        labels = torch.tensor([labels], dtype=torch.long).to(model.device)

        ppl, loss = compute_ast_ppl(model, full_ids, labels)
        ppl_scores.append(ppl)

        if args.save_json:
            row = {
                "prompt": prompt,
                "decoded_ok": decoded_ok_flag,
                "compile_ok": comp_ok,
                "roundtrip_ok": rt_ok,
                "ppl": ppl,
                "loss": loss,
                "ted": ted,
                "node_f1": f1,
            }
            if decoded_ok_flag:
                try:
                    row["generated_code"] = ast_to_code(decoded_tree)
                except Exception:
                    row["generated_code"] = None

            saved_rows.append(row)

    decode_rate = 100.0 * decode_ok / total if total else 0.0
    compile_rate = 100.0 * compile_ok / total if total else 0.0
    roundtrip_rate = 100.0 * roundtrip_ok / total if total else 0.0

    avg_ted = sum(ted_scores) / len(ted_scores) if ted_scores else None
    avg_f1 = sum(node_f1_scores) / len(node_f1_scores) if node_f1_scores else None
    avg_ppl = sum(ppl_scores) / len(ppl_scores) if ppl_scores else None

    print("\n==================== RESULTS ====================")
    print(f"Samples evaluated : {total}")
    print(f"Decode%           : {decode_rate:.2f}")
    print(f"Compile%          : {compile_rate:.2f}")
    print(f"RoundTrip%        : {roundtrip_rate:.2f}")
    print(f"TED↓              : {avg_ted:.4f}" if avg_ted is not None else "TED↓              : N/A")
    print(f"NodeF1↑           : {avg_f1:.4f}" if avg_f1 is not None else "NodeF1↑           : N/A")
    print(f"PPL↓              : {avg_ppl:.4f}" if avg_ppl is not None else "PPL↓              : N/A")
    print("=================================================\n")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(saved_rows, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved detailed results to {args.save_json}")


if __name__ == "__main__":
    main()
