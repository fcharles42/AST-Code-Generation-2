import os
import sys
import json
import math
import argparse
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------
# Project root import setup
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ast_codec.tokenizer import ASTTokenizer
from ast_codec.python_ast import tokens_to_ast, ast_to_tokens, ast_to_code

# -------------------------
# Helpers
# -------------------------
OPEN = "<open>"
CLOSE = "<close>"
INDENT = "<indent>"
DEDENT = "<dedent>"


def indent_balance_ok(tokens):
    """Checks block correctness: never negative depth, ends at depth 0."""
    depth = 0
    for t in tokens:
        if t == INDENT:
            depth += 1
        elif t == DEDENT:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def bracket_balance_ok(tokens):
    """Checks open/close count match."""
    return tokens.count(OPEN) == tokens.count(CLOSE)


def node_type_counter(tree):
    """Returns Counter of AST node type names from an ast.AST object."""
    import ast
    c = Counter()

    for node in ast.walk(tree):
        c[node.__class__.__name__] += 1

    return c


def node_f1(gen_tree, ref_tree):
    """Node-type multiset F1."""
    gen = node_type_counter(gen_tree)
    ref = node_type_counter(ref_tree)

    # true positives = overlap count
    tp = sum((gen & ref).values())
    fp = sum((gen - ref).values())
    fn = sum((ref - gen).values())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


def normalized_ted_approx(gen_tree, ref_tree):
    """
    Approximate Tree Edit Distance using node multiset distance.

    True TED requires a full tree alignment algorithm (Zhang-Shasha).
    This approximation is still meaningful for early results.

    distance = insertions + deletions
    normalized by ref size.
    """
    gen = node_type_counter(gen_tree)
    ref = node_type_counter(ref_tree)

    deletions = sum((ref - gen).values())
    insertions = sum((gen - ref).values())

    dist = deletions + insertions
    norm = dist / (sum(ref.values()) + 1e-9)
    return norm


def compute_ast_ppl(model, input_ids, labels):
    """
    Computes perplexity given labels with -100 masking.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss.item()
    return math.exp(loss), loss


# -------------------------
# Main evaluation
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint", required=True, help="Phase2 LoRA checkpoint path")
    parser.add_argument("--ast_vocab", default="data/processed/ast_vocab.json")
    parser.add_argument("--data", default="data/processed/nl_ast_pairs.jsonl")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_json", default=None, help="Optional output JSON file")
    args = parser.parse_args()

    # -------------------------
    # Load tokenizers
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ast_tokenizer = ASTTokenizer(args.ast_vocab)

    BASE_VOCAB_SIZE = len(tokenizer)
    AST_OFFSET = BASE_VOCAB_SIZE

    AST_BOS = ast_tokenizer.bos_id + AST_OFFSET
    AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

    vocab = ast_tokenizer.token_to_id
    inv_vocab = ast_tokenizer.id_to_token

    # -------------------------
    # Load model + LoRA
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.config.use_cache = False

    # Resize embeddings to include AST vocab
    model.resize_token_embeddings(BASE_VOCAB_SIZE + len(ast_tokenizer))

    # Load phase2 LoRA
    model = PeftModel.from_pretrained(
        model,
        args.checkpoint,
        is_trainable=False,
    )

    model.eval()

    # -------------------------
    # Load dataset
    # -------------------------
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" not in obj or "ast_tokens" not in obj:
                continue
            data.append(obj)
            if len(data) >= args.num_samples:
                break

    print(f"[INFO] Loaded {len(data)} evaluation samples")

    # -------------------------
    # Metric accumulators
    # -------------------------
    total = 0

    decode_ok = 0
    compile_ok = 0
    roundtrip_ok = 0

    ted_scores = []
    node_f1_scores = []
    ppl_scores = []

    saved_rows = []

    # -------------------------
    # Evaluate loop
    # -------------------------
    for ex in tqdm(data):
        total += 1

        prompt = ex["prompt"]
        ref_ast_tokens = ex["ast_tokens"]

        # -------------------------
        # Encode prompt
        # -------------------------
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        input_ids = torch.tensor([prompt_ids + [AST_BOS]], dtype=torch.long)
        input_ids = input_ids.to(model.device)

        # -------------------------
        # Generate AST ids
        # -------------------------
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy for stable evaluation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=AST_EOS,
            )

        gen_seq = gen_out[0].tolist()

        # Remove prompt prefix
        gen_ast_ids = gen_seq[len(prompt_ids) + 1 :]  # after AST_BOS

        # Cut at EOS if present
        if AST_EOS in gen_ast_ids:
            eos_idx = gen_ast_ids.index(AST_EOS)
            gen_ast_ids = gen_ast_ids[:eos_idx]

        # Convert ids -> tokens
        gen_ast_tokens = []
        for tid in gen_ast_ids:
            raw_id = tid - AST_OFFSET
            if raw_id in inv_vocab:
                gen_ast_tokens.append(inv_vocab[raw_id])

        # -------------------------
        # Structural checks
        # -------------------------
        valid_brackets = bracket_balance_ok(gen_ast_tokens)
        valid_indent = indent_balance_ok(gen_ast_tokens)

        decoded_tree = None
        decoded_ok = False

        if valid_brackets and valid_indent:
            try:
                decoded_tree = tokens_to_ast(gen_ast_tokens)
                decoded_ok = True
            except Exception:
                decoded_ok = False

        if decoded_ok:
            decode_ok += 1

        # -------------------------
        # Round-trip check
        # -------------------------
        rt_ok = False
        if decoded_ok:
            try:
                rt_tokens = ast_to_tokens(decoded_tree)
                rt_ok = (rt_tokens == gen_ast_tokens)
            except Exception:
                rt_ok = False

        if rt_ok:
            roundtrip_ok += 1

        # -------------------------
        # Compilation check
        # -------------------------
        comp_ok = False
        if decoded_ok:
            try:
                import ast
                ast.fix_missing_locations(decoded_tree)
                compile(decoded_tree, "<gen>", "exec")
                comp_ok = True
            except Exception:
                comp_ok = False

        if comp_ok:
            compile_ok += 1

        # -------------------------
        # Similarity metrics (if decode ok)
        # -------------------------
        ted = None
        f1 = None

        ref_tree = None
        try:
            ref_tree = tokens_to_ast(ref_ast_tokens)
        except Exception:
            ref_tree = None

        if decoded_ok and ref_tree is not None:
            ted = normalized_ted_approx(decoded_tree, ref_tree)
            _, _, f1 = node_f1(decoded_tree, ref_tree)

            ted_scores.append(ted)
            node_f1_scores.append(f1)

        # -------------------------
        # PPL on reference AST (teacher forcing)
        # -------------------------
        # Build full input_ids = prompt + BOS + ref_ast + EOS
        ref_ast_ids = []
        for tok in ref_ast_tokens:
            if tok in vocab:
                ref_ast_ids.append(vocab[tok] + AST_OFFSET)
            else:
                # unknown tokens shouldn't happen, but safe fallback
                ref_ast_ids.append(ast_tokenizer.pad_id + AST_OFFSET)

        full_ids = prompt_ids + [AST_BOS] + ref_ast_ids + [AST_EOS]

        # labels masked on prompt + BOS
        labels = [-100] * len(prompt_ids)
        labels += [-100]
        labels += ref_ast_ids
        labels += [AST_EOS]

        full_ids = torch.tensor([full_ids], dtype=torch.long).to(model.device)
        labels = torch.tensor([labels], dtype=torch.long).to(model.device)

        ppl, loss = compute_ast_ppl(model, full_ids, labels)
        ppl_scores.append(ppl)

        # -------------------------
        # Save per-example row
        # -------------------------
        if args.save_json is not None:
            row = {
                "prompt": prompt,
                "decoded_ok": decoded_ok,
                "compile_ok": comp_ok,
                "roundtrip_ok": rt_ok,
                "ppl": ppl,
                "loss": loss,
                "ted": ted,
                "node_f1": f1,
            }

            if decoded_ok:
                try:
                    row["generated_code"] = ast_to_code(decoded_tree)
                except Exception:
                    row["generated_code"] = None

            saved_rows.append(row)

    # -------------------------
    # Aggregate results
    # -------------------------
    decode_rate = 100.0 * decode_ok / total
    compile_rate = 100.0 * compile_ok / total
    roundtrip_rate = 100.0 * roundtrip_ok / total

    avg_ted = sum(ted_scores) / len(ted_scores) if ted_scores else None
    avg_f1 = sum(node_f1_scores) / len(node_f1_scores) if node_f1_scores else None
    avg_ppl = sum(ppl_scores) / len(ppl_scores) if ppl_scores else None

    print("\n==================== RESULTS ====================")
    print(f"Samples evaluated : {total}")
    print(f"Decode%           : {decode_rate:.2f}")
    print(f"Compile%          : {compile_rate:.2f}")
    print(f"RoundTrip%        : {roundtrip_rate:.2f}")

    if avg_ted is not None:
        print(f"TED↓              : {avg_ted:.4f}")
    else:
        print("TED↓              : N/A")

    if avg_f1 is not None:
        print(f"NodeF1↑           : {avg_f1:.4f}")
    else:
        print("NodeF1↑           : N/A")

    if avg_ppl is not None:
        print(f"PPL↓              : {avg_ppl:.4f}")
    else:
        print("PPL↓              : N/A")

    print("=================================================\n")

    # -------------------------
    # Save JSON if requested
    # -------------------------
    if args.save_json is not None:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(saved_rows, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved per-example results to {args.save_json}")


if __name__ == "__main__":
    main()