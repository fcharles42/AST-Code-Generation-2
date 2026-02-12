# scripts/00_build_ast_vocab.py
import os, sys, json, ast

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ast_codec.codec import (
    OPEN, CLOSE, LIST_BEGIN, LIST_END,
    ID_BEGIN, ID_END,
    STR_BEGIN, STR_END,
    INT_BEGIN, INT_END,
    FLOAT_BEGIN, FLOAT_END,
    BOOL_BEGIN, BOOL_END,
    NONE,
)

AST_BOS = "<ast_bos>"
AST_EOS = "<ast_eos>"

OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    vocab = []

    # These are the only "special" tokens we control.
    vocab.extend([
        AST_BOS, AST_EOS,
        OPEN, CLOSE,
        LIST_BEGIN, LIST_END,

        ID_BEGIN, ID_END,

        STR_BEGIN, STR_END,
        INT_BEGIN, INT_END,
        FLOAT_BEGIN, FLOAT_END,
        BOOL_BEGIN, BOOL_END,

        NONE,
    ])

    # AST node class names
    node_names = []
    for name in dir(ast):
        obj = getattr(ast, name)
        if isinstance(obj, type) and issubclass(obj, ast.AST):
            node_names.append(name)
    node_names = sorted(set(node_names))

    # Field tokens
    field_names = set()
    for n in node_names:
        cls = getattr(ast, n)
        if hasattr(cls, "_fields") and cls._fields:
            for f in cls._fields:
                field_names.add(f)
    field_tokens = sorted([f"<field:{f}>" for f in field_names])

    # Operator tokens
    op_names = []
    for name in dir(ast):
        obj = getattr(ast, name)
        if isinstance(obj, type) and (
            issubclass(obj, ast.operator)
            or issubclass(obj, ast.unaryop)
            or issubclass(obj, ast.boolop)
            or issubclass(obj, ast.cmpop)
        ):
            op_names.append(name)
    op_tokens = sorted([f"<op:{x}>" for x in set(op_names)])

    # Context tokens
    ctx_names = []
    for name in dir(ast):
        obj = getattr(ast, name)
        if isinstance(obj, type) and issubclass(obj, ast.expr_context):
            ctx_names.append(name)
    ctx_tokens = sorted([f"<ctx:{x}>" for x in set(ctx_names)])

    vocab.extend(node_names)
    vocab.extend(field_tokens)
    vocab.extend(op_tokens)
    vocab.extend(ctx_tokens)

    # Remove duplicates while preserving order
    seen = set()
    final_vocab = []
    for t in vocab:
        if t not in seen:
            seen.add(t)
            final_vocab.append(t)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, indent=2)

    print(f"[OK] Wrote vocab with {len(final_vocab)} tokens to {OUT_PATH}")


if __name__ == "__main__":
    main()
