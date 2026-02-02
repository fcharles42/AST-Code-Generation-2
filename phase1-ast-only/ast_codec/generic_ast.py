"""
Generic Tree-sitter based AST tokenizer.

This module mirrors the tokenization style in python_ast.py but works across
multiple languages by using Tree-sitter. It emits structural tokens and field
markers to capture the parse tree shape in a language-agnostic way.

Dependencies:
- pip install tree_sitter==0.21.3
Note! Works only with tree-sitter 0.21.3
- pip install tree_sitter_languages

If tree_sitter_languages is not available, you can provide an environment
variable TREE_SITTER_LIB pointing to a compiled language bundle (a .so/.dylib)
that contains the languages you need. See Language.build_library in Tree-sitter docs.

Usage examples:
- python generic_ast.py --lang js test.js

"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

try:
    # Core Tree-sitter
    from tree_sitter import Language, Parser, Node
except Exception as e:  # pragma: no cover - helpful message if missing
    raise RuntimeError(
        "tree_sitter is required. Install with: pip install tree_sitter"
    ) from e


# Structural tokens
OPEN = "<open>"
CLOSE = "<close>"
INDENT = "<indent>"
DEDENT = "<dedent>"
EOS = "<eos>"

# Literal tokens (superset to be tolerant across languages)
LITERAL_TOKENS = {"<STR>", "<INT>", "<FLOAT>", "<BOOL>", "<NULL>", "<CHAR>", "<NUM>"}

class FlatAlphaRenamer:
    """Flat renamer: every distinct identifier maps to id_<n> per file."""

    def __init__(self):
        self.map: Dict[str, str] = {}
        self.counter: int = 0

    def _new(self) -> str:
        name = f"id_{self.counter}"
        self.counter += 1
        return name

    def rename(self, raw: str) -> str:
        if raw not in self.map:
            self.map[raw] = self._new()
        return self.map[raw]


def id_token(name: str) -> str:
    return f"<id:{name}>"


# =====================
# Language loading
# =====================
_LANG_ALIASES= {
    # Common aliases and normalization
    "js": "javascript",
    "ts": "typescript",
    "c++": "cpp",
    "csharp": "c_sharp",
    "c#": "c_sharp",
    "objc": "objective_c",
}


def _normalize_lang(name: str) -> str:
    name = name.strip().lower()
    return _LANG_ALIASES.get(name, name)


def load_language(lang_name: str) -> Language:
    """Load a Tree-sitter Language object for the given language name.

    Tries the following strategies in order:
    1) tree_sitter_languages.get_language (if installed)
    2) Load from a user-provided compiled bundle via TREE_SITTER_LIB env var
       using Language(path, name)
    """

    lang_name = _normalize_lang(lang_name)

    tried: List[str] = []
    try:
        from tree_sitter_languages import get_language  # type: ignore
        candidates = [lang_name]
        # Try also the alias-normalized form and common variants
        variants = {
            "javascript": ["javascript", "js", "ecmascript"],
            "typescript": ["typescript", "ts", "tsx"],
            "python": ["python", "py"],
            "cpp": ["cpp", "c++", "cxx"],
            "c": ["c"],
            "rust": ["rust", "rs"],
            "java": ["java"],
            "go": ["go", "golang"],
            "ruby": ["ruby", "rb"],
            "php": ["php"],
            "c_sharp": ["c_sharp", "csharp", "c#"],
            "swift": ["swift"],
            "kotlin": ["kotlin", "kt"],
            "scala": ["scala"],
            "haskell": ["haskell", "hs"],
            "lua": ["lua"],
            "zig": ["zig"],
            "rust": ["rust", "rs"],
        }
        base = _normalize_lang(lang_name)
        candidates.extend(variants.get(base, []))
        last_err: Optional[Exception] = None
        for cand in OrderedDict((c, None) for c in candidates).keys():
            tried.append(cand)
            try:
                return get_language(cand)
            except Exception as e:  # try next candidate
                last_err = e
                continue
        if last_err is not None:
            pass
    except Exception:
        pass # tree_sitter_languages not installed or failed to import; try bundle

    # Strategy 2: User-provided compiled bundle (.so/.dylib)
    bundle_path = os.environ.get("TREE_SITTER_LIB")
    if not bundle_path:
        tried_msg = f" Tried via tree_sitter_languages with candidates: {tried}." if tried else ""
        raise RuntimeError(
            "Could not load language. Install 'tree_sitter_languages' or set TREE_SITTER_LIB "
            "to a compiled languages bundle." + tried_msg
        )

    try:
        return Language(bundle_path, lang_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load language '{lang_name}' from bundle at {bundle_path}."
        ) from e

# =====================

# Node classification helpers
def _bytes_slice(src: bytes, node: Node) -> bytes:
    return src[node.start_byte : node.end_byte]


def _is_boolean_node(node: Node, text: str) -> bool:
    t = node.type
    if t in {"true", "false", "boolean", "boolean_literal"}:
        return True
    return text in {"true", "false"}


def _is_nullish_node(node: Node, text: str) -> bool:
    return node.type in {"null", "nil", "none", "nullptr", "undefined"} or text in {
        "null",
        "nil",
        "None",
        "nullptr",
        "undefined",
    }


def _is_string_node(node: Node) -> bool:
    t = node.type
    return (
        "string" in t
        or t in {"string_literal", "raw_string_literal", "interpreted_string_literal"}
    )


def _is_char_node(node: Node) -> bool:
    t = node.type
    return t in {"char", "character", "character_literal"}


def _is_number_node(node: Node) -> bool:
    t = node.type
    return (
        "number" in t
        or "numeric" in t
        or t.endswith("_literal")
        or t in {"int_literal", "float_literal", "decimal_literal", "integer"}
    )


def _is_identifier_node(node: Node) -> bool:
    t = node.type
    return (
        "identifier" in t
        or t
        in {
            "name",
            "label",
            "symbol",
            "scoped_identifier",
            "namespace_identifier",
            "macro_identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
            "shorthand_property_identifier",
        }
    )


# =====================

# Tree serialization
def _emit_field_block(out: List[str], field_name: str, emit_children) -> None:
    out.append(f"<field:{field_name}>")
    out.append(INDENT)
    emit_children()
    out.append(DEDENT)


def serialize_node(node: Node, src: bytes, out: List[str], renamer: FlatAlphaRenamer) -> None:
    """Serialize a Tree-sitter node to tokens.

    - Emits node.type, then <open>.
    - If the node is a leaf literal/identifier, records a standardized value
      under <field:value> and closes.
    - Otherwise, groups named children by field name (if any) and serializes.
      Unfielded named children are grouped under <field:children>.
    """

    if not node.is_named:
        return
    out.append(node.type)
    out.append(OPEN)

    # Leaf handling: no named children
    if node.named_child_count == 0:
        text = _bytes_slice(src, node).decode("utf-8", errors="replace").strip()

        if _is_identifier_node(node):
            renamed = renamer.rename(text) if text else renamer.rename(node.type)
            _emit_field_block(out, "value", lambda: out.append(id_token(renamed)))

        elif _is_boolean_node(node, text):
            _emit_field_block(out, "value", lambda: out.append("<BOOL>"))

        elif _is_nullish_node(node, text):
            _emit_field_block(out, "value", lambda: out.append("<NULL>"))

        elif _is_string_node(node):
            _emit_field_block(out, "value", lambda: out.append("<STR>"))

        elif _is_char_node(node):
            _emit_field_block(out, "value", lambda: out.append("<CHAR>"))

        elif _is_number_node(node):
            # Optionally distinguish int/float based on text; keep generic here
            _emit_field_block(out, "value", lambda: out.append("<NUM>"))

        else:
            # For unknown leaf named nodes, emit the raw type as structure with no value
            pass

        out.append(CLOSE)
        return

    # Non-leaf: group named children by field name (preserving encounter order)
    groups: "OrderedDict[str, List[Node]]" = OrderedDict()
    order: List[str] = []
    for i in range(node.child_count):
        ch = node.child(i)
        if not ch.is_named:
            continue
        fname = node.field_name_for_child(i)
        key = fname if fname is not None else "children"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ch)

    # Emit groups in encounter order
    for key in order:
        children = groups[key]
        _emit_field_block(
            out,
            key,
            lambda chs=children: [serialize_node(c, src, out, renamer) for c in chs],
        )

    out.append(CLOSE)


def tree_to_tokens(tree, source_bytes: bytes) -> List[str]:
    renamer = FlatAlphaRenamer()
    out: List[str] = []
    serialize_node(tree.root_node, source_bytes, out, renamer)
    return out


def code_to_tokens(lang_name: str, code: str) -> List[str]:
    # First attempt: use tree_sitter_languages.get_parser if available
    parser: Optional[Parser] = None
    try:
        from tree_sitter_languages import get_parser as _get_parser  # type: ignore

        # Try both the provided name and normalized alias
        tried = []
        for cand in OrderedDict((c, None) for c in [lang_name, _normalize_lang(lang_name)]).keys():
            tried.append(cand)
            try:
                parser = _get_parser(cand)
                break
            except Exception:
                continue
    except Exception:
        parser = None

    if parser is None:
        # Fallback: manual Language load and parser wiring
        lang = load_language(lang_name)
        parser = Parser()
        parser.set_language(lang)

    source = code.encode("utf-8")
    tree = parser.parse(source)
    return tree_to_tokens(tree, source)


# CLI =============

def _read_file(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generic Tree-sitter AST tokenizer")
    p.add_argument("path", nargs="?", help="Source file path (reads stdin if omitted)")
    p.add_argument("--lang", required=True, help="Language name (e.g. javascript, python, cpp, rust, java)")
    p.add_argument("--print-counts", action="store_true", help="Print basic sanity counts")
    args = p.parse_args(argv)

    if args.path:
        code = _read_file(args.path)
    else:
        code = sys.stdin.read()

    tokens = code_to_tokens(args.lang, code)

    if args.print_counts:
        print(f"opens={tokens.count(OPEN)} closes={tokens.count(CLOSE)}", file=sys.stderr)
        print(f"indents={tokens.count(INDENT)} dedents={tokens.count(DEDENT)}", file=sys.stderr)
        if tokens:
            print(f"first={tokens[0]} last={tokens[-1]}", file=sys.stderr)

    for t in tokens:
        print(t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
