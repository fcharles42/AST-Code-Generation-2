# ast_codec/codec.py
import ast
import base64
from typing import List, Dict, Any

# ============================
# Core structural tokens
# ============================
OPEN = "<open>"
CLOSE = "<close>"

LIST_BEGIN = "<list_begin>"
LIST_END = "<list_end>"

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

# ============================
# Payload wrapper tokens
# ============================
ID_BEGIN = "<id>"
ID_END = "</id>"

STR_BEGIN = "<lit_str>"
STR_END = "</lit_str>"

INT_BEGIN = "<lit_int>"
INT_END = "</lit_int>"

FLOAT_BEGIN = "<lit_float>"
FLOAT_END = "</lit_float>"

BOOL_BEGIN = "<lit_bool>"
BOOL_END = "</lit_bool>"

NONE = "<lit_none>"

# ============================
# Helper token formats
# ============================
def field_token(name: str) -> str:
    return f"<field:{name}>"


def ctx_token(ctx: ast.expr_context) -> str:
    return f"<ctx:{ctx.__class__.__name__}>"


def op_token(op) -> str:
    return f"<op:{op.__class__.__name__}>"


def is_ctx_token(tok: str) -> bool:
    return tok.startswith("<ctx:") and tok.endswith(">")


def is_op_token(tok: str) -> bool:
    return tok.startswith("<op:") and tok.endswith(">")


def parse_ctx(tok: str):
    name = tok[len("<ctx:"):-1]
    return getattr(ast, name)()


def parse_op(tok: str):
    name = tok[len("<op:"):-1]
    return getattr(ast, name)()


# ============================
# Base64 payload encoding
# ============================
def encode_payload(text: str) -> str:
    """
    Encodes arbitrary string into whitespace-safe base64.
    """
    b = text.encode("utf-8", errors="replace")
    return base64.b64encode(b).decode("ascii")


def decode_payload(payload: str) -> str:
    b = base64.b64decode(payload.encode("ascii"))
    return b.decode("utf-8", errors="replace")


# ============================
# Literal + identifier handling
# ============================
def _serialize_literal(value) -> List[str]:
    if value is None:
        return [NONE]

    if isinstance(value, bool):
        return [BOOL_BEGIN, "True" if value else "False", BOOL_END]

    if isinstance(value, int):
        return [INT_BEGIN, str(value), INT_END]

    if isinstance(value, float):
        # repr() gives "nan", "inf", "-inf" properly
        return [FLOAT_BEGIN, repr(value), FLOAT_END]

    if isinstance(value, str):
        return [STR_BEGIN, encode_payload(value), STR_END]

    # fallback
    return [STR_BEGIN, encode_payload(str(value)), STR_END]


def _serialize_id(name: str) -> List[str]:
    return [ID_BEGIN, encode_payload(name), ID_END]


# ============================
# AST -> TOKENS
# ============================
def ast_to_tokens(tree: ast.AST) -> List[str]:
    tokens: List[str] = []

    def visit(node: ast.AST):
        tokens.append(node.__class__.__name__)
        tokens.append(OPEN)

        for field in node._fields or []:
            tokens.append(field_token(field))
            value = getattr(node, field, None)

            if isinstance(value, list):
                tokens.append(LIST_BEGIN)
                for item in value:
                    if isinstance(item, ast.AST):
                        visit(item)
                    elif isinstance(item, ast.expr_context):
                        tokens.append(ctx_token(item))
                    elif isinstance(item, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                        tokens.append(op_token(item))
                    else:
                        tokens.extend(_serialize_literal(item))
                tokens.append(LIST_END)

            elif isinstance(value, ast.AST):
                visit(value)

            elif isinstance(value, ast.expr_context):
                tokens.append(ctx_token(value))

            elif isinstance(value, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                tokens.append(op_token(value))

            elif isinstance(value, str) and field in {"id", "arg", "name", "attr"}:
                tokens.extend(_serialize_id(value))

            else:
                tokens.extend(_serialize_literal(value))

        tokens.append(CLOSE)

    visit(tree)
    return tokens


# ============================
# TOKENS -> AST
# ============================
class Reader:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str:
        if self.i >= len(self.tokens):
            return EOS
        return self.tokens[self.i]

    def next(self) -> str:
        tok = self.peek()
        self.i += 1
        return tok


def tokens_to_ast(tokens: List[str]) -> ast.AST:
    def parse_literal(r: Reader):
        tok = r.peek()

        if tok == NONE:
            r.next()
            return None

        if tok == BOOL_BEGIN:
            r.next()
            val = r.next()
            if r.next() != BOOL_END:
                raise ValueError("Malformed <lit_bool>")
            return True if val == "True" else False

        if tok == INT_BEGIN:
            r.next()
            val = int(r.next())
            if r.next() != INT_END:
                raise ValueError("Malformed <lit_int>")
            return val

        if tok == FLOAT_BEGIN:
            r.next()
            raw = r.next()
            if r.next() != FLOAT_END:
                raise ValueError("Malformed <lit_float>")

            # Robust handling for repr(float) outputs
            raw_lower = raw.lower()
            if raw_lower == "nan":
                return float("nan")
            if raw_lower == "inf":
                return float("inf")
            if raw_lower == "-inf":
                return float("-inf")

            # Normal float
            return float(raw)

        if tok == STR_BEGIN:
            r.next()
            payload = r.next()
            if r.next() != STR_END:
                raise ValueError("Malformed <lit_str>")
            return decode_payload(payload)

        raise ValueError(f"Unexpected literal token: {tok}")

    def parse_id(r: Reader):
        if r.next() != ID_BEGIN:
            raise ValueError("Expected <id>")
        payload = r.next()
        if r.next() != ID_END:
            raise ValueError("Expected </id>")
        return decode_payload(payload)

    def parse_node(r: Reader):
        nodetype = r.next()
        if not hasattr(ast, nodetype):
            raise ValueError(f"Unknown AST node type: {nodetype}")

        cls = getattr(ast, nodetype)

        if r.next() != OPEN:
            raise ValueError("Expected <open>")

        fields: Dict[str, Any] = {}

        while True:
            tok = r.peek()
            if tok == EOS:
                raise ValueError("Unexpected EOS while parsing")
            if tok == CLOSE:
                break

            f_tok = r.next()
            if not f_tok.startswith("<field:"):
                raise ValueError(f"Expected <field:...>, got {f_tok}")

            field = f_tok[len("<field:"):-1]
            tok = r.peek()

            if tok == LIST_BEGIN:
                r.next()
                values = []

                while True:
                    tok2 = r.peek()
                    if tok2 == EOS:
                        raise ValueError("Unexpected EOS inside list")
                    if tok2 == LIST_END:
                        break

                    if is_ctx_token(tok2):
                        values.append(parse_ctx(r.next()))
                    elif is_op_token(tok2):
                        values.append(parse_op(r.next()))
                    elif tok2 == ID_BEGIN:
                        values.append(parse_id(r))
                    elif tok2 in {NONE, BOOL_BEGIN, INT_BEGIN, FLOAT_BEGIN, STR_BEGIN}:
                        values.append(parse_literal(r))
                    else:
                        values.append(parse_node(r))

                r.next()  # LIST_END
                fields[field] = values

            elif is_ctx_token(tok):
                fields[field] = parse_ctx(r.next())

            elif is_op_token(tok):
                fields[field] = parse_op(r.next())

            elif tok == ID_BEGIN:
                fields[field] = parse_id(r)

            elif tok in {NONE, BOOL_BEGIN, INT_BEGIN, FLOAT_BEGIN, STR_BEGIN}:
                fields[field] = parse_literal(r)

            else:
                fields[field] = parse_node(r)

        r.next()  # CLOSE

        try:
            node = cls(**fields)
        except TypeError:
            node = cls()
            for k, v in fields.items():
                setattr(node, k, v)

        return node

    r = Reader(tokens)
    tree = parse_node(r)
    ast.fix_missing_locations(tree)
    return tree


# ============================
# AST -> CODE
# ============================
def ast_to_code(tree: ast.AST) -> str:
    try:
        return ast.unparse(tree)
    except Exception:
        import astunparse
        return astunparse.unparse(tree)
