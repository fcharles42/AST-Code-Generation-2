import ast
from datasets import load_dataset

# =====================
# Structural tokens
# =====================
OPEN = "<open>"
CLOSE = "<close>"
INDENT = "<indent>"
DEDENT = "<dedent>"

EOS = "<eos>"

# Fields that actually represent BLOCKS
BLOCK_FIELDS = {"body", "orelse", "finalbody", "handlers"}

LITERAL_TOKENS = {"<STR>", "<INT>", "<FLOAT>", "<BOOL>", "<NONE>"}

# =====================
# Literal handling 
# =====================
def literal_token(value):
    if isinstance(value, str):
        return "<STR>"
    if isinstance(value, bool):
        return "<BOOL>"
    if value is None:
        return "<NONE>"
    if isinstance(value, int):
        return "<INT>"
    if isinstance(value, float):
        return "<FLOAT>"
    raise ValueError(f"Unsupported literal: {type(value).__name__}")

def parse_literal(tok):
    return {
        "<STR>": "",
        "<INT>": 0,
        "<FLOAT>": 0.0,
        "<BOOL>": False,
        "<NONE>": None,
    }.get(tok, None)

# =====================
# Operator handling 
# =====================
def op_token(op):
    return f"<op:{op.__class__.__name__}>"

def parse_op(tok):
    return getattr(ast, tok[4:-1])()

# =====================
# Renaming variables 
# =====================
class FlatAlphaRenamer(ast.NodeTransformer):
    def __init__(self):
        self.var_map = {}
        self.fn_map = {}
        self.var_counter = 0
        self.fn_counter = 0

    def _new_var(self):
        name = f"var_{self.var_counter}"
        self.var_counter += 1
        return name

    def _new_fn(self):
        name = f"fn_{self.fn_counter}"
        self.fn_counter += 1
        return name

    def visit_FunctionDef(self, node):
        # function name
        if node.name not in self.fn_map:
            self.fn_map[node.name] = self._new_fn()
        node.name = self.fn_map[node.name]

        # reset variable namespace per function
        self.var_map = {}
        self.var_counter = 0

        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        if node.arg not in self.var_map:
            self.var_map[node.arg] = self._new_var()
        node.arg = self.var_map[node.arg]
        return node

    def visit_Name(self, node):
        if node.id not in self.var_map:
            self.var_map[node.id] = self._new_var()
        node.id = self.var_map[node.id]
        return node

def id_token(name: str):
    return f"<id:{name}>"

def parse_id(tok):
    return tok[4:-1]

# =====================
# AST → TOKENS 
# =====================
def serialize(node, out):
    out.append(node.__class__.__name__)
    out.append(OPEN)

    # ---- Explicit operator nodes ----
    if isinstance(node, ast.BinOp):
        out.append("<field:left>")
        serialize(node.left, out)
        out.append("<field:op>")
        out.append(op_token(node.op))
        out.append("<field:right>")
        serialize(node.right, out)

    elif isinstance(node, ast.UnaryOp):
        out.append("<field:op>")
        out.append(op_token(node.op))
        out.append("<field:operand>")
        serialize(node.operand, out)

    elif isinstance(node, ast.BoolOp):
        out.append("<field:op>")
        out.append(op_token(node.op))
        out.append("<field:values>")
        out.append(INDENT)
        for v in node.values:
            serialize(v, out)
        out.append(DEDENT)

    elif isinstance(node, ast.Compare):
        out.append("<field:left>")
        serialize(node.left, out)

        out.append("<field:ops>")
        out.append(INDENT)
        for op in node.ops:
            out.append(op_token(op))
        out.append(DEDENT)

        out.append("<field:comparators>")
        out.append(INDENT)
        for c in node.comparators:
            serialize(c, out)
        out.append(DEDENT)

    # ---- Generic handling ----
    else:
        for field in node._fields or []:
            value = getattr(node, field, None)
            out.append(f"<field:{field}>")

            if isinstance(value, list):
                if field in BLOCK_FIELDS:
                    out.append(INDENT)
                    for v in value:
                        serialize(v, out)
                    out.append(DEDENT)
                else:
                    out.append(INDENT)
                    for v in value:
                        if isinstance(v, ast.AST):
                            serialize(v, out)
                        elif isinstance(v, ast.expr_context):
                            out.append(f"<ctx:{v.__class__.__name__}>")
                        elif isinstance(v, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                            out.append(op_token(v))
                        else:
                            out.append(literal_token(v))
                    out.append(DEDENT)

            elif isinstance(value, ast.AST):
                serialize(value, out)

            elif isinstance(value, ast.expr_context):
                out.append(f"<ctx:{value.__class__.__name__}>")

            elif isinstance(value, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                out.append(op_token(value))
            
            elif isinstance(value, str) and field in {"id", "arg", "name"}:
                out.append(id_token(value))

            else:
                out.append(literal_token(value))

    out.append(CLOSE)

def ast_to_tokens(tree):
    tree = FlatAlphaRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    tokens = []
    serialize(tree, tokens)
    return tokens

# =====================
# TOKENS → AST 
# =====================
class Reader:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def peek(self):
        if self.i >= len(self.toks):
            return EOS
        return self.toks[self.i]

def parse_node(r):
    nodetype = r.next()
    if not hasattr(ast, nodetype):
        raise ValueError(f"Unknown AST node type: {nodetype}")
    cls = getattr(ast, nodetype)

    if r.next() != OPEN:
        raise ValueError("Expected <open> after node type")

    fields = {}

    while True:
        tok = r.peek()

        if tok == EOS:
            raise ValueError("Unexpected <eos> while parsing AST node")

        if tok == CLOSE:
            break

        field_tok = r.next()
        field = field_tok[7:-1]

        if field not in cls._fields:
            raise ValueError(
                f"Invalid field '{field}' for AST node {cls.__name__}"
            )

        tok = r.peek()

        if tok == INDENT:
            r.next()
            values = []
            while True:
                tok = r.peek()

                if tok == EOS:
                    raise ValueError("Unexpected <eos> inside block")

                if tok == DEDENT:
                    break

                if tok.startswith("<op:"):
                    values.append(parse_op(r.next()))
                else:
                    values.append(parse_node(r))

            r.next()  # consume DEDENT
            fields[field] = values

        elif tok.startswith("<op:"):
            fields[field] = parse_op(r.next())

        elif tok.startswith("<ctx:"):
            ctx_tok = r.next()
            fields[field] = getattr(ast, ctx_tok[5:-1])()

        elif tok.startswith("<id:"):
            fields[field] = parse_id(r.next())

        elif tok in LITERAL_TOKENS:
            fields[field] = parse_literal(r.next())

        else:
            fields[field] = parse_node(r)

    r.next()  

    try:
        return cls(**fields)
    except TypeError:
        node = cls()
        for k, v in fields.items():
            setattr(node, k, v)
        return node

def tokens_to_ast(tokens):
    r = Reader(tokens)
    return parse_node(r)

# =====================
# CODE ↔ AST ↔ CODE
# =====================
def code_to_ast(code):
    return ast.parse(code)

def ast_to_code(tree):
    try:
        return ast.unparse(tree)
    except AttributeError:
        import astunparse
        return astunparse.unparse(tree)

# =====================
# ROUND-TRIP TEST
# =====================
if __name__ == "__main__":
    code = """
def f(a, b):
    if a < b < 10:
        return a + b
"""
    tokens = ast_to_tokens(code_to_ast(code))

    assert tokens.count(OPEN) == tokens.count(CLOSE)
    assert tokens.count(INDENT) == tokens.count(DEDENT)
    assert tokens[0] == "Module"
    assert tokens[-1] == CLOSE

    tree = tokens_to_ast(tokens)
    ast.fix_missing_locations(tree)
    compile(tree, "<ast>", "exec")
    print(ast_to_code(tree))
