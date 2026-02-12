# ast_codec/fsm_decoder.py
"""
Optional constrained decoding (finite state machine).

For v1 pipeline, you can skip using it.
But this file exists so later you can plug in logits masking.
"""

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


def inside_payload(stack):
    """
    stack keeps open payload tags.
    If non-empty, we allow base tokenizer tokens.
    """
    return len(stack) > 0


def update_stack(tok: str, stack):
    if tok in {ID_BEGIN, STR_BEGIN, INT_BEGIN, FLOAT_BEGIN, BOOL_BEGIN}:
        stack.append(tok)
    elif tok in {ID_END, STR_END, INT_END, FLOAT_END, BOOL_END}:
        if stack:
            stack.pop()
    return stack