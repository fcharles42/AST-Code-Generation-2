# ast_codec/tokenize_ast_sequence.py

"""
Converts codec token sequences into model input_ids:

- Structural tokens like "<open>", "Module", "<field:body>" must exist in tokenizer vocab.
- Payload chunks (base64 strings) are NOT in vocab.
  Those are tokenized using the base tokenizer.encode().

Example AST tokens:
    ["<id>", "dmFyXzA=", "</id>"]

Becomes:
    [id("<id>")] + encode("dmFyXzA=") + [id("</id>")]
"""

from typing import List
from transformers import PreTrainedTokenizerBase


PAYLOAD_BEGIN_TOKENS = {
    "<id>",
    "<lit_str>",
    "<lit_int>",
    "<lit_float>",
    "<lit_bool>",
}

PAYLOAD_END_TOKENS = {
    "</id>",
    "</lit_str>",
    "</lit_int>",
    "</lit_float>",
    "</lit_bool>",
}

PAYLOAD_PAIR = {
    "<id>": "</id>",
    "<lit_str>": "</lit_str>",
    "<lit_int>": "</lit_int>",
    "<lit_float>": "</lit_float>",
    "<lit_bool>": "</lit_bool>",
}


def encode_codec_tokens(tokens: List[str], tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """
    Convert codec_v2 tokens into HF tokenizer ids.

    Structural tokens must already be present in tokenizer vocab.
    Payload chunks are base64 strings, tokenized using tokenizer.encode().
    """
    ids = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # Payload wrapper begin
        if tok in PAYLOAD_BEGIN_TOKENS:
            end_tok = PAYLOAD_PAIR[tok]

            begin_id = tokenizer.convert_tokens_to_ids(tok)
            end_id = tokenizer.convert_tokens_to_ids(end_tok)

            if begin_id == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer missing payload begin token: {tok}")
            if end_id == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer missing payload end token: {end_tok}")

            ids.append(begin_id)
            i += 1

            # Collect payload tokens until end tag
            payload_parts = []
            while i < len(tokens) and tokens[i] != end_tok:
                payload_parts.append(tokens[i])
                i += 1

            if i >= len(tokens):
                raise ValueError(f"Missing closing token {end_tok} for payload {tok}")

            # IMPORTANT:
            # Do NOT assume payload is 1 token.
            # Codec might output it as 1 token, but future changes may not.
            payload_text = "".join(payload_parts)

            # Encode payload using base tokenizer
            payload_ids = tokenizer.encode(payload_text, add_special_tokens=False)
            ids.extend(payload_ids)

            # Add end tag
            ids.append(end_id)

            i += 1
            continue

        # Literal NONE is standalone structural token
        if tok == "<lit_none>":
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid == tokenizer.unk_token_id:
                raise ValueError("Tokenizer missing token: <lit_none>")
            ids.append(tid)
            i += 1
            continue

        # Normal AST structural token
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            raise ValueError(f"Unknown structural token not in vocab: {tok}")

        ids.append(tid)
        i += 1

    return ids
