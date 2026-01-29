# AST-First Pretraining for Code Generation

This repository implements a **two-phase training pipeline** for learning
**Python abstract syntax tree (AST) structure first**, followed by
**natural-language → AST alignment**.

- **Phase 1** trains a language model to generate *valid Python ASTs* without
  natural language or surface code.
- **Phase 2** conditions AST generation on natural-language descriptions.

---

## High-Level Overview

### Phase 1: AST Structure Pretraining
- Input: serialized Python AST tokens
- Learns Python grammar and tree structure explicitly

### Phase 2: Natural Language → AST
- Input: natural-language description + AST BOS token
- Output: serialized AST tokens
- Uses Phase 1 checkpoint
- Loss applied only on AST tokens

---

## Datasets

**Phase 1**
- `bigcode/the-stack` (Python)
- Function-level ASTs extracted and serialized

**Phase 2**
- `CodeSearchNet` (Python)
- Docstrings used as natural-language prompts
- Code parsed into AST targets

---
## Reproducing the Pipeline

### Phase 1: AST-only dataset
python scripts/preprocess_stack.py
python scripts/build_ast_vocab.py
python scripts/encode_dataset.py

### Phase 1 training
python train_model_phase1.py

### Phase 2: NL → AST dataset
python scripts/build_nl_ast_pairs.py

### Phase 2 training
python train_model_phase2.py
Then add a one-line disclaimer:

## Status

- AST codec: complete
- Phase 1 training: complete
- Phase 2 training: dataset creation is complete; training pending
- Evaluation: not yet implemented
