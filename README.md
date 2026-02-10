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

## Folder outputs
- data/processed/
- checkpoints/
- results/

## Pipeline order

python scripts/00_build_ast_vocab.py
python scripts/01_build_phase1_ast_dataset.py
python scripts/02_build_phase2_nl_ast_dataset.py
python scripts/03_tokenize_phase1.py
python scripts/04_tokenize_phase2.py
python scripts/05_train_phase1.py
python scripts/06_train_phase2.py
python scripts/07_merge_lora.py
python scripts/08_eval.py
python scripts/09_sanity_generate.py

## Notes
- Uses Qwen/Qwen2.5-0.5B as base model
- Uses LoRA fine-tuning
- Uses deterministic AST vocab generation
- Uses hybrid AST + BPE encoding for identifiers and literals