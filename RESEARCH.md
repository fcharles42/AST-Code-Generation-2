+
# Research Notes: AST-First Training

This document describes the **methodological motivation and design decisions**
behind the AST-first training pipeline implemented in this repository.

---

This repository contains a two-phase training pipeline for learning Python abstract syntax tree (AST) structure:

- Phase 1 teaches the model to model valid Python AST structure independent of surface code or natural language.
- Phase 2 conditions AST generation on natural-language descriptions.
---

## AST Serialization

### Design Goals
- Fully reversible (AST → tokens → AST)
- Explicit tree structure


### Structural Tokens
- `<open>`, `<close>` — node boundaries
- `<indent>`, `<dedent>` — block structure
- `<field:*>` — explicit AST fields
- `<op:*>`, `<ctx:*>` — operators and contexts

This representation allows exact reconstruction of Python ASTs and enforces
structural correctness during generation.


## Identifier Normalization

All variables and function names are **alpha-renamed** at the function level:
x, y, result → var_0, var_1, var_2

This removes:
- Naming noise
- Dataset-specific biases
- Memorization shortcuts

The model is forced to learn **structure and control flow**, not identifiers.


## Phase 1: AST-Only Pretraining

### Objective
Model the distribution:

P(AST₁, AST₂, ..., ASTₙ)

No natural language or surface code is used.

### Dataset Construction
- Source: `bigcode/the-stack` (Python)
- ASTs extracted at function granularity
- Each function wrapped in a `Module`
- Length capped at 4096 AST tokens

### Training Setup
- Base model: Qwen2.5-0.5B
- 4-bit quantization + LoRA
- AST vocabulary appended to base tokenizer
- AST tokens occupy a disjoint ID range


---

## Phase 2: Natural Language → AST Alignment

### Objective
Condition AST generation on natural language:

P(AST | NL)

### Dataset
- Source: CodeSearchNet (Python)
- Input: function docstring tokens
- Output: serialized AST tokens
- Length filtered to fit context window

### Training Details
- Phase 1 LoRA checkpoint loaded
- Base model frozen
- AST embeddings frozen
- Loss applied only to AST tokens
- Prompt tokens masked from loss