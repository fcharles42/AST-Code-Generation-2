# AST Code Generation

# Structure
- `ast_codec/`: AST serialization and tokenization
- `scripts/`: Dataset preprocessing, vocab building, encoding
- `train_model.py`: Phase-1 AST-only training (structure-first)

# Phase 1
- Input: AST token sequences
- Objective: next-token LM over AST
- Model: Qwen-0.5B + Unsloth + LoRA