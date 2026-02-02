import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel

from ast_codec.tokenizer import ASTTokenizer

# =====================
# Config
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "/kaggle/working/data/processed/nl_ast_pairs.jsonl"
AST_VOCAB_PATH = os.path.join(BASE_DIR, "data", "processed", "ast_vocab.json")
LORA_CHECKPOINT = "/kaggle/input/checkpoint/checkpoints/ast_model/checkpoint-523"

MAX_SEQ_LEN = 2048
LR = 1e-5
EPOCHS = 1

# =====================
# Load model + tokenizer
# =====================
base_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)
base_tokenizer.pad_token = base_tokenizer.eos_token

model, _ = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# =====================
# AST tokenizer
# =====================
ast_tokenizer = ASTTokenizer(AST_VOCAB_PATH)

BASE_VOCAB_SIZE = len(base_tokenizer)
NUM_AST_TOKENS = len(ast_tokenizer)

AST_OFFSET = BASE_VOCAB_SIZE
AST_BOS = ast_tokenizer.bos_id + AST_OFFSET
AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

model.resize_token_embeddings(BASE_VOCAB_SIZE + NUM_AST_TOKENS)

# =====================
# Load LoRA
# =====================
model = PeftModel.from_pretrained(
    model,
    LORA_CHECKPOINT,
    is_trainable=True,
)

# Freeze base embeddings (same as before)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = False

model.print_trainable_parameters()

# =====================
# Dataset
# =====================
class PromptASTDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(l) for l in f]
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =====================
# Collate 
# =====================
def collate(batch):
    input_ids, labels = [], []

    vocab = ast_tokenizer.token_to_id
    PAD_ID = ast_tokenizer.pad_id
    CANONICAL_ID = next(vocab[t] for t in ast_tokenizer.vocab if t.startswith("<id:"))

    for ex in batch:
        # Encode natural language prompt
        prompt_ids = base_tokenizer.encode(
            ex["prompt"],
            add_special_tokens=False,
        )

        # Encode AST tokens
        ast_ids = []
        for tok in ex["ast_tokens"]:
            if tok in vocab:
                ast_ids.append(vocab[tok])
            elif tok.startswith("<id:"):
                ast_ids.append(CANONICAL_ID)
            else:
                ast_ids.append(PAD_ID)

        ast_ids = [i + AST_OFFSET for i in ast_ids]

        ids = (
            prompt_ids
            + [AST_BOS]
            + ast_ids
            + [AST_EOS]
        )

        ids = ids[:MAX_SEQ_LEN]

        lbl = ids.copy()

        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lbl, dtype=torch.long))

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=base_tokenizer.pad_token_id,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=base_tokenizer.pad_token_id,
        ),
    }

# =====================
# Train
# =====================
dataset = PromptASTDataset(DATA_PATH)

trainer = Trainer(
    model=model,
    tokenizer=base_tokenizer,
    train_dataset=dataset,
    data_collator=collate,
    args=TrainingArguments(
        output_dir="/kaggle/working/checkpoints/phase2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=50,
        save_steps=400,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    ),
)

trainer.train()
