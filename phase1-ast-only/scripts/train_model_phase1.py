import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import json
import unsloth
import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from ast_codec.tokenizer import ASTTokenizer


# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
AST_VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ast_vocab.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "python_ast_ids.jsonl")

MAX_SEQ_LEN = 4096


# -------------------------
# Load model + tokenizer via Unsloth
# -------------------------
model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype = torch.float16,
    load_in_4bit = True,  
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

base_tokenizer.pad_token = base_tokenizer.eos_token
BASE_VOCAB_SIZE = len(base_tokenizer)

# -------------------------
# Load AST tokenizer
# -------------------------
ast_tokenizer = ASTTokenizer(AST_VOCAB_PATH)

NUM_AST_TOKENS = len(ast_tokenizer)
AST_OFFSET = BASE_VOCAB_SIZE
AST_BOS = ast_tokenizer.bos_id + AST_OFFSET
AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

model.resize_token_embeddings(BASE_VOCAB_SIZE + NUM_AST_TOKENS)
# Unfreeze embeddings
for param in model.get_input_embeddings().parameters():
    param.requires_grad = True


# -------------------------
# Dataset
# -------------------------
class ASTDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch):
    input_ids, labels = [], []

    for ex in batch:
        ast_ids = [tid + AST_OFFSET for tid in ex["ast_ids"]]

        ids = [AST_BOS] + ast_ids
        lbl = ast_ids + [AST_EOS]

        ids = ids[:MAX_SEQ_LEN]
        lbl = lbl[:MAX_SEQ_LEN]

        input_ids.append(torch.tensor(ids))
        labels.append(torch.tensor(lbl))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=base_tokenizer.pad_token_id,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
    }



# -------------------------
# Enable LoRA via Unsloth
# -------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
],
    lora_dropout = 0.05,
)

model.print_trainable_parameters()


# -------------------------
# Train
# -------------------------
dataset = ASTDataset(DATA_PATH)

trainer = Trainer(
    model=model,
    tokenizer=base_tokenizer,
    train_dataset=dataset,
    data_collator=collate,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=200,
        output_dir="checkpoints/ast_model",
        report_to="none",
        save_total_limit=2,
        optim="paged_adamw_8bit",  
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    ),
)

ckpt_path = "checkpoints/ast_model/checkpoint-400"

trainer.train(resume_from_checkpoint=ckpt_path)
