import os
import unsloth
import json
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel
from ast_codec.tokenizer import ASTTokenizer

# =====================
# Config
# =====================
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "data/processed/prompt_ast_ids.jsonl" 
AST_VOCAB_PATH = "data/processed/ast_vocab.json"

MAX_SEQ_LEN = 4096
LR = 5e-5
EPOCHS = 1

DEVICE = "cuda" 

# =====================
# Load base model
# =====================
model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.float16,
    load_in_4bit=True,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

base_tokenizer.pad_token = base_tokenizer.eos_token
BASE_VOCAB_SIZE = len(base_tokenizer)

# =====================
# Load AST tokenizer
# =====================
ast_tokenizer = ASTTokenizer(AST_VOCAB_PATH)

NUM_AST_TOKENS = len(ast_tokenizer)
AST_OFFSET = BASE_VOCAB_SIZE
AST_BOS = ast_tokenizer.bos_id + AST_OFFSET
AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

# Extend embeddings (critical)
model.resize_token_embeddings(BASE_VOCAB_SIZE + NUM_AST_TOKENS)

for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

# =====================
# LoRA
# =====================
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
)

model.print_trainable_parameters()

# =====================
# Dataset
# =====================
class PromptASTDataset(Dataset):
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
        # Encode prompt
        prompt_ids = base_tokenizer.encode(
            ex["prompt"],
            add_special_tokens=False,
        )

        ast_ids = [i + AST_OFFSET for i in ex["ast_ids"]]

        ids = (
            prompt_ids +
            [AST_BOS] +
            ast_ids +
            [AST_EOS]
        )

        lbl = (
            [-100] * (len(prompt_ids) + 1) +
            ast_ids +
            [AST_EOS]
        )

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
        output_dir="checkpoints/phase2_prompt_ast",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=50,
        save_steps=400,
        save_total_limit=4,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    ),
)

trainer.train()
