import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset
from peft import PeftModel
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

# =====================
# Config
# =====================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TOKENIZED_PATH = "/kaggle/working/data/processed/tokenized.pt"
LORA_CHECKPOINT = "/kaggle/input/checkpoint/checkpoints/ast_model/checkpoint-523"

LR = 1e-5

# =====================
# Load tokenizer
# =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token

# =====================
# Load model
# =====================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = False
model.resize_token_embeddings(model.get_input_embeddings().weight.size(0))

# =====================
# Load LoRA
# =====================
model = PeftModel.from_pretrained(
    model,
    LORA_CHECKPOINT,
    is_trainable=True,
)

# Freeze base embeddings
for p in model.get_input_embeddings().parameters():
    p.requires_grad = False

model.print_trainable_parameters()

# =====================
# Dataset
# =====================
class TokenizedDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100,
        ),
    }

dataset = TokenizedDataset(TOKENIZED_PATH)

# =====================
# Train
# =====================
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=collate,
    args=TrainingArguments(
        output_dir="/kaggle/working/checkpoints/phase2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=3000,
        learning_rate=LR,
        fp16=True,
        logging_steps=50,
        save_steps=400,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,
    ),
)

trainer.train()
