import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unsloth
import json
import torch
import random
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel
from ast_codec.tokenizer import ASTTokenizer

# =====================
# Config
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "nl_ast_pairs.jsonl"
)

AST_VOCAB_PATH = os.path.join(
    BASE_DIR, "data", "processed", "ast_vocab.json"
)

LORA_CHECKPOINT = os.path.join(
    BASE_DIR, "checkpoints", "ast_model", "checkpoint-523"
)

MAX_SEQ_LEN = 2048
LR = 1e-5
EPOCHS = 1
DEVICE = "cuda" 

# =====================
# Load base model
# =====================

base_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    trust_remote_code=True,
    use_fast=True,
)
base_tokenizer.pad_token = base_tokenizer.eos_token

# Load base model
model, _ = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B",
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=True
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.attn_implementation = "sdpa"

BASE_VOCAB_SIZE = len(base_tokenizer)
ast_tokenizer = ASTTokenizer(AST_VOCAB_PATH)
NUM_AST_TOKENS = len(ast_tokenizer) - 1

model.resize_token_embeddings(
    BASE_VOCAB_SIZE + NUM_AST_TOKENS,
    mean_resizing=False,   
)

print("Tokenizer vocab:", len(base_tokenizer))
print("Embedding size BEFORE LoRA:", model.get_input_embeddings().weight.shape[0])

model = PeftModel.from_pretrained(
    model,
    LORA_CHECKPOINT,
    is_trainable=True,
)

for param in model.get_input_embeddings().parameters():
    param.requires_grad = False

model.gradient_checkpointing_enable()
model.config.use_cache = False

print(model.get_input_embeddings().weight.shape)
print(len(base_tokenizer))
model.print_trainable_parameters()

# =====================
# Load AST tokenizer
# =====================

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

AST_OFFSET = BASE_VOCAB_SIZE
AST_START_ID = ast_tokenizer.bos_id + AST_OFFSET
AST_EOS = ast_tokenizer.eos_id + AST_OFFSET

assert AST_OFFSET + len(ast_tokenizer) <= model.get_input_embeddings().weight.shape[0]

# =====================
# Dataset
# =====================
class PromptASTDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = [json.loads(l) for l in f]
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(batch):
    input_ids, labels = [], []

    for ex in batch:
        prompt_ids = base_tokenizer.encode(
            ex["prompt"],
            add_special_tokens=False,
        )

        ast_ids = [i + AST_OFFSET for i in ex["ast_ids"]]

        ids = (
            prompt_ids +
            [AST_START_ID] +
            ast_ids
        )

        lbl = (
            [-100] * len(prompt_ids) +
            [-100] +        
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
        output_dir="/kaggle/working/checkpoints/phase2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        max_grad_norm=1.0,
        logging_steps=50,
        disable_tqdm=False,
        save_steps=400,
        save_total_limit=4,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    ),
)
trainer.train()
