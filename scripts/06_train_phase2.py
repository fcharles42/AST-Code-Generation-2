# scripts/06_train_phase2.py
import os, sys, torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Use phase1 checkpoint from Kaggle input first (preferred path)
PHASE1_DIR = "/kaggle/input/phase1/content/AST-Code-Generation-2/checkpoints/phase1_lora"
if not os.path.exists(PHASE1_DIR):
    # Fallback to local checkpoint if Kaggle input doesn't exist
    PHASE1_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase1_lora")

PHASE2_OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase2_lora")

TOKENIZED_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase2_tokenized.pt")

LR = 1e-5
MAX_STEPS = 1250
BATCH_SIZE = 2
GRAD_ACCUM = 4


class TokenizedDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {"input_ids": input_ids, "labels": labels}


def main():
    os.makedirs(PHASE2_OUT_DIR, exist_ok=True)

    if not os.path.exists(TOKENIZED_PATH):
        raise FileNotFoundError(f"Missing phase2_tokenized.pt: {TOKENIZED_PATH}")

    if not os.path.exists(PHASE1_DIR):
        raise FileNotFoundError(f"Missing phase1 checkpoint: {PHASE1_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        PHASE1_DIR,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loaded tokenizer from phase1")
    print("[INFO] Tokenizer size:", len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model,
        PHASE1_DIR,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    dataset = TokenizedDataset(TOKENIZED_PATH)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=PHASE2_OUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
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

    trainer.model.save_pretrained(PHASE2_OUT_DIR)
    tokenizer.save_pretrained(PHASE2_OUT_DIR)

    print(f"[OK] Saved Phase2 adapter to {PHASE2_OUT_DIR}")


if __name__ == "__main__":
    main()
