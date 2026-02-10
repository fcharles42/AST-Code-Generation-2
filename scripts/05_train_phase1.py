# scripts/05_train_phase1.py
import os, sys, json, torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

AST_VOCAB_PATH = os.path.join(REPO_ROOT, "data", "processed", "ast_vocab.json")
TOKENIZED_PATH = os.path.join(REPO_ROOT, "data", "processed", "phase1_tokenized.pt")

OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase1_lora")

LR = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


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
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(TOKENIZED_PATH):
        raise FileNotFoundError(f"Missing phase1_tokenized.pt: {TOKENIZED_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    with open(AST_VOCAB_PATH, "r", encoding="utf-8") as f:
        ast_vocab = json.load(f)

    added = tokenizer.add_tokens(ast_vocab, special_tokens=False)
    print("[INFO] Added AST tokens:", added)
    print("[INFO] Tokenizer size:", len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = TokenizedDataset(TOKENIZED_PATH)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=OUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=True,
            logging_steps=50,
            save_steps=200,
            save_total_limit=2,
            report_to="none",
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
        ),
    )

    trainer.train()

    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[OK] Saved Phase1 adapter to {OUT_DIR}")


if __name__ == "__main__":
    main()
