import json

class ASTTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path) as f:
            vocab = json.load(f)

        self.vocab = vocab  
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
    
    def __len__(self):
        return len(self.vocab)

    def encode(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids]
