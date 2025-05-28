class Tokenizer:
    def __init__(self, file_path):
        chars = set()
        
        with open(file_path, encoding="utf-8") as corpus:
            for line in corpus:
                chars.update(ch for ch in line)

        self.vocab = sorted(chars)
        self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    def decode(self, indices: list[int]) -> str:
        return ''.join([self.itos[i] for i in indices])
    
    def vocab_size(self):
        return len(self.vocab)