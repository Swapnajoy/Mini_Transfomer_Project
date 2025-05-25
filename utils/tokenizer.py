class Tokenizer:
    def __init__(self, file_path):
        words = set()
        
        with open(file_path, encoding="utf-8") as corpus:
            for line in corpus:
                words.update(line.strip().split())

        self.vocab = ['unk', 'pad'] + sorted(words)     #to handle unseen words during inference and batching with padding
        self.stoi = {word:i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, self.stoi['unk']) for word in text.strip().split()]
    
    def decode(self, indices: list[int]) -> str:
        return " ".join([self.itos[i] for i in indices])
    
    def vocab_size(self):
        return len(self.vocab)