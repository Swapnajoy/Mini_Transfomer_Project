import os

class Tokenizer:
    def __init__(self):
        words = set()
        
        with open(r"data\alice_in_wonderland.txt", encoding="utf-8") as corpus:
            for line in corpus:
                words.update(line.strip().split())

        self.vocab = sorted(words)
        self.stoi = {word:i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[word] for word in text.strip().split()]
    
    def decode(self, indices: list[int]) -> str:
        return " ".join([self.itos[i] for i in indices])