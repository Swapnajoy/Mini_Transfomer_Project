class CharTokenizer:
    def __init__(self, file_path=None, text=None):
        #requires either file_path or text
        assert (text is not None) or (file_path is not None)
        
        if text is None:
            with open(file_path, encoding="utf-8") as corpus:
                text = corpus.read()

        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join([self.itos[i] for i in indices])
    
    @property
    def vocab_size(self):
        return len(self.vocab)