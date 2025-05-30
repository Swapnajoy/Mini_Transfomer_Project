class WordTokenizer:
    def __init__(self, file_path=None, text=None):
        #requires either file_path or text
        assert (text is not None) or (file_path is not None)
        
        words = set()

        if text is None:
            with open(file_path, encoding="utf-8") as corpus:
                for line in corpus:
                    words.update(line.strip().split())

        else:
            words.update(text.strip().split())

        words = sorted(words)
        self.vocab = ['<UNK>'] + words
        self.stoi = {word: i for i, word in enumerate(words)}
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, 0) for word in text.split()]

    def decode(self, indices: list[int]) -> str:
        return ' '.join([self.itos[i] for i in indices])
    
    @property
    def vocab_size(self):
        return len(self.vocab)