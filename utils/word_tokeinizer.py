import json

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
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, 0) for word in text.split()]

    def decode(self, indices: list[int]) -> str:
        return ' '.join([self.itos[i] for i in indices])
    
    def save(self, file_path):
        tokenizer_dict = {
            "vocab": self.vocab,
            "word2idx": self.stoi
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            tokenizer_dict = json.load(f)

        obj = cls(text="")
        obj.vocab = tokenizer_dict['vocab']
        obj.stoi = tokenizer_dict['word2idx']
        obj.itos = {i: word for word, i in obj.stoi.items()}
        return obj
    
    @property
    def vocab_size(self):
        return len(self.vocab)