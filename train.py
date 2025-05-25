import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.tokenizer = Tokenizer(file_path)
    
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
            self.tokens = self.tokenizer.encode(text)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index:index+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[index+1:index+1+self.seq_len], dtype=torch.long)
        return x, y