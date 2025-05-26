import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from utils.tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.tokenizer = Tokenizer(file_path)
        self.tokens = self.tokenizer.encode(open(file_path, encoding='utf-8').read())
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index:index+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[index+1:index+1+self.seq_len], dtype=torch.long)
        return x, y
    
    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)