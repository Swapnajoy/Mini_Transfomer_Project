import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index:index+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[index+1:index+1+self.seq_len], dtype=torch.long)
        return x, y