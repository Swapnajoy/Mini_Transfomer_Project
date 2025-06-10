import torch
import torch.nn as nn
from models.encoder_only.embedding import EmbeddingLayer

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dp = nn.Dropout(p=0.4)
        self.lin = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.LSTM(x)
        return self.lin(self.dp(out))