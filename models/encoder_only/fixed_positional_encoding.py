import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        pe = torch.empty((max_seq_len, embed_dim), dtype=torch.float32)

        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe [:, 0::2] = torch.sin(pos * div_term)
        pe [:, 1::2] = torch.cos(pos * div_term)
        
        self.register_buffer("sinusoidal_positional_encoding", pe)

        self.dp = nn.Dropout(p=0.1)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.dp(x + self.sinusoidal_positional_encoding[:seq_len,:].to(x.device))