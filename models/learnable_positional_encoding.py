import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dp = nn.Dropout(p=0.1)

    def forward(self, x):
        pos_id = torch.arange(0, x.shape[1]).unsqueeze(0).to(x.device)
        pos_embed = self.pos_embed(pos_id)
        return self.dp(x + pos_embed)