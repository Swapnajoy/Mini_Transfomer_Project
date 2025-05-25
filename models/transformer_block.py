import torch
import torch.nn as nn
from models.attention import MultiHeadAttentionBlock

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, enc_ffn_h_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttentionBlock(embed_dim, hidden_dim, num_heads)
        self.dp1 = nn.Dropout(p=0.2)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, enc_ffn_h_dim),
            nn.GELU(),
            nn.Linear(enc_ffn_h_dim, embed_dim)
        )
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x += self.dp1(self.attention(self.ln1(x)))
        x += self.dp2(self.feedforward(self.ln2(x)))
        return x