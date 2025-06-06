import torch
import torch.nn as nn

from models.decoder_only.masked_attention import MaskedMultiHeadAttentionBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dec_ffn_h_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.mmha = MaskedMultiHeadAttentionBlock(embed_dim, hidden_dim, num_heads)
        self.dp1 = nn.Dropout(p=0.2)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dec_ffn_h_dim),
            nn.GELU(),
            nn.Linear(dec_ffn_h_dim, embed_dim)
        )
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x):
        scores, weights = self.mmha(self.ln1(x))
        x = x + self.dp1(scores)
        x = x + self.dp2(self.ffn(self.ln2(x)))
        return x, weights