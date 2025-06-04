import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.decoder_only.masked_attention import MaskedMultiHeadAttentionBlock

class DecoderBlockSeq2Seq(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dec_ffn_h_dim):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.ln1 = nn.LayerNorm(embed_dim)
        self.mmha = MaskedMultiHeadAttentionBlock(embed_dim, hidden_dim, num_heads)
        self.dp1 = nn.Dropout(p=0.2)

        self.Wq = nn.Linear(embed_dim, hidden_dim)
        self.Wk = nn.Linear(embed_dim, hidden_dim)
        self.Wv = nn.Linear(embed_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, embed_dim)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dec_ffn_h_dim),
            nn.GELU(),
            nn.Linear(dec_ffn_h_dim, embed_dim)
        )
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x, y):
        scores, _ = self.mmha(self.ln1(x))
        x = x + self.dp1(scores)

        Q = self.Wq(x)
        K = self.Wk(y)
        V = self.Wv(y)

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.head_dim), dim = -1)
        weighted_values = torch.matmul(attention_weights,V)

        weighted_values = weighted_values.permute(0, 2, 1, 3)
        weighted_values = weighted_values.reshape(weighted_values.shape[0], weighted_values.shape[1], -1)

        out = self.Wo(weighted_values)

        out = out + self.dp2(self.ffn(self.ln2(out)))

        return out