import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedMultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.Wq = nn.Linear(embed_dim, hidden_dim)
        self.Wk = nn.Linear(embed_dim, hidden_dim)
        self.Wv = nn.Linear(embed_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, embed_dim)

    def forward(self,x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        alignment = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.head_dim)
        seq_len = alignment.shape[-1]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal = 1).bool()

        alignment = alignment.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attention_weights = F.softmax(alignment, dim=-1)

        weighted_values = torch.matmul(attention_weights,V)

        weighted_values = weighted_values.permute(0, 2, 1, 3)
        weighted_values = weighted_values.reshape(weighted_values.shape[0], weighted_values.shape[1], -1)

        return self.Wo(weighted_values), attention_weights