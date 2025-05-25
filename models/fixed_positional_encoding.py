import torch
import torch.nn as nn
import math

print("hi")
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        pe = torch.empty((max_seq_len, embed_dim), dtype=torch.float32)
        for i in range(max_seq_len):
            for j in range(embed_dim):
                if j%2 == 0:
                    pe[i, j] = math.sin(i/math.pow(10000, j/embed_dim))
                else:
                    pe[i, j] = math.cos(i/math.pow(10000, (j-1)/embed_dim))
        
        self.register_buffer("sinusoidal_positional_encoding", pe)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.sinusoidal_positional_encoding[:seq_len,:]

model = SinusoidalPositionalEncoding(6, 10)
x = torch.rand((128, 6, 10))
print(model(x).shape)