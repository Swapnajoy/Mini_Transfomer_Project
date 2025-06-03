import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_only.embedding import EmbeddingLayer
from models.encoder_only.learnable_positional_encoding import LearnablePositionalEncoding
from models.encoder_only.fixed_positional_encoding import SinusoidalPositionalEncoding
from models.decoder_only.decoder_block import DecoderBlock

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, hidden_dim, num_heads, dec_ffn_h_dim, num_dec, use_sinusoidal=True):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size, embed_dim)
        
        if use_sinusoidal:
            self.pe = SinusoidalPositionalEncoding(seq_len, embed_dim)
        else:
            self.pe = LearnablePositionalEncoding(seq_len, embed_dim)

        self.decoders = nn.ModuleList([DecoderBlock(embed_dim, hidden_dim, num_heads, dec_ffn_h_dim) for _ in range(num_dec)])
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, vocab_size)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.embed(x)
        x = self.pe(x)

        attention_weights = []

        for layer in self.decoders:
            x, weights = layer(x)
            attention_weights.append(weights)
        x = self.dp(self.ffn(self.ln(x)))
        return x, attention_weights