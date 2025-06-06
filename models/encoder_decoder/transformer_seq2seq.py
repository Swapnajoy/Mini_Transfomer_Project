import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_only.embedding import EmbeddingLayer
from models.encoder_only.learnable_positional_encoding import LearnablePositionalEncoding
from models.encoder_only.fixed_positional_encoding import SinusoidalPositionalEncoding
from models.encoder_only.encoder_block import EncoderBlock
from models.encoder_decoder.decoder_block import DecoderBlockSeq2Seq

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, hidden_dim, num_heads, enc_ffn_h_dim, dec_ffn_h_dim, num_enc, num_dec, use_sinusoidal=True):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size, embed_dim)
        
        if use_sinusoidal:
            self.pe = SinusoidalPositionalEncoding(seq_len, embed_dim)
        else:
            self.pe = LearnablePositionalEncoding(seq_len, embed_dim)

        self.encoders = nn.ModuleList([EncoderBlock(embed_dim, hidden_dim, num_heads, enc_ffn_h_dim) for _ in range(num_enc)])
        self.decoders = nn.ModuleList([DecoderBlockSeq2Seq(embed_dim, hidden_dim, num_heads, dec_ffn_h_dim) for _ in range(num_dec)])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, vocab_size)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, src, tgt):
        src = self.embed(src)
        src = self.pe(src)

        tgt = self.embed(tgt)
        tgt = self.pe(tgt)

        enc_attn_wgts = []
        dec_attn_wgts = []

        enc_out = src
        for layer in self.encoders:
            enc_out, weights = layer(enc_out)
            enc_attn_wgts.append(weights)
        
        dec_out = tgt
        for layer in self.decoders:
            dec_out, weights = layer(dec_out, enc_out)
            dec_attn_wgts.append(weights)

        out = self.dp(self.ffn(self.ln(dec_out)))
        return out, enc_attn_wgts, dec_attn_wgts