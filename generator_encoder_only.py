import os

import torch
import torch.nn as nn

from utils.ch_tokenizer import CharTokenizer
from models.encoder_only.enc_only_transformer import TransformerLanguageModel
from config_encoder_only import CHECKPOINT_PATH, TXT_FILE_PATH, SEQ_LEN, MODEL_CONFIG, TOPK_CONFIG

checkpoint_path = CHECKPOINT_PATH
txt_file_path = TXT_FILE_PATH

tokenizer = CharTokenizer(txt_file_path)
vocab_size = tokenizer.vocab_size

seq_len = SEQ_LEN
embed_dim = MODEL_CONFIG['embed_dim']
num_heads = MODEL_CONFIG['num_heads']
hidden_dim = MODEL_CONFIG['hidden_dim']
enc_ffn_h_dim = MODEL_CONFIG['enc_ffn_h_dim']
num_enc = MODEL_CONFIG['num_enc']
use_sinusoidal = MODEL_CONFIG['use_sinusoidal']

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerLanguageModel(vocab_size, embed_dim, seq_len, hidden_dim, num_heads, enc_ffn_h_dim, num_enc, use_sinusoidal).to(device)

model.load_state_dict(torch.load(checkpoint_path))

model.eval()

seed_text = "Alice was beginning to get very tired of sitting by her sister on the\nbank, and of having"

generated_tokens = tokenizer.encode(seed_text)

if len(generated_tokens) > seq_len:
    input_tokens = torch.tensor(generated_tokens[-seq_len:]).unsqueeze(0).to(device)
else:    
    input_tokens = torch.tensor(generated_tokens).unsqueeze(0).to(device)


#Generate 500 characters and use top-k and temperature sampling
temperature = TOPK_CONFIG['temperature']
k = TOPK_CONFIG['k']

with torch.no_grad():
    for _ in range(500):
        pred_logits, _ = model(input_tokens)

        probs = torch.softmax(pred_logits[0, -1, :]/temperature, dim = -1)
        topk_probs, topk_indices = torch.topk(probs, k)
        next_token = torch.multinomial(topk_probs, 1).item()
        next_token_idx = topk_indices[next_token].unsqueeze(0)

        generated_tokens.append(next_token_idx.cpu().item())
        next_token_idx = next_token_idx.to(device).unsqueeze(0)
        input_tokens = torch.cat((input_tokens, next_token_idx), dim = 1)

        if input_tokens.shape[1] > seq_len:
            input_tokens = input_tokens[:, -seq_len:]
    
    generated_words = tokenizer.decode(generated_tokens)
    print(generated_words)