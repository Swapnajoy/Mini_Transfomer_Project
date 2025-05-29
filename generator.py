import os

import torch
import torch.nn as nn

from utils.tokenizer import Tokenizer
from models.transformerLM import TransformerLanguageModel

checkpoint_path = "/home/ros/repos/Mini_Transfomer_Project/training_experiments/transformerLM_ep50_b32_lr0.001_dataset_alice_in_wonderland.txt/model_epoch_50.pth"
txt_file_path = "data/alice_in_wonderland.txt"

tokenizer = Tokenizer(txt_file_path)
vocab_size = tokenizer.vocab_size()

seq_len = 256
embed_dim = 384
num_heads = 6
hidden_dim = 384
enc_ffn_h_dim = 1536
num_enc = 6
use_sinusoidal = True

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerLanguageModel(vocab_size, embed_dim, seq_len, hidden_dim, num_heads, enc_ffn_h_dim, num_enc, use_sinusoidal).to(device)

model.load_state_dict(torch.load(checkpoint_path))

model.eval()

seed_text = "Alice was beginning to get very tired"

generated_tokens = tokenizer.encode(seed_text)

if len(generated_tokens) > seq_len:
    input_tokens = torch.tensor(generated_tokens[-seq_len:]).unsqueeze(0).to(device)
else:    
    input_tokens = torch.tensor(generated_tokens).unsqueeze(0).to(device)


#Generate 100 words and use top-k and temperature sampling
temperature = 0.7
k = 5

with torch.no_grad():
    for _ in range(100):
        pred_logits = model(input_tokens)

        probs = torch.softmax(pred_logits[0, -1, :]/temperature, dim = -1)
        topk_probs, topk_indices = torch.topk(probs, k)
        next_token = torch.multinomial(topk_probs, 1)

        generated_tokens.append(next_token.cpu().item())

        next_token = next_token.to(device).unsqueeze(0)

        input_tokens = torch.cat((input_tokens, next_token), dim = 1)

        if input_tokens.shape[1] > seq_len:
            input_tokens = input_tokens[:, -seq_len:]
    
    generated_words = tokenizer.decode(generated_tokens)
    print(generated_words)