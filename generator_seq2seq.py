import torch
import torch.nn as nn

from models.encoder_decoder.transformer_seq2seq import TransformerSeq2Seq
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('tokenizers/seq2seq_shared_tokenizer.json')

device = ('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = tokenizer.get_vocab_size()
embed_dim = 256
seq_len = 128
hidden_dim = 256
num_heads = 4
enc_ffn_h_dim = 1024
dec_ffn_h_dim = 1024
num_enc = 4
num_dec = 4
use_sinusoidal = True

model = TransformerSeq2Seq(vocab_size=vocab_size, 
                           embed_dim=embed_dim, 
                           seq_len=seq_len, 
                           hidden_dim=hidden_dim, 
                           num_heads=num_heads,
                           enc_ffn_h_dim=enc_ffn_h_dim,
                           dec_ffn_h_dim=dec_ffn_h_dim,
                           num_enc=num_enc,
                           num_dec=num_dec,
                           use_sinusoidal=use_sinusoidal
                           ).to(device)

checkpoint_path = 'training_experiments/seq2seq/ep30_b64_lr0.0003_dataset_iwslt2017_en_de_token_ch/model_epoch_25.pth'

model.load_state_dict(torch.load(checkpoint_path))

seed_text = input("Enter seed text:")

enc_ids = tokenizer.encode(seed_text).ids
enc_in = torch.tensor(enc_ids, dtype=torch.long).unsqueeze(0).to(device)

bos_id = tokenizer.token_to_id('[BOS]')
dec_in = torch.tensor([[bos_id]], dtype=torch.long).to(device)

eos_id = tokenizer.token_to_id('[EOS]')

temperature = 0.7
k = 5

while True:

    pred, _, _ = model(enc_in, dec_in)

    probs = torch.softmax(pred[0, -1, :]/temperature, dim = -1)

    topk_probs, topk_indices = torch.topk(probs, k)
    next_token = torch.multinomial(topk_probs, 1).item()
    next_token_id = topk_indices[next_token].unsqueeze(0)

    dec_in = torch.cat([dec_in, next_token_id.unsqueeze(0)], dim=1)

    if next_token_id.item() == eos_id or dec_in.size(1) >= seq_len:
        break

dec_out = dec_in[0].tolist()
decoded = tokenizer.decode(dec_out, skip_special_tokens=True)
translated = decoded.replace("Ġ", " ").replace("Ċ", "\n").strip()

print(f"Translation: {translated}")