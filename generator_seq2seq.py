import torch
import torch.nn as nn

from models.encoder_decoder.transformer_seq2seq import TransformerSeq2Seq
from tokenizers import Tokenizer
from config_seq2seq import MODEL_CONFIG, SEQ_LEN, CHECKPOINT_PATH, TOPK_CONFIG

tokenizer = Tokenizer.from_file('tokenizers/seq2seq_shared_tokenizer.json')

device = ('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = tokenizer.get_vocab_size()
seq_len = SEQ_LEN

embed_dim = MODEL_CONFIG['embed_dim']
hidden_dim = MODEL_CONFIG['hidden_dim']
num_heads = MODEL_CONFIG['num_heads']
enc_ffn_h_dim = MODEL_CONFIG['enc_ffn_h_dim']
dec_ffn_h_dim = MODEL_CONFIG['dec_ffn_h_dim']
num_enc = MODEL_CONFIG['num_enc']
num_dec = MODEL_CONFIG['num_dec']
use_sinusoidal = MODEL_CONFIG['use_sinusoidal']

model = TransformerSeq2Seq(
    vocab_size=vocab_size, 
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

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

bos_id = tokenizer.token_to_id('[BOS]')
eos_id = tokenizer.token_to_id('[EOS]')
pad_id = tokenizer.token_to_id('[PAD]')

print("Enter a German sentence for translation or type exit to quit")


while True:
    seed_text = input("German: ").strip()
    if seed_text.lower() == 'exit':
        print('Exiting.')
        break

    count = 0
    enc_ids = tokenizer.encode(seed_text).ids
    enc_in = torch.tensor(enc_ids, dtype=torch.long).unsqueeze(0).to(device)

    dec_in = torch.tensor([[bos_id]], dtype=torch.long).to(device)

    temperature = TOPK_CONFIG['temperature']
    k = TOPK_CONFIG['k']
    max_len = 64

    while True:

        pred, _, _ = model(enc_in, dec_in)
        count += 1
        probs = torch.softmax(pred[0, -1, :]/temperature, dim = -1)

        topk_probs, topk_indices = torch.topk(probs, k)
        next_token = torch.multinomial(topk_probs, 1).item()
        next_token_id = topk_indices[next_token].unsqueeze(0)

        dec_in = torch.cat([dec_in, next_token_id.unsqueeze(0)], dim=1)

        if next_token_id.item() == eos_id or dec_in.size(1) >= max_len:
            break

    dec_out = dec_in[0].tolist()
    decoded = tokenizer.decode(dec_out, skip_special_tokens=True)
    translated = decoded.replace("Ġ", " ").replace("Ċ", "\n").strip()

    print(f"English: {translated}, \n token_count: {count}")