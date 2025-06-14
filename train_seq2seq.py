import os
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.seq2seq_dataset import Seq2SeqDataset
from models.encoder_decoder.transformer_seq2seq import TransformerSeq2Seq
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer

batch_size = 64
lr = 0.0003
epochs = 40
num_workers = 4
seq_len = 64

train_src_path = 'data/iwslt2017_en_de/train_de.txt'
train_tgt_path = 'data/iwslt2017_en_de/train_en.txt'
val_src_path = 'data/iwslt2017_en_de/val_de.txt'
val_tgt_path = 'data/iwslt2017_en_de/val_en.txt'

tokenizer = Tokenizer.from_file('tokenizers/seq2seq_shared_tokenizer.json')

with open(train_src_path, 'r', encoding='utf-8') as f, open(train_tgt_path, 'r', encoding='utf-8') as g:
    de_sentences = f.readlines()
    en_sentences = g.readlines()

train_dataset = Seq2SeqDataset(de_sentences, en_sentences, tokenizer, max_seq_len=seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

with open(val_src_path, 'r', encoding='utf-8') as f, open(val_tgt_path, 'r', encoding='utf-8') as g:
    de_sentences = f.readlines()
    en_sentences = g.readlines()

val_dataset = Seq2SeqDataset(de_sentences, en_sentences, tokenizer, max_seq_len=seq_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

steps_per_epoch = len(train_loader)
max_steps = epochs * steps_per_epoch

vocab_size = tokenizer.get_vocab_size()
embed_dim = 256
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


eos_id = tokenizer.token_to_id('[EOS]')
pad_id = tokenizer.token_to_id('[PAD]')
token_weights = torch.ones(vocab_size).to(device)
token_weights[eos_id] = 2.0
token_weights[eos_id] = 0.0

label_smoothing = 0.1

criterion = nn.CrossEntropyLoss(weight=token_weights, label_smoothing=label_smoothing, reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_steps)

CHECKPOINT_DIR = 'training_experiments'
CHECKPOINT_PREFIX = 'seq2seq'
dataset_name = 'iwslt2017_en_de'
SAVE_FREQ = 1

timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

experiment_name = f"ep{epochs}_b{batch_size}_lr{lr}_dataset_{dataset_name}_token_bpe_{timestamp}"
experiment_dir = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX, experiment_name)
log_path = os.path.join(experiment_dir, "training_info.txt")

os.makedirs(experiment_dir, exist_ok=True)

if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch, training_loss, validation_loss, perplexity\n")

print(f"Entering training loop")

for epoch in range(epochs):
    model.train()
    running_loss = 0
    eos_correct = 0
    eos_total = 0

    for item in tqdm(train_loader):
        src = item['src'].to(device)
        tgt = item['tgt'].to(device)
        label = item['label'].to(device)

        pred, _, _ = model(src, tgt)

        optimizer.zero_grad()

        logits = pred.view(-1, vocab_size)
        labels = label.view(-1)

        # Base loss (token-wise)
        loss_raw = criterion(logits, labels)
        loss_raw = loss_raw.view(label.size())  # shape: (B, T)

        # EOS-aware loss masking
        with torch.no_grad():
            mask = (label != pad_id).float()
            pred_ids = torch.argmax(pred, dim=-1)

            for i in range(label.size(0)):
                eos_pos = (label[i] == eos_id).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    eos_idx = eos_pos[0].item()
                    mask[i, eos_idx + 1:] = 0.0

                    pred_eos_pos = (pred_ids[i] == eos_id).nonzero(as_tuple=True)[0]
                    if pred_eos_pos.numel() > 0 and pred_eos_pos[0].item() == eos_idx:
                        eos_correct += 1
                    eos_total += 1

        loss = (loss_raw * mask).sum() / mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    training_loss = running_loss / steps_per_epoch
    eos_accuracy = eos_correct / eos_total if eos_total > 0 else 0.0
    #print(f"epoch:{epoch+1}/{epochs}, avg. loss:{training_loss:.5f}, EOS Accuracy: {eos_accuracy:.3f}")

    # Validation + Save
    if (epoch + 1) % SAVE_FREQ == 0:
        model.eval()
        val_loss_total = 0
        eos_correct = 0
        eos_total = 0

        with torch.no_grad():
            for item in val_loader:
                src = item['src'].to(device)
                tgt = item['tgt'].to(device)
                label = item['label'].to(device)

                pred, _, _ = model(src, tgt)

                logits = pred.view(-1, vocab_size)
                labels = label.view(-1)

                loss_raw = criterion(logits, labels)
                loss_raw = loss_raw.view(label.size())

                mask = (label != pad_id).float()
                pred_ids = torch.argmax(pred, dim=-1)

                for i in range(label.size(0)):
                    eos_pos = (label[i] == eos_id).nonzero(as_tuple=True)[0]
                    if eos_pos.numel() > 0:
                        eos_idx = eos_pos[0].item()
                        mask[i, eos_idx + 1:] = 0.0

                        pred_eos_pos = (pred_ids[i] == eos_id).nonzero(as_tuple=True)[0]
                        if pred_eos_pos.numel() > 0 and pred_eos_pos[0].item() == eos_idx:
                            eos_correct += 1
                        eos_total += 1

                loss = (loss_raw * mask).sum() / mask.sum()
                val_loss_total += loss.item()

        val_loss = val_loss_total / len(val_loader)
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        val_eos_accuracy = eos_correct / eos_total if eos_total > 0 else 0.0

        checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        print(f"epoch : {epoch+1}, training_loss : {training_loss:.5f}, validation_loss : {val_loss:.5f}, perplexity : {perplexity:.5f}, EOS Accuracy : {val_eos_accuracy:.3f}")

        log_line = f"{epoch+1},{training_loss:.5f},{val_loss:.5f},{perplexity:.5f},{val_eos_accuracy:.3f}\n"
        with open(log_path, 'a') as f:
            f.write(log_line)

print("Training Completed.")

"""
model.eval()

with torch.no_grad():

    running_loss = 0
    for item in val_loader:
        src = item['src'].to(device)
        tgt = item['tgt'].to(device)
        label = item['label'].to(device)

        pred, _, _ = model(src, tgt)

        loss = criterion(pred.view(-1, vocab_size), label.view(-1))
                
        running_loss += loss.item()

    validation_loss = running_loss/len(val_loader)
    print(f"Overall validation loss:{validation_loss:.3f}")

final_model_path = os.path.join(experiment_dir, "model_final.pth")
torch.save(model.state_dict(), final_model_path)
"""

