import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

#See utils for available tokenizer options. Update experiment_name accordingly.
from utils.ch_tokenizer import CharTokenizer
from datasets.text_dataset import TextDataset
from models.decoder_only.dec_only_transformer import DecoderOnlyTransformer

from config_decoder_only import DATASET_PATH, SEQ_LEN, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, CHECKPOINT_PREFIX, SAVE_FREQ

txt_file_path = DATASET_PATH
seq_len = SEQ_LEN

tokenizer = CharTokenizer(txt_file_path)
vocab_size = tokenizer.vocab_size
MODEL_CONFIG['vocab_size'] = vocab_size

print(f"Dataset used:{txt_file_path.split('/')[-1]}, Vocab Size:{vocab_size}")

with open(txt_file_path, 'r', encoding='utf-8') as f:
    full_text = f.read()

split_idx = int(0.8 * len(full_text))
train_text = full_text[:split_idx]
val_text = full_text[split_idx:]

train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)

train_dataset = TextDataset(train_tokens, seq_len)
val_dataset = TextDataset(val_tokens, seq_len)

batch_size = TRAIN_CONFIG['batch_size']
lr = TRAIN_CONFIG['lr']
epochs = TRAIN_CONFIG['epochs']
num_workers = TRAIN_CONFIG['num_workers']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

embed_dim = MODEL_CONFIG['embed_dim']
num_heads = MODEL_CONFIG['num_heads']
hidden_dim = MODEL_CONFIG['hidden_dim']
dec_ffn_h_dim = MODEL_CONFIG['dec_ffn_h_dim']
num_dec = MODEL_CONFIG['num_dec']
use_sinusoidal = MODEL_CONFIG['use_sinusoidal']

dataset_name = os.path.basename(txt_file_path)
experiment_name = f"ep{epochs}_b{batch_size}_lr{lr}_dataset_{dataset_name}_token_ch"
experiment_dir = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX, experiment_name)
log_path = os.path.join(experiment_dir, "training_info.txt")

os.makedirs(experiment_dir, exist_ok=True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = DecoderOnlyTransformer(vocab_size, embed_dim, seq_len, hidden_dim, num_heads, dec_ffn_h_dim, num_dec, use_sinusoidal=use_sinusoidal).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,training_loss,validation_loss,perplexity\n")

#Training Loop
print(f"Entering training loop")

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for idx, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)

        optimizer.zero_grad()
        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    training_loss = running_loss/steps_per_epoch 
    print(f"epoch:{(epoch+1)}/{epochs}, avg. loss:{training_loss:.5f}")

    if (epoch+1) % SAVE_FREQ == 0:

        model.eval()
        
        with torch.no_grad():
            running_loss = 0
            for idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                pred, _ = model(x)

                loss = criterion(pred.view(-1, vocab_size), y.view(-1))
                
                running_loss += loss.item()

            val_loss = running_loss/len(val_loader)
            perplexity = torch.exp(torch.tensor(val_loss)).item()

        checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        print(f"epoch : {epoch+1}, training_loss : {training_loss:.5f}, validation_loss : {val_loss:.5f}, perplexity : {perplexity:.5f}\n")
        log_line = f"{epoch+1},{training_loss:.5f},{val_loss:.5f},{perplexity:.5f}\n"
        with open(os.path.join(experiment_dir, "training_info.txt"), 'a') as f:
            f.write(log_line)

print("Training Completed.")

model.eval()

with torch.no_grad():

    running_loss = 0
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)

        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        
        running_loss += loss.item()

    validation_loss = running_loss/len(val_loader)
    print(f"Overall validation loss:{validation_loss:.3f}")

final_model_path = os.path.join(experiment_dir, "model_final.pth")
torch.save(model.state_dict(), final_model_path)