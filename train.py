import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.ch_tokenizer import CharTokenizer
from datasets.text_dataset import TextDataset
from models.transformerLM import TransformerLanguageModel

from config import DATASET_PATH, SEQ_LEN, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, CHECKPOINT_PREFIX, SAVE_FREQ

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

embed_dim = MODEL_CONFIG['embed_dim']
num_heads = MODEL_CONFIG['num_heads']
hidden_dim = MODEL_CONFIG['hidden_dim']
enc_ffn_h_dim = MODEL_CONFIG['enc_ffn_h_dim']
num_enc = MODEL_CONFIG['num_enc']
use_sinusoidal = MODEL_CONFIG['use_sinusoidal']

dataset_name = os.path.basename(txt_file_path)
experiment_name = f"transformerLM_ep{epochs}_b{batch_size}_lr{lr}_dataset_{dataset_name}"
experiment_dir = os.path.join("training_experiments", experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerLanguageModel(vocab_size, embed_dim, seq_len, hidden_dim, num_heads, enc_ffn_h_dim, num_enc, use_sinusoidal=use_sinusoidal).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

#Training Loop
print(f"Entering training loop")

training_info = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "dataset": dataset_name,
}

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
        training_info[f'{epoch+1} epoch training_loss'] = training_loss

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
            print(f"Validation loss after {epoch+1} epochs:{validation_loss:.3f}")
            training_info[f'{epoch+1} epoch val_loss'] = validation_loss

        checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

print("Training Completed.")

model.eval()

with torch.no_grad():

    running_loss = 0
    for idx, (x, y) in enumerate(tqdm(val_loader)):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        
        running_loss += loss.item()

    validation_loss = running_loss/len(val_loader)
    print(f"Overall validation loss:{validation_loss:.3f}")

training_info['Overall_validation_loss'] = validation_loss

with open(os.path.join(experiment_dir, "training_info.txt"), 'w') as f:
    for key, value in training_info.items():
        f.write(f"{key}: {value}\n")