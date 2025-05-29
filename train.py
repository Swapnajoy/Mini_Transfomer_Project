import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.tokenizer import Tokenizer
from datasets.text_dataset import TextDataset
from models.transformerLM import TransformerLanguageModel

txt_file_path = "data/alice_in_wonderland.txt"
seq_len = 256

tokenizer = Tokenizer(txt_file_path)
vocab_size = tokenizer.vocab_size

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

batch_size = 64
lr = 0.0006
epochs = 50
num_workers = 4

embed_dim = 384
num_heads = 6
hidden_dim = 384
enc_ffn_h_dim = 1536
num_enc = 6
use_sinusoidal = True

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
total_steps = epochs * (len(train_dataset) // batch_size)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

steps_per_epoch = len(train_loader)

#Training Loop
print(f"Entering training loop")
model.train()

for epoch in range(epochs):
    running_loss = 0
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        optimizer.zero_grad()
        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (idx+1)%(steps_per_epoch//5) == 0:
            print(f"Step:{(idx+1)}/{steps_per_epoch}, loss: {loss.item():.3f}")
         
    print(f"epoch:{(epoch+1)}/{epochs}, avg. loss:{running_loss/steps_per_epoch:.3f}")

    if epoch % 10 == 0:
        checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

print("Training Completed.")

model.eval()

with torch.no_grad():

    running_loss = 0
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        running_loss += loss.item()

    validation_loss = running_loss/(idx+1)
    print(f"Overall validation loss:{validation_loss:.3f}")

training_info = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "dataset": dataset_name,
    "validation_loss": validation_loss
}

with open(os.path.join(experiment_dir, "training_info.txt"), 'w') as f:
    for key, value in training_info.items():
        f.write(f"{key}: {value}\n")