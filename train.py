import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets.text_dataset import TextDataset
from models.transformerLM import TransformerLanguageModel

txt_file_path = "data/alice_in_wonderland.txt"
seq_len = 8

dataset = TextDataset(txt_file_path, seq_len=seq_len)
vocab_size = dataset.vocab_size

train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
lr = 0.0003
epochs = 20
num_workers = 0

embed_dim = 128
num_heads = 4
hidden_dim = 128
enc_ffn_h_dim = 512
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

num_steps = len(train_loader)

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

        running_loss += loss.item()

        if (idx+1)%50 == 0:
            print(f"Step:{(idx+1)}/{num_steps}, loss: {loss.item():.3f}")
         
    print(f"epoch:{(epoch+1)}/{epochs}, avg. loss:{running_loss/num_steps}")

    if (epoch+1) % 10 == 0:
        checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

print("Training Completed.")

model.eval()

with torch.no_grad():

    running_loss = 0
    for idx, (x, y) in enumerate(train_loader):
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