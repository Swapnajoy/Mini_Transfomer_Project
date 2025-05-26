import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.text_dataset import TextDataset
from models.transformerLM import TransformerLanguageModel

txt_file_path = "data/alice_in_wonderland.txt"
seq_len = 8

dataset = TextDataset(txt_file_path, seq_len=seq_len)

vocab_size = dataset.vocab_size

batch_size = 32
lr = 0.0003
epochs = 30
num_workers = 0

embed_dim = 128
num_heads = 4
hidden_dim = 128
enc_ffn_h_dim = 512
num_enc = 6
use_sinusoidal = True

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerLanguageModel(vocab_size, embed_dim, seq_len, hidden_dim, num_heads, enc_ffn_h_dim, num_enc, use_sinusoidal=use_sinusoidal).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

num_steps = len(dataloader)

#Training Loop
print(f"Entering training loop")
model.train()
for epoch in range(epochs):
    running_loss = 0
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        optimizer.zero_grad()
        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx+1)%20 == 0:
            print(f"Step:{(idx+1)}/{num_steps}, loss: {loss.item():.3f}")
         
    print(f"epoch:{(epoch+1)}/{epochs}, avg. loss:{running_loss/num_steps}")
