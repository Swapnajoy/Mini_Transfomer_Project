import os

from datasets import load_dataset

dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

train_dataset = dataset['train']
val_dataset = dataset['validation']

data_path = 'data/iwslt2017_en_de'
os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, 'train_en.txt'), 'w', encoding='utf-8') as f, open(os.path.join(data_path, 'train_de.txt'), 'w', encoding='utf-8') as g:
    for item in train_dataset:
        f.write(item['translation']['en'] + '\n')
        g.write(item['translation']['de'] + '\n')


with open(os.path.join(data_path, 'val_en.txt'), 'w', encoding='utf-8') as f, open(os.path.join(data_path, 'val_de.txt'), 'w', encoding='utf-8') as g:
    for item in val_dataset:
        f.write(item['translation']['en'] + '\n')
        g.write(item['translation']['de'] + '\n')
