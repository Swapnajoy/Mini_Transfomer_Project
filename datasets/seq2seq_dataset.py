import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('okenizers/seq2seq_shared_tokenizer.json')

with open('data/iwslt2017_en_de/train_de.txt', 'r', encoding='utf-8') as f, open('data/iwslt2017_en_de/train_en.txt', 'r', encoding='utf-8') as g:
    de_sentences = f.readlines()
    en_sentences = g.readlines()

def encode_pair(src, tgt, tokenizer, max_seq_len=128):
    src_encoded = tokenizer.encode(src)
    tgt_encoded = tokenizer.encode(tgt)

    src_ids = src_encoded.ids[:max_seq_len]
    tgt_ids = tgt_encoded.ids[:max_seq_len]

    return src_ids, tgt_ids

class Seq2SeqDataset(Dataset):
    def __init__(self, src_list, tgt_list, tokenizer, max_seq_len=128):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.token_to_id('<pad>')

    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        src_ids, tgt_ids = encode_pair(self.src_list[idx], self.tgt_list[idx], self.tokenizer, self.max_seq_len)
        src_ids += [self.pad_id] * (self.max_seq_len - len(src_ids))
        tgt_ids += [self.pad_id] * (self.max_seq_len - len(tgt_ids))

        dec_in = [self.tokenizer.token_to_id('<s>')] + tgt_ids[:-1]
        dec_tgt = tgt_ids

        return{
            'src': torch.tensor(src_ids),
            'tgt': torch.tensor(dec_in),
            'label': torch.tensor(dec_tgt)
        }
