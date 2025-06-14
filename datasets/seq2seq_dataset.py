import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('tokenizers/seq2seq_shared_tokenizer.json')

def encode_pair(src, tgt, tokenizer, max_seq_len=128):
    src_encoded = tokenizer.encode(src)
    tgt_encoded = tokenizer.encode(tgt)

    src_ids = src_encoded.ids[:max_seq_len]
    tgt_ids = tgt_encoded.ids[:max_seq_len-1]
    
    eos_id = tokenizer.token_to_id('[EOS]')
    tgt_ids.append(eos_id)

    return src_ids, tgt_ids

class Seq2SeqDataset(Dataset):
    def __init__(self, src_list, tgt_list, tokenizer, max_seq_len=128):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')


    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        src_ids, tgt_ids = encode_pair(self.src_list[idx], self.tgt_list[idx], self.tokenizer, self.max_seq_len)

        src_ids += [self.pad_id] * (self.max_seq_len - len(src_ids))
        tgt_ids += [self.pad_id] * (self.max_seq_len - len(tgt_ids))

        dec_in = [self.bos_id] + tgt_ids[:-1]
        dec_tgt = tgt_ids

        return{
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(dec_in, dtype=torch.long),
            'label': torch.tensor(dec_tgt, dtype=torch.long)
        }
