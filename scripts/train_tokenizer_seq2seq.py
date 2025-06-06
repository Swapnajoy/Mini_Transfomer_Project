from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers, pre_tokenizers

import os

train_en_path = 'data/iwslt2017_en_de/train_en.txt'
train_de_path = 'data/iwslt2017_en_de/train_de.txt'

tokenizer_save_path = 'tokenizers/seq2seq_shared_tokenizer.json'

files = [train_en_path, train_de_path]

tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

tokenizer.normalizer = normalizers.Sequence([NFKC()])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
)

tokenizer.train(files, trainer)

tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)
tokenizer.save(tokenizer_save_path)

print(f"Tokenizer saved to {tokenizer_save_path}")