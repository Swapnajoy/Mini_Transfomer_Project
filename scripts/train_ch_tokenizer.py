import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ch_tokenizer import CharTokenizer

txt_file_path1 = "data/alice_in_wonderland.txt"
txt_file_path2 = "data/tiny_shakespeare.txt"

tokenizer = CharTokenizer(txt_file_path1)
tokenizer.save("tokenizers/alice_ch_tokenizer.json")
print("Saving alice_ch_tokenizer.json")

tokenizer = CharTokenizer(txt_file_path2)
tokenizer.save("tokenizers/shakespeare_ch_tokenizer.json")
print("Saving shakespeare_ch_tokenizer.json")