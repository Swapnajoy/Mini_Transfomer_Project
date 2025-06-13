import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.word_tokenizer import WordTokenizer

txt_file_path1 = "data/alice_in_wonderland.txt"
txt_file_path2 = "data/tiny_shakespeare.txt"

tokenizer = WordTokenizer(txt_file_path1)
tokenizer.save("tokenizers/alice_word_tokenizer.json")
print("Saving alice_word_tokenizer.json")

tokenizer = WordTokenizer(txt_file_path2)
tokenizer.save("tokenizers/shakespeare_word_tokenizer.json")
print("Saving shakespeare_word_tokenizer.json")