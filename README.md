# 📌 Overview
This project explores the architecture, training, and inference of large language models from scratch using PyTorch. It also revisits the LSTM architecture from the pre-transformer era to provide historical and performance-based comparison.

The following model types are implemented and analyzed:

* Encoder-only Transformers (e.g., BERT-style)

* Decoder-only Transformers (e.g., GPT-style)

* Full Seq2Seq Transformers (e.g., for machine translation)

* Baseline RNN model using LSTM

# 📚 Datasets Used
All datasets used in this project are stored in the `data/` folder.
To evaluate and compare various model architectures, the following datasets were employed:

1. Tiny Shakespeare
    - Size: ~1MB of text.
    - Use Case: Useful for quick experimentation with character-level generation due to its small size and stylistic consistency.
    - Applied in: Encoder-only, decoder-only, and LSTM models.

2. Alice in Wonderland by Lewis Carroll
    - Size: ~150KB of text.
    - Use Case: Classic English literature offering structured narrative text, used for experimenting with character-level and word-level tokenizations owing o its small vocabulary size.
    - Applied in: Encoder-only, decoder-only, and LSTM models.

3. IWSLT2017 (EN↔DE Translation)
    - Size: ~40MB of ~200K sentence pairs.
    - Type: Parallel English-German corpus
    - Use Case: Designed for benchmarking machine translation models.
    - Applied in: Seq2Seq Transformer with shared Byte-Pair Encoding (BPE) tokenizer.

# 🧩 Tokenization Strategies
Tokenization plays a crucial role in how language models process and understand text. This project experiments with multiple tokenization schemes to analyze their effect on model performance and generalization.

1. Character-Level Tokenizer
    - Type: Custom, from-scratch implementation -> `utils/ch_tokenizer.py`.
    - Vocabulary: All unique characters from the dataset.
    - Pros: Simple to implement and captures fine-grained detail.
    - Cons: Longer sequences and lower convergence due to sparse signal.

2. Word-Level Tokenizer (Custom)
    - Type: Custom tokenizer splitting on whitespace and punctuation -> `utils/word_tokeinizer.py`
    - Vocabulary: Built manually from training corpus.
    - Pros: More semantic understanding than character-level and faster convergence.
    - Cons: Vocabulary explosion with rare or compound words and poor generalization to unseen words.

3. Byte-Pair Encoding (BPE)
    - Type: Subword-level tokenizer using HuggingFace's tokenizers library.
    - Shared Vocabulary: Trained on combined English-German corpus from IWSLT2017.
    - Training script `scripts/train_tokenizer_seq2seq.py`.
    - Trained tokenizer `tokenizers/seq2seq_shared_tokenizer.json`.
    - Pros: Handles rare words and morphology gracefully, vocabulary size can be tuned and suitable for multilingual tasks.
    - Cons: Slightly more complex setup.

# 📦 Dataset Classes
Custom PyTorch `Dataset` classes were implemented to efficiently handle training samples for different model-tokenizer combinations.

1. `TextDataset` – for Character/Word Tokenizers
    - Used with: Character-level and word-level tokenizers.
    - Purpose: Prepares fixed-length input (x) and target (y) sequences for next-token prediction.
    - Mechanism: For each sample, `x = tokens[i : i+seq_len], y = tokens[i+1 : i+1+seq_len]`
    - Output: (x, y) as integer token sequences.

2. `Seq2SeqDataset` – for BPE-based Translation
    - Used with: BPE tokenizer (`seq2seq_shared_tokenizer.json`)
    - Purpose: Prepares padded input-output sentence pairs for Seq2Seq translation tasks.
    - Mechanism: Tokenizes and pads both `src` and `tgt` sequences and generatesencoder input, decoder input with `[BOS]` and decoder target (right shifted).
    - Output: Dictionary with `src`, `tgt`, and `label` tensors.

Document:

  training for loss curves, overfitting, plateauing etc. Experiment with the hyperparameters.
  
  inference, how the generated text compares with the other models. Generalization capability.
  
Visualize the results

Finally build a mini transformer based language model

In Progress
