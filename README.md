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



Document the observations during:

  training for loss curves, overfitting, plateauing etc. Experiment with the hyperparameters.
  
  inference, how the generated text compares with the other models. Generalization capability.
  
Visualize the results

Finally build a mini transformer based language model.

In Progress
