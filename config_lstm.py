import os

# Dataset Configurations
# ======================

DATASET_PATH = "data/alice_in_wonderland.txt"
SEQ_LEN = 128

# Model Configurations
# ====================

MODEL_CONFIG = {
    "embed_dim" : 128,
    "hidden_dim" : 256,
    "vocab_size" : None,
    }

# Training Configurations
# ====================

TRAIN_CONFIG = {
    "batch_size" : 64,
    "lr" : 0.00005,
    "epochs" : 20,
    "num_workers" : 4,
    }

# Checkpoint Configurations
# =========================

CHECKPOINT_DIR = "training_experiments"
CHECKPOINT_PREFIX = "lstm"
SAVE_FREQ = 2

# Generator Configurations
# ========================

CHECKPOINT_PATH = "training_experiments/lstm/ep20_b64_lr5e-05_dataset_alice_in_wonderland.txt_token_ch/model_epoch_8.pth"
TXT_FILE_PATH = "data/alice_in_wonderland.txt"
TOKENIZER_PATH = "tokenizers/alice_ch_tokenizer.json"

# Top-K Sampling Configurations
# =============================

TOPK_CONFIG = {
    "temperature" : 0.7,
    "k" : 5,
}