import os

# Dataset Configurations
# ======================

DATASET_PATH = "data/tiny_shakespeare.txt"
SEQ_LEN = 256

# Model Configurations
# ====================

MODEL_CONFIG = {
    "embed_dim" : 256,
    "num_heads" : 4,
    "hidden_dim" : 256,
    "dec_ffn_h_dim" : 1024,
    "num_dec" : 4,
    "use_sinusoidal" : True,
    "vocab_size" : None,
    }

# Training Configurations
# ====================

TRAIN_CONFIG = {
    "batch_size" : 64,
    "lr" : 0.0002,
    "epochs" : 20,
    "num_workers" : 8,
    }

# Checkpoint Configurations
# =========================

CHECKPOINT_DIR = "training_experiments"
CHECKPOINT_PREFIX = "decoder_only"
SAVE_FREQ = 1

# Generator Configurations
# ========================

CHECKPOINT_PATH = ""
TXT_FILE_PATH = "data/tiny_shakespeare.txt"

# Top-K Sampling Configurations
# =============================

TOPK_CONFIG = {
    "temperature" : 0.7,
    "k" : 5,
}