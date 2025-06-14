import os

# Dataset Configurations
# ======================

DATASET_PATH = "data/iwslt2017_en_de"
SEQ_LEN = 64

# Model Configurations
# ====================

MODEL_CONFIG = {
    "embed_dim" : 256,
    "num_heads" : 4,
    "hidden_dim" : 256,
    "enc_ffn_h_dim" : 1024,
    "dec_ffn_h_dim" : 1024,
    "num_enc" : 4,
    "num_dec" : 4,
    "use_sinusoidal" : True,
    "vocab_size" : None,
    }

# Training Configurations
# ====================

TRAIN_CONFIG = {
    "batch_size" : 64,
    "lr" : 0.0003,
    "epochs" : 40,
    "num_workers" : 4,
    }

# Checkpoint Configurations
# =========================

CHECKPOINT_DIR = "training_experiments"
CHECKPOINT_PREFIX = "seq2seq"
SAVE_FREQ = 1

# Generator Configurations
# ========================

CHECKPOINT_PATH = "training_experiments/encoder_only/ep30_b64_lr0.0006_dataset_alice_in_wonderland.txt_token_ch/model_epoch_15.pth"
TXT_FILE_PATH = "data/alice_in_wonderland.txt"

# Top-K Sampling Configurations
# =============================

TOPK_CONFIG = {
    "temperature" : 0.7,
    "k" : 5,
}