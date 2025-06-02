import os

# Dataset Configurations
# ======================

DATASET_PATH = "data/alice_in_wonderland.txt"
SEQ_LEN = 128

# Model Configurations
# ====================

MODEL_CONFIG = {
    "embed_dim" : 384,
    "num_heads" : 6,
    "hidden_dim" : 384,
    "enc_ffn_h_dim" : 1536,
    "num_enc" : 6,
    "use_sinusoidal" : True,
    "vocab_size" : None,
    }

# Training Configurations
# ====================

TRAIN_CONFIG = {
    "batch_size" : 64,
    "lr" : 0.0006,
    "epochs" : 30,
    "num_workers" : 4,
    }

# Checkpoint Configurations
# =========================

CHECKPOINT_DIR = "training_experiments"
CHECKPOINT_PREFIX = "transformerLM"
SAVE_FREQ = 3

# Generator Configurations
# ========================

CHECKPOINT_PATH = "training_experiments/transformerLM/ep30_b64_lr0.0006_dataset_alice_in_wonderland.txt_token_ch/model_epoch_12.pth"
TXT_FILE_PATH = "data/alice_in_wonderland.txt"

# Top-K Sampling Configurations
# =============================

TOPK_CONFIG = {
    "temperature" : 0.7,
    "k" : 5,
}