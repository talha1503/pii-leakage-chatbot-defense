# """
# Configuration file for PII Leakage Analysis Project
# """

# import torch

# class Config:
#     # Model Configuration
#     MODEL_NAME = "gpt2"  # Base model
#     MAX_LENGTH = 256
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Data Configuration
#     NUM_TRAIN_SAMPLES = 1000
#     NUM_VAL_SAMPLES = 200
#     NUM_TEST_SAMPLES = 200
#     PII_INJECTION_RATE = 0.8  # 80% of conversations contain PII
    
#     # PII Types to inject
#     PII_TYPES = [
#         "name",
#         "email",
#         "phone",
#         "ssn",
#         "credit_card",
#         "address",
#         "date_of_birth",
#         "medical_info"
#     ]
    
#     # Training Configuration
#     BATCH_SIZE = 4
#     GRADIENT_ACCUMULATION_STEPS = 4
#     LEARNING_RATE = 5e-5
#     NUM_EPOCHS_SFT = 3
#     NUM_EPOCHS_DPO = 2
#     NUM_EPOCHS_GRPO = 1
#     WARMUP_STEPS = 100
    
#     # DPO Configuration
#     DPO_BETA = 0.1  # KL penalty coefficient
    
#     # GRPO Configuration
#     GRPO_NUM_SAMPLES = 2  # Number of samples per prompt for ranking
#     GRPO_TEMPERATURE = 0.7
    
#     # Privacy Reward Configuration
#     PII_LEAK_PENALTY = -1.0
#     NO_LEAK_REWARD = 1.0
#     UTILITY_WEIGHT = 0.5  # Balance between privacy and utility
    
#     # Attack Configuration
#     NUM_ATTACK_QUERIES = 100
#     ATTACK_TYPES = ["direct", "indirect", "contextual"]
    
#     # Evaluation Configuration
#     EVAL_BATCH_SIZE = 8
    
#     # Paths
#     DATA_DIR = "./data"
#     MODEL_DIR = "./models"
#     RESULTS_DIR = "./results"
    
#     # Random Seed
#     SEED = 42



"""
Configuration file for PII Leakage Analysis Project
"""

import torch

class Config:
    # Model Configuration
    MODEL_NAME = "gpt2"
    MAX_LENGTH = 256
    
    # USE MPS FOR M4 MAC
    if torch.backends.mps.is_available():
        DEVICE = "mps"  # Apple Silicon GPU
        print("✓ Using M4 GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        DEVICE = "cuda"  # NVIDIA GPU
        print("✓ Using CUDA GPU")
    else:
        DEVICE = "cpu"
        print("⚠ Using CPU")
    
    # Rest of your config stays the same
    NUM_TRAIN_SAMPLES = 1000
    NUM_VAL_SAMPLES = 200
    NUM_TEST_SAMPLES = 200
    PII_INJECTION_RATE = 0.8
    
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS_SFT = 3
    NUM_EPOCHS_DPO = 2
    NUM_EPOCHS_GRPO = 2
    WARMUP_STEPS = 100
    
    DPO_BETA = 0.1
    GRPO_NUM_SAMPLES = 4
    GRPO_TEMPERATURE = 0.7
    
    PII_LEAK_PENALTY = -1.0
    NO_LEAK_REWARD = 1.0
    UTILITY_WEIGHT = 0.5
    
    NUM_ATTACK_QUERIES = 100
    ATTACK_TYPES = ["direct", "indirect", "contextual"]
    
    EVAL_BATCH_SIZE = 8
    
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    RESULTS_DIR = "./results"
    
    SEED = 42