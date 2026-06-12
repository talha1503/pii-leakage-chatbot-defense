#!/usr/bin/env python
# coding: utf-8

# # PII Leakage Analysis in Chatbot Settings
# ## CS690F: Trustworthy and Responsible AI - Fall 2025
# 
# **Group 1:**
# - Swetha Krishnan
# - Talha Mohammed
# - Zakir Chafekar
# - Thasmitha Bangalure Shekhar
# - Anushka Agarwal
# 
# ### Project Overview
# This notebook implements a comprehensive analysis of PII leakage in chatbot settings, comparing multiple defense strategies:
# - **Baseline**: No privacy protection
# - **SFT + Scrubbing**: Supervised fine-tuning with PII filtering
# - **DPO**: Direct Preference Optimization with privacy-aware preferences
# - **GRPO**: Group Relative Policy Optimization with ranked privacy rewards
# 
# ---

# ## 1. Setup and Installation

# In[1]:



# Install required packages
# get_ipython().system('pip install -q transformers torch datasets faker spacy matplotlib seaborn tqdm')
# get_ipython().system('python -m spacy download en_core_web_sm')

# print("✓ All packages installed successfully!")


# ## 2. Check GPU Availability (IF RUNNING IN COLAB)

# In[2]:


import torch

if torch.cuda.is_available():
    print(f"✓ GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠ GPU not available, using CPU (training will be slower)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")


# ## 2. Check GPU Availability (IF Running in MAC with GPU)

# In[3]:


import torch

print("PyTorch Version:", torch.__version__)
print("\nMPS (Metal) Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✓ Using device: {device}")

    # Test tensor operation
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y  # Matrix multiplication on GPU
    print("✓ GPU test successful!")
    print(f"  Result shape: {z.shape}")
else:
    print("\n⚠ MPS not available")


# ## 3. Upload Project Files
# 
# Upload all the Python files:
# - `config.py`
# - `data.py`
# - `model.py`
# - `pii_detector.py`
# - `attack.py`
# - `evaluation.py`
# - `train.py`

# In[4]:


# Check if files are uploaded
import os

required_files = [
    'config.py', 'data.py', 'model.py', 'pii_detector.py',
    'attack.py', 'evaluation.py', 'train.py'
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("❌ Missing files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\nPlease upload all project files before continuing.")
else:
    print("✓ All required files found!")


# ## 4. Run Complete Experiment
# 
# This will run all phases:
# 1. Data generation with PII injection
# 2. Baseline model training
# 3. SFT with PII scrubbing
# 4. DPO training
# 5. GRPO training
# 6. Comprehensive evaluation
# 7. Failure analysis
# 
# **Note:** This will take approximately 2-3 hours on Colab Pro with GPU.

# In[5]:


# Run the complete pipeline
get_ipython().system('python train.py')


# ## 5. Alternative: Run Individual Phases
# 
# If you want more control, you can run phases separately:

# ### Phase 1: Data Generation

# In[6]:


from config import Config
from data import DataGenerator

config = Config()
generator = DataGenerator(config)
generator.save_datasets()

print("\n✓ Data generation complete!")


# ### Inspect Generated Data

# In[7]:


import json

# Load and display sample conversations
with open('./data/train.json', 'r') as f:
    train_data = json.load(f)

print(f"Total training samples: {len(train_data)}")
print(f"\nSample conversation:")
print("="*50)

sample = train_data[0]
for turn in sample['conversation']:
    print(f"{turn['role'].upper()}: {turn['content']}")

print(f"\nContains PII: {sample['contains_pii']}")
if sample['contains_pii']:
    print(f"PII Types: {sample['pii_leaked']}")
    print(f"\nProfile:")
    for key, value in sample['profile'].items():
        print(f"  {key}: {value}")


# ### Phase 2: Train Baseline Model

# In[8]:


from model import ChatbotModel

baseline_model = ChatbotModel(config)
baseline_model.train_sft(
    train_data_path='./data/train.json',
    val_data_path='./data/val.json',
    output_dir='./models/baseline',
    use_scrubbing=False
)

print("\n✓ Baseline model training complete!")


# ### Phase 3: Train SFT with Scrubbing

# In[9]:


sft_model = ChatbotModel(config)
sft_model.train_sft(
    train_data_path='./data/train.json',
    val_data_path='./data/val.json',
    output_dir='./models/sft_scrubbed',
    use_scrubbing=True
)

print("\n✓ SFT with scrubbing training complete!")


# ### Phase 4: Train DPO Model

# In[10]:


dpo_model = ChatbotModel(config)
dpo_model.load_model('./models/baseline')  # Start from baseline

dpo_model.train_dpo(
    dpo_data_path='./data/dpo_pairs.json',
    output_dir='./models/dpo'
)

print("\n✓ DPO model training complete!")


# ### Phase 5: Train GRPO Model

# In[11]:


grpo_model = ChatbotModel(config)
grpo_model.load_model('./models/baseline')  # Start from baseline

grpo_model.train_grpo(
    train_data_path='./data/train.json',
    output_dir='./models/grpo'
)

print("\n✓ GRPO model training complete!")


# ## 6. Test Individual Models
# 
# Test model responses interactively:

# In[18]:


# Load a model
test_model = ChatbotModel(config)
test_model.load_model('./models/baseline')  # Change to test different models

# Test prompt
test_prompt = """User: I need help with my account
Assistant:"""

response = test_model.generate_response(test_prompt)
print(f"Prompt: {test_prompt}")
print(f"\nResponse: {response}")

# Check for PII leakage
from pii_detector import PIIDetector
detector = PIIDetector()
detected_pii = detector.detect_pii(response)

if detected_pii:
    print(f"\n⚠ PII Detected:")
    for pii in detected_pii:
        print(f"  - {pii['type']}: {pii['value']}")
else:
    print(f"\n✓ No PII detected")


# ## 7. View Results
# 
# After running the complete pipeline, view the results:

# In[ ]:


# Display final report
with open('./results/final_report.txt', 'r') as f:
    report = f.read()
    print(report)


# In[ ]:


# Load detailed results
with open('./results/detailed_results.json', 'r') as f:
    detailed_results = json.load(f)

# Display key metrics
import pandas as pd

metrics_data = []
for model_name, results in detailed_results.items():
    metrics_data.append({
        'Model': model_name,
        'Leakage Rate': f"{results['privacy']['leakage_rate']:.2%}",
        'Task Success': f"{results['utility']['task_success_rate']:.2%}",
        'Attack Resistance': f"{1 - results['attack_resistance']['overall_success_rate']:.2%}",
        'Privacy-Utility Score': f"{results['privacy_utility_score']:.3f}"
    })

df = pd.DataFrame(metrics_data)
print("\nModel Comparison:")
print(df.to_string(index=False))


# ## 8. Visualizations
# 
# Display generated plots:

# In[ ]:


from IPython.display import Image, display
import os

plot_files = [
    'privacy_utility_tradeoff.png',
    'attack_resistance.png',
    'overall_comparison.png'
]

for plot_file in plot_files:
    path = f'./results/{plot_file}'
    if os.path.exists(path):
        print(f"\n{plot_file.replace('_', ' ').title()}:")
        display(Image(filename=path))
    else:
        print(f"⚠ {plot_file} not found")


# ## 9. Run Attack Simulation on Specific Model

# In[ ]:


from attack import AttackSimulator

# Load model to attack
model_to_attack = ChatbotModel(config)
model_to_attack.load_model('./models/baseline')  # Change to test different models

# Run attacks
simulator = AttackSimulator(model_to_attack, config)
attack_results = simulator.run_all_attacks(num_queries_per_attack=30)

print("\n" + "="*60)
print("Attack Results Summary")
print("="*60)
for attack_type, results in attack_results.items():
    if attack_type != 'aggregate':
        print(f"{attack_type.capitalize()} Attack: {results['success_rate']:.2%}")


# ## 10. Download Results
# 
# Download all results and models:

# In[ ]:


# Create zip file of results
get_ipython().system('zip -r results.zip ./results ./data ./models')

print("✓ Results zipped!")
print("Download 'results.zip' from the files panel")


# ## 11. Custom Experiments
# 
# Modify hyperparameters and re-run experiments:

# In[ ]:


# Modify config for custom experiments
from config import Config

custom_config = Config()

# Adjust parameters
custom_config.NUM_EPOCHS_SFT = 5  # More epochs
custom_config.LEARNING_RATE = 3e-5  # Different learning rate
custom_config.PII_INJECTION_RATE = 0.9  # More PII in data

print("Custom configuration:")
print(f"  Epochs: {custom_config.NUM_EPOCHS_SFT}")
print(f"  Learning Rate: {custom_config.LEARNING_RATE}")
print(f"  PII Injection Rate: {custom_config.PII_INJECTION_RATE}")

# Use this config for training...
# custom_model = ChatbotModel(custom_config)
# ...


# ## Summary
# 
# This notebook provides a complete implementation of the PII leakage analysis project:
# 
# ### Key Features:
# - ✓ Synthetic data generation with realistic PII injection
# - ✓ Four defense strategies: Baseline, SFT+Scrubbing, DPO, GRPO
# - ✓ Comprehensive privacy evaluation
# - ✓ Multiple attack simulations
# - ✓ Privacy-utility tradeoff analysis
# - ✓ Detailed visualizations and reports
# 
# 
