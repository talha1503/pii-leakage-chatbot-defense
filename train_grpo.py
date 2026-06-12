"""
GRPO Training Script using TRL's GRPOTrainer
Group Relative Policy Optimization for Privacy-Preserving Chatbots

Based on: https://huggingface.co/docs/trl/main/en/grpo_trainer
"""

import os
import json
import torch
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from config import Config
from pii_detector import PIIDetector


def load_conversation_data(data_path: str) -> List[Dict]:
    """Load conversation data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def extract_prompts_from_conversations(data: List[Dict]) -> List[str]:
    """
    Extract prompts from conversation data.
    
    Data format:
    {
        "conversation": [...],
        "profile": {...},
        "pii_leaked": [...],
        "template_type": "...",
        "contains_pii": true/false,
        "id": "train_xxx"
    }
    
    We extract the user messages to create prompts for GRPO.
    """
    prompts = []
    
    for item in data:
        # Skip items without conversation
        if "conversation" not in item or not item["conversation"]:
            continue
        
        conversation = item["conversation"]
        
        # Build prompt from conversation
        prompt_parts = []
        for turn in conversation:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant" and prompt_parts:
                # Stop before the first assistant response
                # (model will learn to generate this)
                break
        
        if prompt_parts:
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            prompts.append(prompt)
    
    return prompts


def create_grpo_dataset(data_path: str) -> Dataset:
    """
    Create HuggingFace Dataset for GRPO training.
    GRPO expects a dataset with 'prompt' column.
    """
    print(f"Loading data from: {data_path}")
    data = load_conversation_data(data_path)
    print(f"Loaded {len(data)} items")
    
    prompts = extract_prompts_from_conversations(data)
    print(f"Extracted {len(prompts)} prompts with conversations")
    
    if len(prompts) == 0:
        raise ValueError("No valid prompts extracted from data. Check your data format.")
    
    return Dataset.from_dict({"prompt": prompts})


class PrivacyRewardFunction:
    """
    Reward function for GRPO that penalizes PII leakage.
    
    Returns higher rewards for responses that:
    1. Don't contain PII (privacy)
    2. Are helpful and coherent (utility)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = PIIDetector()
        
        # Reward/penalty values from config
        self.no_leak_reward = getattr(config, 'NO_LEAK_REWARD', 1.0)
        self.pii_leak_penalty = getattr(config, 'PII_LEAK_PENALTY', -2.0)
        self.utility_weight = getattr(config, 'UTILITY_WEIGHT', 0.3)
    
    def __call__(
        self, 
        prompts: List[str], 
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Calculate privacy-aware rewards for completions.
        
        Args:
            prompts: List of input prompts
            completions: List of model-generated completions
            
        Returns:
            List of scalar reward values
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            reward = self._compute_single_reward(prompt, completion)
            rewards.append(reward)
        
        return rewards
    
    def _compute_single_reward(self, prompt: str, completion: str) -> float:
        """Compute reward for a single prompt-completion pair."""
        
        # Handle empty completions
        if not completion or not completion.strip():
            return -1.0
        
        # 1. Privacy Reward: Penalize PII leakage
        try:
            detected_pii = self.detector.detect_pii(completion)
            
            if len(detected_pii) == 0:
                privacy_reward = self.no_leak_reward
            else:
                # More PII = bigger penalty
                privacy_reward = self.pii_leak_penalty * len(detected_pii)
        except Exception as e:
            print(f"Warning: PII detection failed: {e}")
            privacy_reward = 0.0
        
        # 2. Utility Reward: Encourage meaningful responses
        utility_reward = self._compute_utility_reward(completion)
        
        # 3. Combine rewards
        total_reward = privacy_reward + self.utility_weight * utility_reward
        
        return float(total_reward)
    
    def _compute_utility_reward(self, completion: str) -> float:
        """
        Compute utility reward based on response quality.
        """
        word_count = len(completion.split())
        
        if word_count == 0:
            return -1.0  # Penalize empty responses
        elif word_count < 5:
            return 0.0   # Very short responses
        elif word_count <= 50:
            return min(word_count / 20, 1.0)  # Scale up to 1.0
        else:
            # Slightly penalize very long responses
            return max(0.5, 1.0 - (word_count - 50) / 200)


def setup_model_and_tokenizer(
    config: Config, 
    model_path: Optional[str] = None
) -> tuple:
    """
    Initialize or load model and tokenizer using AutoModel.
    
    Args:
        config: Configuration object
        model_path: Optional path to load pretrained model from
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name_or_path = model_path if (model_path and os.path.exists(model_path)) else config.MODEL_NAME
    
    print(f"Loading model from: {model_name_or_path}")
    
    # Load model with AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load tokenizer with AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side for generation
    tokenizer.padding_side = "left"
    
    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Pad token: {tokenizer.pad_token}")
    
    return model, tokenizer


def train_grpo(
    config: Config,
    train_data_path: str,
    output_dir: str,
    base_model_path: Optional[str] = None,
    resume_from_checkpoint: bool = False
):
    """
    Train model using GRPO (Group Relative Policy Optimization).
    
    GRPO works by:
    1. Sampling multiple completions per prompt (num_generations)
    2. Computing rewards for each completion using reward function
    3. Computing group-relative advantages (comparing within the group)
    4. Updating policy using advantage-weighted policy gradient
    
    Args:
        config: Configuration object
        train_data_path: Path to training data JSON
        output_dir: Directory to save trained model
        base_model_path: Optional path to load base model from
        resume_from_checkpoint: Whether to resume from existing checkpoint
    """
    print("\n" + "=" * 70)
    print("GRPO TRAINING (TRL GRPOTrainer)")
    print("=" * 70)
    
    # Check if model already exists
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        if not resume_from_checkpoint:
            print(f"Model already exists at {output_dir}. Skipping training.")
            print("Set resume_from_checkpoint=True to continue training.")
            return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, base_model_path)
    
    # Get device info
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Load and prepare dataset
    print(f"\nLoading training data from {train_data_path}")
    train_dataset = create_grpo_dataset(train_data_path)
    print(f"Created GRPO dataset with {len(train_dataset)} prompts")
    
    # Show sample prompts
    print("\nSample prompts:")
    for i in range(min(3, len(train_dataset))):
        prompt = train_dataset[i]["prompt"]
        display = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"  {i+1}. {display}")
    
    # Initialize reward function
    reward_fn = PrivacyRewardFunction(config)
    print("\n✓ Privacy reward function initialized")
    
    # Get GRPO parameters from config
    num_generations = getattr(config, 'GRPO_NUM_SAMPLES', 4)
    max_completion_length = getattr(config, 'GRPO_MAX_COMPLETION_LENGTH', 128)
    temperature = getattr(config, 'GRPO_TEMPERATURE', 0.7)
    batch_size = getattr(config, 'BATCH_SIZE', 4)
    gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 4)
    num_epochs = getattr(config, 'NUM_EPOCHS_GRPO', 3)
    learning_rate = getattr(config, 'LEARNING_RATE', 5e-5)
    beta = getattr(config, 'DPO_BETA', 0.1)
    warmup_steps = getattr(config, 'WARMUP_STEPS', 100)
    seed = getattr(config, 'SEED', 42)
    
    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        
        # === Sampling Configuration ===
        num_generations=num_generations,  # Completions per prompt
        max_completion_length=max_completion_length,
        temperature=temperature,
        
        # === Training Configuration ===
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        
        # === GRPO Specific ===
        beta=beta,  # KL penalty coefficient
        
        # === Optimization ===
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        weight_decay=0.01,
        
        # === Logging and Saving ===
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=2,
        
        # === Device Settings ===
        fp16=torch.cuda.is_available(),
        bf16=False,
        
        # === Misc ===
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    print("\nGRPO Configuration:")
    print(f"  - Number of generations per prompt: {grpo_config.num_generations}")
    print(f"  - Max completion length: {grpo_config.max_completion_length}")
    print(f"  - Temperature: {grpo_config.temperature}")
    print(f"  - Batch size: {grpo_config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {grpo_config.gradient_accumulation_steps}")
    print(f"  - Learning rate: {grpo_config.learning_rate}")
    print(f"  - Beta (KL penalty): {grpo_config.beta}")
    print(f"  - Epochs: {grpo_config.num_train_epochs}")
    
    # Initialize GRPO trainer
    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    print("\n" + "-" * 70)
    print("Starting GRPO training...")
    print("-" * 70 + "\n")
    
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'=' * 70}")
    print(f"✓ GRPO training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"{'=' * 70}")


def evaluate_grpo_model(
    config: Config,
    model_path: str,
    test_data_path: str,
    num_samples: int = 50
) -> Dict:
    """
    Evaluate trained GRPO model on privacy and utility metrics.
    
    Args:
        config: Configuration object
        model_path: Path to trained model
        test_data_path: Path to test data
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATING GRPO MODEL")
    print("=" * 70)
    
    # Load model using AutoModel
    model, tokenizer = setup_model_and_tokenizer(config, model_path)
    model.eval()
    
    device = next(model.parameters()).device
    
    # Load test data
    data = load_conversation_data(test_data_path)
    prompts = extract_prompts_from_conversations(data)
    
    # Limit samples
    prompts = prompts[:num_samples]
    print(f"Evaluating on {len(prompts)} prompts")
    
    # Initialize detector
    detector = PIIDetector()
    
    # Metrics
    results = {
        "total_samples": len(prompts),
        "pii_leakage_count": 0,
        "pii_leakage_rate": 0.0,
        "avg_response_length": 0.0,
        "empty_responses": 0,
        "leaked_pii_types": {},
    }
    
    total_length = 0
    
    print("\nGenerating responses...")
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(prompts)}")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response (only new tokens)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Check for empty response
        if not response:
            results["empty_responses"] += 1
            continue
        
        total_length += len(response.split())
        
        # Detect PII
        try:
            detected_pii = detector.detect_pii(response)
            
            if detected_pii:
                results["pii_leakage_count"] += 1
                for pii in detected_pii:
                    pii_type = pii.get("type", "unknown")
                    results["leaked_pii_types"][pii_type] = results["leaked_pii_types"].get(pii_type, 0) + 1
        except Exception as e:
            print(f"Warning: PII detection failed for sample {i}: {e}")
    
    # Calculate final metrics
    valid_samples = len(prompts) - results["empty_responses"]
    
    if valid_samples > 0:
        results["pii_leakage_rate"] = results["pii_leakage_count"] / valid_samples
        results["avg_response_length"] = total_length / valid_samples
    
    # Print results
    print("\n" + "-" * 50)
    print("Evaluation Results:")
    print("-" * 50)
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Valid responses: {valid_samples}")
    print(f"  Empty responses: {results['empty_responses']}")
    print(f"  PII leakage count: {results['pii_leakage_count']}")
    print(f"  PII leakage rate: {results['pii_leakage_rate']:.2%}")
    print(f"  Avg response length: {results['avg_response_length']:.1f} words")
    
    if results["leaked_pii_types"]:
        print("\n  Leaked PII types:")
        for pii_type, count in sorted(results["leaked_pii_types"].items(), key=lambda x: -x[1]):
            print(f"    - {pii_type}: {count}")
    
    return results


def main():
    """Main function to run GRPO training."""
    print("\n" + "=" * 70)
    print("GRPO TRAINING SCRIPT")
    print("Privacy-Preserving Chatbot Training")
    print("=" * 70)
    
    # Initialize config
    config = Config()
    
    # Set random seed
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    # Print device info
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ CUDA not available, using CPU")
    
    # Paths
    train_data_path = f"{config.DATA_DIR}/train.json"
    output_dir = f"{config.MODEL_DIR}/grpo"
    baseline_dir = f"{config.MODEL_DIR}/baseline"
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"\n Error: Training data not found at {train_data_path}")
        print("Please run data generation first.")
        return 1
    
    # Determine base model
    base_model_path = None
    if os.path.exists(baseline_dir) and os.path.exists(os.path.join(baseline_dir, "config.json")):
        print(f"\n✓ Found baseline model at {baseline_dir}")
        base_model_path = baseline_dir
    else:
        print(f"\n⚠ Baseline model not found. Starting from pretrained {config.MODEL_NAME}")
    
    # Train GRPO
    try:
        train_grpo(
            config=config,
            train_data_path=train_data_path,
            output_dir=output_dir,
            base_model_path=base_model_path,
            resume_from_checkpoint=False
        )
        
        # Evaluate if test data exists
        test_data_path = f"{config.DATA_DIR}/test.json"
        if os.path.exists(test_data_path):
            print("\n\nRunning evaluation on test set...")
            results = evaluate_grpo_model(
                config=config,
                model_path=output_dir,
                test_data_path=test_data_path,
                num_samples=50
            )
            
            # Save evaluation results
            results_path = f"{config.RESULTS_DIR}/grpo_evaluation.json"
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Evaluation results saved to {results_path}")
        
        print("\n" + "=" * 70)
        print("✓ GRPO TRAINING COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n Error during GRPO training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())