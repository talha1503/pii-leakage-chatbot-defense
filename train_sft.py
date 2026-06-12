"""
SFT Training Script using TRL's SFTTrainer
Supervised Fine-Tuning for Privacy-Preserving Chatbots

Based on: https://huggingface.co/docs/trl/en/sft_trainer
"""

import os
import json
import torch
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from config import Config
from pii_detector import PIIDetector


def load_conversation_data(data_path: str) -> List[Dict]:
    """Load conversation data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def format_conversation_to_chat(conversation: List[Dict]) -> List[Dict]:
    """
    Format conversation to chat template format.
    
    Input format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Output format (same, but cleaned):
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    formatted = []
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
        if role in ["user", "assistant"] and content:
            formatted.append({"role": role, "content": content})
    
    return formatted


def format_messages_to_text(messages: List[Dict]) -> str:
    """
    Format messages to plain text format.
    Used as fallback for models without chat template.
    
    Important: Format must match the templates used in DataCollatorForCompletionOnlyLM
    """
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            text_parts.append(f"User: {content}")
        elif role == "assistant":
            text_parts.append(f"\nAssistant: {content}")
    
    return "\n".join(text_parts)


def scrub_pii_from_messages(
    messages: List[Dict], 
    detector: PIIDetector
) -> List[Dict]:
    """
    Scrub PII from messages using PIIDetector.
    """
    scrubbed = []
    for msg in messages:
        content = msg["content"]
        
        # Detect and replace PII
        detected_pii = detector.detect_pii(content)
        
        for pii in detected_pii:
            pii_value = pii.get("value", "")
            pii_type = pii.get("type", "PII")
            
            if pii_value:
                content = content.replace(pii_value, f"[{pii_type.upper()}]")
        
        scrubbed.append({
            "role": msg["role"],
            "content": content
        })
    
    return scrubbed


def create_sft_dataset(
    data_path: str, 
    tokenizer: AutoTokenizer,
    use_scrubbing: bool = False
) -> Dataset:
    """
    Create HuggingFace Dataset for SFT training.
    
    SFT expects either:
    1. A 'text' column with formatted conversations
    2. A 'messages' column with chat format
    
    We use the 'text' format for compatibility with completion-only training.
    """
    print(f"Loading data from: {data_path}")
    data = load_conversation_data(data_path)
    print(f"Loaded {len(data)} items")
    
    detector = None
    if use_scrubbing:
        print("PII scrubbing enabled - scrubbing training data...")
        detector = PIIDetector()
    
    # Prepare dataset entries
    all_messages = []
    all_texts = []
    
    for item in data:
        # Skip items without conversation
        if "conversation" not in item or not item["conversation"]:
            continue
        
        conversation = item["conversation"]
        
        # Format conversation
        messages = format_conversation_to_chat(conversation)
        
        if len(messages) < 2:  # Need at least user + assistant
            continue
        
        # Apply PII scrubbing if enabled
        if use_scrubbing and detector:
            messages = scrub_pii_from_messages(messages, detector)
        
        all_messages.append(messages)
        
        # Create text format for training
        text = format_messages_to_text(messages)
        all_texts.append(text)
    
    print(f"Created dataset with {len(all_messages)} conversations")
    
    if len(all_messages) == 0:
        raise ValueError("No valid conversations extracted from data. Check your data format.")
    
    # Create dataset with both formats
    return Dataset.from_dict({
        "messages": all_messages,
        "text": all_texts
    })


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
    
    # Set padding side for training (right padding for SFT)
    tokenizer.padding_side = "right"
    
    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Pad token: {tokenizer.pad_token}")
    print(f"  - Has chat template: {tokenizer.chat_template is not None}")
    
    return model, tokenizer


def train_sft(
    config: Config,
    train_data_path: str,
    output_dir: str,
    val_data_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    use_scrubbing: bool = False,
    use_completion_only: bool = True,
    resume_from_checkpoint: bool = False
):
    """
    Train model using SFT (Supervised Fine-Tuning).
    
    SFT works by:
    1. Formatting conversations into training text
    2. Training the model to predict the next token
    3. Optionally masking user turns (completion-only training)
    
    Args:
        config: Configuration object
        train_data_path: Path to training data JSON
        output_dir: Directory to save trained model
        val_data_path: Optional path to validation data JSON
        base_model_path: Optional path to load base model from
        use_scrubbing: Whether to scrub PII from training data
        use_completion_only: Whether to mask user turns (train only on assistant responses)
        resume_from_checkpoint: Whether to resume from existing checkpoint
    """
    scrub_str = " (with PII scrubbing)" if use_scrubbing else ""
    mask_str = " (completion-only)" if use_completion_only else ""
    print("\n" + "=" * 70)
    print(f"SFT TRAINING (TRL SFTTrainer){scrub_str}{mask_str}")
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
    
    # Load and prepare datasets
    print(f"\nLoading training data from {train_data_path}")
    train_dataset = create_sft_dataset(train_data_path, tokenizer, use_scrubbing)
    print(f"Created training dataset with {len(train_dataset)} conversations")
    
    eval_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        print(f"Loading validation data from {val_data_path}")
        eval_dataset = create_sft_dataset(val_data_path, tokenizer, use_scrubbing)
        print(f"Created validation dataset with {len(eval_dataset)} conversations")
    
    # Show sample conversations
    print("\nSample conversations:")
    for i in range(min(2, len(train_dataset))):
        text = train_dataset[i]["text"]
        display = text[:200] + "..." if len(text) > 200 else text
        print(f"  {i+1}. {display}")
    
    # Get training parameters from config
    batch_size = getattr(config, 'BATCH_SIZE', 4)
    gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 4)
    num_epochs = getattr(config, 'NUM_EPOCHS', 3)
    learning_rate = getattr(config, 'LEARNING_RATE', 5e-5)
    warmup_steps = getattr(config, 'WARMUP_STEPS', 100)
    max_length = getattr(config, 'MAX_LENGTH', 512)
    seed = getattr(config, 'SEED', 42)
    
    # === Create Data Collator for Completion-Only Training ===
    data_collator = None
    if use_completion_only:
        # These templates must match the format in format_messages_to_text()
        response_template = "\nAssistant:"
        instruction_template = "User:"
        
        # Tokenize templates to get token IDs
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
        
        print(f"\n✓ Completion-only training enabled")
        print(f"  - Instruction template: '{instruction_template}' -> {instruction_template_ids}")
        print(f"  - Response template: '{response_template}' -> {response_template_ids}")
        print(f"  - Only assistant responses will be used for loss computation")
        
        # Create the collator that masks instruction tokens
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            instruction_template=instruction_template_ids,
            tokenizer=tokenizer,
            mlm=False,
        )
    
    # Configure SFT training
    sft_config = SFTConfig(
        output_dir=output_dir,
        
        # === Dataset Configuration ===
        max_seq_length=max_length,
        
        # === Training Configuration ===
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        
        # === Optimization ===
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        
        # === Logging and Saving ===
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        
        # === Device Settings ===
        fp16=torch.cuda.is_available(),
        bf16=False,
        
        # === Misc ===
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=True,
        
        # === SFT Specific ===
        dataset_text_field="text",  # Use text column
        packing=False,  # Disable packing for cleaner training
    )
    
    print("\nSFT Configuration:")
    print(f"  - Max sequence length: {sft_config.max_seq_length}")
    print(f"  - Batch size: {sft_config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {sft_config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  - Learning rate: {sft_config.learning_rate}")
    print(f"  - Epochs: {sft_config.num_train_epochs}")
    print(f"  - PII scrubbing: {use_scrubbing}")
    print(f"  - Completion-only training: {use_completion_only}")
    
    # Initialize SFT trainer
    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,  # None if not using completion-only
    )
    
    # Train
    print("\n" + "-" * 70)
    print("Starting SFT training...")
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
    print(f"✓ SFT training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"{'=' * 70}")


def evaluate_sft_model(
    config: Config,
    model_path: str,
    test_data_path: str,
    num_samples: int = 50
) -> Dict:
    """
    Evaluate trained SFT model on privacy and utility metrics.
    
    Args:
        config: Configuration object
        model_path: Path to trained model
        test_data_path: Path to test data
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATING SFT MODEL")
    print("=" * 70)
    
    # Load model using AutoModel
    model, tokenizer = setup_model_and_tokenizer(config, model_path)
    model.eval()
    
    # Set padding side for generation
    tokenizer.padding_side = "left"
    
    device = next(model.parameters()).device
    
    # Load test data
    data = load_conversation_data(test_data_path)
    
    # Extract prompts (user messages only)
    prompts = []
    for item in data:
        if "conversation" not in item or not item["conversation"]:
            continue
        
        conversation = item["conversation"]
        prompt_parts = []
        
        for turn in conversation:
            if turn.get("role") == "user":
                prompt_parts.append(f"User: {turn['content']}")
            elif turn.get("role") == "assistant" and prompt_parts:
                break
        
        if prompt_parts:
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            prompts.append(prompt)
    
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
    """Main function to run SFT training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument("--scrub", action="store_true", help="Enable PII scrubbing")
    parser.add_argument("--no-mask", action="store_true", help="Disable completion-only masking (train on full conversation)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model for evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SFT TRAINING SCRIPT")
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
    val_data_path = f"{config.DATA_DIR}/val.json"
    test_data_path = f"{config.DATA_DIR}/test.json"
    
    # Determine output directory based on scrubbing
    if args.scrub:
        output_dir = f"{config.MODEL_DIR}/sft_scrubbed"
    else:
        output_dir = f"{config.MODEL_DIR}/baseline"
    
    # Evaluation only mode
    if args.eval_only:
        model_path = args.model_path or output_dir
        if not os.path.exists(model_path):
            print(f"\n Error: Model not found at {model_path}")
            return 1
        
        results = evaluate_sft_model(
            config=config,
            model_path=model_path,
            test_data_path=test_data_path,
            num_samples=50
        )
        return 0
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"\n❌ Error: Training data not found at {train_data_path}")
        print("Please run data generation first.")
        return 1
    
    # Determine whether to use completion-only masking
    use_completion_only = not args.no_mask
    
    # Train SFT
    try:
        train_sft(
            config=config,
            train_data_path=train_data_path,
            output_dir=output_dir,
            val_data_path=val_data_path if os.path.exists(val_data_path) else None,
            base_model_path=None,  # Start from pretrained
            use_scrubbing=args.scrub,
            use_completion_only=use_completion_only,
            resume_from_checkpoint=args.resume
        )
        
        # Evaluate if test data exists
        if os.path.exists(test_data_path):
            print("\n\nRunning evaluation on test set...")
            results = evaluate_sft_model(
                config=config,
                model_path=output_dir,
                test_data_path=test_data_path,
                num_samples=50
            )
            
            # Save evaluation results
            suffix = "_scrubbed" if args.scrub else "_baseline"
            results_path = f"{config.RESULTS_DIR}/sft{suffix}_evaluation.json"
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Evaluation results saved to {results_path}")
        
        print("\n" + "=" * 70)
        print("✓ SFT TRAINING COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n Error during SFT training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())