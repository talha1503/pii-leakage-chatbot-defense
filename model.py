"""
Model Architecture and Training Utilities - FIXED VERSION
Includes custom collate function for variable-length sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Dict, Optional
import json
import os
from tqdm import tqdm
import numpy as np
from config import Config


class ConversationDataset(Dataset):
    """Dataset for conversation data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation into training text"""
        formatted = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                formatted.append(f"User: {content}")
            else:
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted) + "\n"
    
    def __getitem__(self, idx):
        item = self.data[idx]
        conversation_text = self.format_conversation(item["conversation"])
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.where(encoding["attention_mask"].bool(), encoding["input_ids"], -100).squeeze(), # Mask out padding, -100 is a special token for padding
            "contains_pii": item["contains_pii"],
            "pii_leaked": item.get("pii_leaked", [])
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Extract individual components
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    contains_pii = [item["contains_pii"] for item in batch]
    pii_leaked = [item["pii_leaked"] for item in batch]
    
    # Stack tensors (they should already be padded to max_length)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
        "contains_pii": contains_pii,
        "pii_leaked": pii_leaked
    }


class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize prompt - also pad to ensure consistent sizes
        prompt = item["prompt"] + "\nAssistant: "
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            padding="max_length",  # Changed to pad prompts too
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize chosen response
        chosen_text = prompt + item["chosen"]
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_text = prompt + item["rejected"]
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_encoding["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoding["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
        }


def dpo_collate_fn(batch):
    """Custom collate function for DPO batches - all sequences now padded"""
    return {
        "prompt_input_ids": torch.stack([item["prompt_input_ids"] for item in batch]),
        "prompt_attention_mask": torch.stack([item["prompt_attention_mask"] for item in batch]),
        "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
        "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
        "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
        "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch]),
    }


class ChatbotModel:
    """Wrapper for GPT-2 based chatbot with multiple training strategies"""
    
    def __init__(self, config: Config, model_name: str = None):
        self.config = config
        self.device = config.DEVICE
        
        # Load model and tokenizer
        model_name = model_name or config.MODEL_NAME
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
        
    def train_sft(self, train_data_path: str, val_data_path: str, 
                  output_dir: str, use_scrubbing: bool = False):
        """
        Supervised Fine-Tuning
        Args:
            use_scrubbing: If True, filters out PII during training
        """
        print(f"\n{'='*50}")
        print(f"Training SFT Model {'(with PII scrubbing)' if use_scrubbing else '(baseline)'}")
        print(f"{'='*50}\n")
        
        # Check if model already exists
        # if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        #     print(f"Model already exists at {output_dir}. Loading and skipping training.")
        #     self.load_model(output_dir)
        #     return
        
        # Load datasets
        train_dataset = ConversationDataset(train_data_path, self.tokenizer, self.config.MAX_LENGTH)
        val_dataset = ConversationDataset(val_data_path, self.tokenizer, self.config.MAX_LENGTH)
        
        # If scrubbing, filter out samples with PII
        if use_scrubbing:
            print("Applying PII scrubbing filter...")
            train_dataset.data = [d for d in train_dataset.data if not d["contains_pii"]]
            print(f"Filtered to {len(train_dataset.data)} samples without PII")
        
        # Use custom collate function
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.EVAL_BATCH_SIZE, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        total_steps = len(train_loader) * self.config.NUM_EPOCHS_SFT
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS_SFT):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS_SFT}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
                progress_bar.set_postfix({"loss": loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            val_loss = self.evaluate(val_loader)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(output_dir)
                print(f"✓ Model saved to {output_dir}")
        
        print(f"\n✓ SFT training complete!")
    
    def train_dpo(self, dpo_data_path: str, output_dir: str, reference_model=None):
        """
        Direct Preference Optimization
        Trains model to prefer privacy-preserving responses
        """
        print(f"\n{'='*50}")
        print(f"Training DPO Model")
        print(f"{'='*50}\n")
        
        # Check if model already exists
        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"Model already exists at {output_dir}. Loading and skipping training.")
            self.load_model(output_dir)
            return
        
        # Load DPO dataset
        dpo_dataset = DPODataset(dpo_data_path, self.tokenizer, self.config.MAX_LENGTH)
        dpo_loader = DataLoader(
            dpo_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=dpo_collate_fn
        )
        
        # Reference model (frozen copy of current model)
        if reference_model is None:
            reference_model = GPT2LMHeadModel.from_pretrained(self.config.MODEL_NAME)
            reference_model.to(self.device)
            reference_model.eval()
        
        optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        self.model.train()
        
        for epoch in range(self.config.NUM_EPOCHS_DPO):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS_DPO}")
            
            total_loss = 0
            progress_bar = tqdm(dpo_loader, desc="Training DPO")
            
            for batch in progress_bar:
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)
                prompt_ids = batch["prompt_input_ids"].to(self.device)
                
                # Get log probabilities from policy model
                chosen_logits = self.model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                rejected_logits = self.model(input_ids=rejected_ids, attention_mask=rejected_mask).logits
                
                # Get log probabilities from reference model
                with torch.no_grad():
                    ref_chosen_logits = reference_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                    ref_rejected_logits = reference_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits
                
                # Calculate DPO loss
                prompt_len = prompt_ids.shape[1]
                
                chosen_logprobs = self._get_log_probs(chosen_logits, chosen_ids, prompt_len)
                rejected_logprobs = self._get_log_probs(rejected_logits, rejected_ids, prompt_len)
                ref_chosen_logprobs = self._get_log_probs(ref_chosen_logits, chosen_ids, prompt_len)
                ref_rejected_logprobs = self._get_log_probs(ref_rejected_logits, rejected_ids, prompt_len)
                
                # DPO objective
                pi_logratios = chosen_logprobs - rejected_logprobs
                ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
                
                loss = -F.logsigmoid(self.config.DPO_BETA * (pi_logratios - ref_logratios)).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(dpo_loader)
            print(f"Average DPO Loss: {avg_loss:.4f}")
        
        self.save_model(output_dir)
        print(f"\n✓ DPO training complete! Model saved to {output_dir}")
    
    def train_grpo(self, train_data_path: str, output_dir: str):
        """
        Group Relative Policy Optimization
        Samples multiple responses and ranks them based on privacy scores
        """
        print(f"\n{'='*50}")
        print(f"Training GRPO Model")
        print(f"{'='*50}\n")
        
        # Check if model already exists
        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"Model already exists at {output_dir}. Loading and skipping training.")
            self.load_model(output_dir)
            return
        
        # Load dataset
        dataset = ConversationDataset(train_data_path, self.tokenizer, self.config.MAX_LENGTH)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True,
            collate_fn=collate_fn
        )
        
        optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        self.model.train()
        
        for epoch in range(self.config.NUM_EPOCHS_GRPO):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS_GRPO}")
            
            total_loss = 0
            progress_bar = tqdm(dataloader, desc="Training GRPO")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Sample multiple responses
                samples = []
                rewards = []
                
                for _ in range(self.config.GRPO_NUM_SAMPLES):
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=input_ids[:, :input_ids.shape[1]//2],
                            max_length=self.config.MAX_LENGTH,
                            temperature=self.config.GRPO_TEMPERATURE,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode and calculate privacy reward
                    decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    reward = self._calculate_privacy_reward(decoded, batch)
                    
                    samples.append(output)
                    rewards.append(reward)
                
                # Rank samples by reward
                ranked_indices = np.argsort(rewards)[::-1]
                
                # Calculate GRPO loss
                loss = 0
                for i, idx in enumerate(ranked_indices):
                    sample = samples[idx]
                    
                    # Calculate log probability
                    outputs = self.model(input_ids=sample, labels=sample)
                    log_prob = -outputs.loss
                    
                    # Weight by rank
                    weight = (self.config.GRPO_NUM_SAMPLES - i) / self.config.GRPO_NUM_SAMPLES
                    loss -= weight * log_prob
                
                loss = loss / self.config.GRPO_NUM_SAMPLES
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average GRPO Loss: {avg_loss:.4f}")
        
        self.save_model(output_dir)
        print(f"\n✓ GRPO training complete! Model saved to {output_dir}")
    
    def _get_log_probs(self, logits, labels, start_idx):
        """Calculate log probabilities for DPO"""
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs[:, start_idx:-1], 
            dim=2, 
            index=labels[:, start_idx+1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs.sum(dim=1)
    
    def _calculate_privacy_reward(self, text: str, batch: Dict) -> float:
        """Calculate privacy reward for GRPO"""
        from pii_detector import PIIDetector
        
        detector = PIIDetector()
        detected_pii = detector.detect_pii(text)
        
        # Base reward
        if len(detected_pii) == 0:
            privacy_reward = self.config.NO_LEAK_REWARD
        else:
            privacy_reward = self.config.PII_LEAK_PENALTY * len(detected_pii)
        
        # Utility reward
        utility_reward = min(len(text.split()) / 20, 1.0)
        
        # Combined reward
        total_reward = privacy_reward + self.config.UTILITY_WEIGHT * utility_reward
        
        return total_reward
    
    def evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(dataloader)
    
    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate response for a given prompt"""
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return response
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_model(self, model_dir: str):
        """Load model and tokenizer"""
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model.to(self.device)