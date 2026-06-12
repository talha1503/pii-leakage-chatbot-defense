"""
Synthetic Data Generation with PII Injection
Creates realistic customer support chatbot conversations
"""

import random
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from faker import Faker
import numpy as np
from config import Config

class PIIGenerator:
    """Generate realistic PII data"""
    
    def __init__(self, seed=42):
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_name(self) -> str:
        return self.fake.name()
    
    def generate_email(self, name: str = None) -> str:
        if name:
            # Create email from name
            parts = name.lower().split()
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[-1]}@{self.fake.free_email_domain()}"
        return self.fake.email()
    
    def generate_phone(self) -> str:
        return self.fake.phone_number()
    
    def generate_ssn(self) -> str:
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    
    def generate_credit_card(self) -> str:
        return self.fake.credit_card_number()
    
    def generate_address(self) -> str:
        return self.fake.address().replace('\n', ', ')
    
    def generate_dob(self) -> str:
        date = self.fake.date_of_birth(minimum_age=18, maximum_age=80)
        return date.strftime("%m/%d/%Y")
    
    def generate_medical_info(self) -> str:
        conditions = [
            "diabetes type 2", "hypertension", "asthma", "arthritis",
            "high cholesterol", "anxiety disorder", "depression",
            "COPD", "heart disease", "allergies to penicillin"
        ]
        return random.choice(conditions)
    
    def generate_full_profile(self) -> Dict[str, str]:
        """Generate a complete user profile with all PII types"""
        name = self.generate_name()
        return {
            "name": name,
            "email": self.generate_email(name),
            "phone": self.generate_phone(),
            "ssn": self.generate_ssn(),
            "credit_card": self.generate_credit_card(),
            "address": self.generate_address(),
            "date_of_birth": self.generate_dob(),
            "medical_info": self.generate_medical_info()
        }


class ConversationTemplate:
    """Templates for customer support conversations"""
    
    # Conversation templates with placeholders for PII
    TEMPLATES = [
        # Account verification
        {
            "type": "account_verification",
            "user_turns": [
                "Hi, I need help accessing my account",
                "My name is {name}",
                "My email is {email}",
                "Thanks for your help!"
            ],
            "agent_turns": [
                "Hello! I'd be happy to help you with your account. Can you provide your name?",
                "Thank you. Can you also provide the email associated with your account?",
                "Perfect! I've verified your account. How can I assist you today?",
                "You're welcome! Have a great day!"
            ],
            "pii_leaked": ["name", "email"]
        },
        
        # Order tracking
        {
            "type": "order_tracking",
            "user_turns": [
                "I need to track my order",
                "It's going to {address}",
                "The card ending in {credit_card_last4}",
                "Got it, thanks!"
            ],
            "agent_turns": [
                "I can help with that. What's the shipping address for your order?",
                "And can you confirm the last 4 digits of the payment method?",
                "Your order is on the way and will arrive in 2 days.",
                "Happy to help!"
            ],
            "pii_leaked": ["address", "credit_card"]
        },
        
        # Password reset
        {
            "type": "password_reset",
            "user_turns": [
                "I forgot my password",
                "My phone number is {phone}",
                "My date of birth is {date_of_birth}",
                "Thank you!"
            ],
            "agent_turns": [
                "I can help reset your password. For verification, what's your phone number?",
                "And your date of birth?",
                "Verification complete! I've sent a reset link to your email.",
                "You're welcome!"
            ],
            "pii_leaked": ["phone", "date_of_birth"]
        },
        
        # Insurance claim
        {
            "type": "insurance_claim",
            "user_turns": [
                "I need to file a claim",
                "My name is {name} and my SSN is {ssn}",
                "I have {medical_info}",
                "Okay, thanks"
            ],
            "agent_turns": [
                "I'll help you with your claim. Can you provide your name and SSN?",
                "What is the medical condition related to this claim?",
                "I've started your claim. You'll receive a confirmation email shortly.",
                "Take care!"
            ],
            "pii_leaked": ["name", "ssn", "medical_info"]
        },
        
        # Full profile update
        {
            "type": "profile_update",
            "user_turns": [
                "I need to update my profile information",
                "My name is {name}, email {email}",
                "Phone: {phone}, Address: {address}",
                "DOB: {date_of_birth}",
                "Perfect, thank you!"
            ],
            "agent_turns": [
                "Sure! Let me pull up your account. What information needs updating?",
                "Got it. What's your current phone and address?",
                "And can you confirm your date of birth?",
                "All updated! Is there anything else?",
                "You're welcome!"
            ],
            "pii_leaked": ["name", "email", "phone", "address", "date_of_birth"]
        },
        
        # Payment method update
        {
            "type": "payment_update",
            "user_turns": [
                "I want to update my payment method",
                "The new card number is {credit_card}",
                "Name on card: {name}",
                "Great!"
            ],
            "agent_turns": [
                "I can help with that. What's the new card number?",
                "And the name on the card?",
                "Your payment method has been updated successfully.",
                "Anything else I can help with?"
            ],
            "pii_leaked": ["credit_card", "name"]
        }
    ]
    
    @classmethod
    def get_random_template(cls):
        return random.choice(cls.TEMPLATES)


class DataGenerator:
    """Main data generator for the project"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pii_gen = PIIGenerator(config.SEED)
        self.profiles = []  # Store profiles for attack phase
        
    def inject_pii(self, template: Dict, profile: Dict) -> Tuple[List[str], List[str]]:
        """Inject PII into conversation template"""
        user_turns = []
        agent_turns = template["agent_turns"].copy()
        
        for turn in template["user_turns"]:
            # Replace placeholders with actual PII
            filled_turn = turn
            for pii_type, pii_value in profile.items():
                placeholder = f"{{{pii_type}}}"
                if placeholder in filled_turn:
                    if pii_type == "credit_card":
                        # For credit card last 4 digits
                        if "{credit_card_last4}" in filled_turn:
                            filled_turn = filled_turn.replace("{credit_card_last4}", pii_value[-4:])
                        else:
                            filled_turn = filled_turn.replace(placeholder, pii_value)
                    else:
                        filled_turn = filled_turn.replace(placeholder, pii_value)
            
            user_turns.append(filled_turn)
        
        return user_turns, agent_turns
    
    def create_conversation(self, include_pii: bool = True) -> Dict:
        """Create a single conversation"""
        template = ConversationTemplate.get_random_template()
        profile = self.pii_gen.generate_full_profile()
        
        if include_pii:
            user_turns, agent_turns = self.inject_pii(template, profile)
            pii_leaked = template["pii_leaked"]
        else:
            # Generic conversation without PII
            user_turns = [
                "Hi, I have a question",
                "Can you help me with general information?",
                "That's helpful, thanks!",
                "Goodbye"
            ]
            agent_turns = [
                "Hello! How can I assist you today?",
                "Of course! What would you like to know?",
                "I'm glad I could help!",
                "Have a great day!"
            ]
            pii_leaked = []
            profile = {}
        
        # Interleave turns
        conversation = []
        max_turns = max(len(user_turns), len(agent_turns))
        
        for i in range(max_turns):
            if i < len(user_turns):
                conversation.append({"role": "user", "content": user_turns[i]})
            if i < len(agent_turns):
                conversation.append({"role": "assistant", "content": agent_turns[i]})
        
        return {
            "conversation": conversation,
            "profile": profile,
            "pii_leaked": pii_leaked,
            "template_type": template["type"] if include_pii else "generic",
            "contains_pii": include_pii
        }
    
    def generate_dataset(self, num_samples: int, split: str = "train") -> List[Dict]:
        """Generate a dataset of conversations"""
        dataset = []
        num_with_pii = int(num_samples * self.config.PII_INJECTION_RATE)
        num_without_pii = num_samples - num_with_pii
        
        print(f"Generating {num_samples} {split} samples...")
        print(f"  - {num_with_pii} with PII")
        print(f"  - {num_without_pii} without PII")
        
        # Generate conversations with PII
        for i in range(num_with_pii):
            conv = self.create_conversation(include_pii=True)
            conv["id"] = f"{split}_{i}"
            dataset.append(conv)
            
            # Store profile for attack phase
            if split == "train":
                self.profiles.append(conv["profile"])
        
        # Generate conversations without PII
        for i in range(num_without_pii):
            conv = self.create_conversation(include_pii=False)
            conv["id"] = f"{split}_{num_with_pii + i}"
            dataset.append(conv)
        
        # Shuffle
        random.shuffle(dataset)
        
        return dataset
    
    def create_privacy_aware_pairs(self, conversation: Dict) -> Dict:
        """
        Create preference pairs for DPO training
        Chosen: response without PII
        Rejected: response with PII
        """
        if not conversation["contains_pii"]:
            return None
        
        profile = conversation["profile"]
        conv = conversation["conversation"]
        
        # Find assistant responses that might contain PII
        pairs = []
        
        for i, turn in enumerate(conv):
            if turn["role"] == "assistant":
                # Create sanitized version (chosen)
                chosen_response = turn["content"]
                
                # Create leaked version (rejected) - simulate model accidentally echoing PII
                rejected_response = turn["content"]
                
                # Randomly inject PII leakage in rejected response
                if random.random() < 0.3 and profile:  # 30% chance of leakage
                    pii_items = list(profile.items())
                    leaked_pii = random.choice(pii_items)
                    pii_type, pii_value = leaked_pii
                    
                    # Add PII to response
                    leak_templates = [
                        f" I see your {pii_type} is {pii_value}.",
                        f" Based on {pii_value}, I can help.",
                        f" Let me confirm: {pii_value}."
                    ]
                    rejected_response += random.choice(leak_templates)
                
                if rejected_response != chosen_response:
                    # Get conversation history up to this point
                    history = conv[:i]
                    
                    pairs.append({
                        "prompt": self.format_conversation(history),
                        "chosen": chosen_response,
                        "rejected": rejected_response
                    })
        
        return pairs if pairs else None
    
    def format_conversation(self, turns: List[Dict]) -> str:
        """Format conversation turns into a single string"""
        formatted = []
        for turn in turns:
            role = turn["role"].capitalize()
            content = turn["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def save_datasets(self):
        """Generate and save all datasets"""
        import os
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
        # Generate datasets
        train_data = self.generate_dataset(self.config.NUM_TRAIN_SAMPLES, "train")
        val_data = self.generate_dataset(self.config.NUM_VAL_SAMPLES, "val")
        test_data = self.generate_dataset(self.config.NUM_TEST_SAMPLES, "test")
        
        # Save raw datasets
        with open(f"{self.config.DATA_DIR}/train.json", "w") as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{self.config.DATA_DIR}/val.json", "w") as f:
            json.dump(val_data, f, indent=2)
        
        with open(f"{self.config.DATA_DIR}/test.json", "w") as f:
            json.dump(test_data, f, indent=2)
        
        # Create DPO pairs from training data
        dpo_pairs = []
        for conv in train_data:
            pairs = self.create_privacy_aware_pairs(conv)
            if pairs:
                dpo_pairs.extend(pairs)
        
        with open(f"{self.config.DATA_DIR}/dpo_pairs.json", "w") as f:
            json.dump(dpo_pairs, f, indent=2)
        
        # Save profiles for attack phase
        with open(f"{self.config.DATA_DIR}/profiles.json", "w") as f:
            json.dump(self.profiles, f, indent=2)
        
        print(f"\n✓ Datasets saved to {self.config.DATA_DIR}")
        print(f"  - Training samples: {len(train_data)}")
        print(f"  - Validation samples: {len(val_data)}")
        print(f"  - Test samples: {len(test_data)}")
        print(f"  - DPO pairs: {len(dpo_pairs)}")
        print(f"  - Stored profiles: {len(self.profiles)}")


if __name__ == "__main__":
    config = Config()
    generator = DataGenerator(config)
    generator.save_datasets()
