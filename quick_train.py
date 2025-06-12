#!/usr/bin/env python3
"""
Quick Training Script for Clickbait Spoiling
Optimized for demonstration while maintaining paper accuracy.
"""

import json
import torch
import logging
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClickbaitDataset(Dataset):
    """Optimized dataset for clickbait spoiling"""
    
    def __init__(self, data_path, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract components (exactly as paper)
        post_text = item['postText'][0] if isinstance(item['postText'], list) else item['postText']
        
        # Get shortest paragraph (as suggested in analysis)
        target_paragraphs = item.get('targetParagraphs', [])
        if target_paragraphs:
            target_paragraph = min(target_paragraphs, key=len)
        else:
            target_paragraph = ""
        
        spoiler = item['spoiler'][0] if isinstance(item.get('spoiler'), list) else item.get('spoiler', '')
        
        # Create input format: postText + targetParagraph + <SEP> + spoiler
        input_text = f"{post_text.strip()} {target_paragraph.strip()}"
        full_text = f"{input_text} <SEP> {spoiler.strip()}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM
        }

class QuickTrainer:
    """Quick trainer with paper hyperparameters"""
    
    def __init__(self):
        self.model_name = 'gpt2-medium'  # 355M parameters as paper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        logger.info(f"Loading {self.model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add special tokens
        special_tokens = {'pad_token': '<PAD>', 'sep_token': '<SEP>'}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, train_data_path, val_data_path, epochs=10, quick_demo=False):
        """Train with paper hyperparameters"""
        
        # For quick demo, use subset of data
        max_train = 500 if quick_demo else None
        max_val = 100 if quick_demo else None
        
        # Create datasets
        train_dataset = ClickbaitDataset(train_data_path, self.tokenizer, max_samples=max_train)
        val_dataset = ClickbaitDataset(val_data_path, self.tokenizer, max_samples=max_val)
        
        # Training arguments (paper hyperparameters)
        output_dir = f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,  # Paper: 10 epochs
            per_device_train_batch_size=4,  # Conservative for memory
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,  # Effective batch = 8
            learning_rate=5e-5,  # Conservative as suggested
            lr_scheduler_type='cosine',  # Cosine decay
            weight_decay=0.01,
            warmup_steps=100,  # Shorter for demo
            
            # Evaluation and saving
            eval_strategy='steps',
            eval_steps=250 if quick_demo else 500,
            save_steps=250 if quick_demo else 500,
            logging_steps=50 if quick_demo else 100,
            
            # Best model selection
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            
            # Memory optimization
            fp16=True,  # Mixed precision
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            
            # Reporting
            report_to='tensorboard',
            run_name=f'clickbait_spoiling_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        )
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked
        )
        
        # Early stopping (if validation loss stops improving)
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping]
        )
        
        # Train
        logger.info("🚀 Starting fine-tuning...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Training for {epochs} epochs")
        
        try:
            trainer.train()
            
            # Save final model
            final_model_path = os.path.join(output_dir, 'final_model')
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            logger.info(f"✅ Training completed! Model saved to {final_model_path}")
            return final_model_path, output_dir
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None, None
    
    def generate_sample(self, post_text, target_paragraph=""):
        """Generate a sample spoiler"""
        input_text = f"{post_text} {target_paragraph} <SEP>"
        
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=30,
                temperature=0.0,  # Greedy (paper-like)
                num_beams=1,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        spoiler = generated.split('<SEP>')[-1].strip()
        return spoiler

def main():
    """Main training function"""
    
    print("🎯 Clickbait Spoiling - Fine-tuning GPT-2 Medium")
    print("=" * 60)
    
    # Initialize trainer
    trainer = QuickTrainer()
    
    # Show sample before training
    print("\n📝 Sample generation BEFORE training:")
    sample_spoiler = trainer.generate_sample(
        "You Won't Believe What This Celebrity Did Next",
        "The celebrity announced retirement to pursue marine biology."
    )
    print(f"Generated spoiler: {sample_spoiler}")
    
    # Training settings
    print("\n🔧 Training Configuration:")
    print("- Model: GPT-2 medium (355M parameters)")
    print("- Learning rate: 5e-5")
    print("- Batch size: 4 (with gradient accumulation = 8)")
    print("- Scheduler: Cosine decay")
    print("- Epochs: 10 (as per paper)")
    print("- Early stopping: Yes")
    
    # Ask for quick demo or full training
    choice = input("\n❓ Quick demo (500 train samples) or full training? [demo/full]: ").strip().lower()
    quick_demo = choice.startswith('demo') or choice == 'd'
    
    if quick_demo:
        print("🏃 Running quick demo with 500 training samples...")
        epochs = 3
    else:
        print("🐌 Running full training with all 3,200 samples...")
        epochs = 10
    
    # Train
    model_path, output_dir = trainer.train(
        'data/train.jsonl',
        'data/validation.jsonl', 
        epochs=epochs,
        quick_demo=quick_demo
    )
    
    if model_path:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Logs saved to: {output_dir}")
        
        # Show sample after training
        print("\n📝 Sample generation AFTER training:")
        sample_spoiler = trainer.generate_sample(
            "You Won't Believe What This Celebrity Did Next",
            "The celebrity announced retirement to pursue marine biology."
        )
        print(f"Generated spoiler: {sample_spoiler}")
        
        print(f"\n🔄 Next steps:")
        print(f"1. Generate predictions: python quick_spoiler.py --batch data/validation.jsonl --model {model_path}")
        print(f"2. Evaluate results: python evaluate_spoilers.py --predictions results.json --compare")
        
        return model_path
    else:
        print("❌ Training failed!")
        return None

if __name__ == "__main__":
    main() 