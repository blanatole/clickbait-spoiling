#!/usr/bin/env python3
"""
Clickbait Spoiling Implementation using GPT-2 Medium
Based on the hyperparameters analysis provided.
"""

import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    AdamW, get_cosine_schedule_with_warmup,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.metrics import accuracy_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import evaluate
import os
import re
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClickbaitDataset(Dataset):
    """Dataset class for clickbait spoiling"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load and preprocess data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract components
        post_text = item['postText'][0] if isinstance(item['postText'], list) else item['postText']
        target_paragraph = self._get_shortest_paragraph(item['targetParagraphs'])
        spoiler = item['spoiler'][0] if isinstance(item['spoiler'], list) else item['spoiler']
        
        # Create input text (post + target paragraph)
        input_text = f"{post_text} {target_paragraph}"
        
        # Create training text (input + spoiler)
        training_text = f"{input_text} <SEP> {spoiler}"
        
        # Tokenize
        tokens = self.tokenizer(
            training_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze().clone()
        }
    
    def _get_shortest_paragraph(self, paragraphs):
        """Get the shortest paragraph that likely contains the answer"""
        return min(paragraphs, key=len) if paragraphs else ""

class ClickbaitSpoiler:
    """Main class for clickbait spoiling"""
    
    def __init__(self, model_name='gpt2-medium'):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {'pad_token': '<PAD>', 'sep_token': '<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Initialized model: {model_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, train_data_path, val_data_path, output_dir='./results'):
        """Train the spoiling model with optimized hyperparameters"""
        
        # Create datasets
        train_dataset = ClickbaitDataset(train_data_path, self.tokenizer)
        val_dataset = ClickbaitDataset(val_data_path, self.tokenizer)
        
        # Training arguments based on paper analysis
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,  # As specified in paper
            per_device_train_batch_size=4,  # Small batch size as suggested
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,  # Effective batch size = 8
            learning_rate=5e-5,  # Conservative learning rate
            lr_scheduler_type='cosine',  # Cosine decay
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(os.path.join(output_dir, 'final_model'))
        self.tokenizer.save_pretrained(os.path.join(output_dir, 'final_model'))
        
        logger.info("Training completed!")
    
    def generate_spoiler(self, post_text, target_paragraph, max_new_tokens=30):
        """Generate spoiler for given post and paragraph"""
        
        # Preprocess input
        input_text = f"{post_text} {target_paragraph} <SEP>"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=450  # Leave room for generation
        ).to(self.device)
        
        # Generate with conservative settings (similar to paper)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy decoding
                top_p=None,
                top_k=None,
                num_beams=4,  # Small beam search
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract spoiler (text after <SEP>)
        if '<SEP>' in generated_text:
            spoiler = generated_text.split('<SEP>')[-1].strip()
        else:
            spoiler = generated_text[len(input_text.replace('<SEP>', '')):].strip()
        
        return spoiler
    
    def evaluate_model(self, test_data_path, output_file='evaluation_results.txt'):
        """Evaluate model performance using BLEU, ROUGE, BERTScore, METEOR"""
        
        # Load test data
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        # Generate predictions
        predictions = []
        references = []
        
        logger.info("Generating predictions...")
        for item in tqdm(test_data):
            post_text = item['postText'][0] if isinstance(item['postText'], list) else item['postText']
            target_paragraph = min(item['targetParagraphs'], key=len) if item['targetParagraphs'] else ""
            true_spoiler = item['spoiler'][0] if isinstance(item['spoiler'], list) else item['spoiler']
            
            # Generate prediction
            pred_spoiler = self.generate_spoiler(post_text, target_paragraph)
            
            predictions.append(pred_spoiler)
            references.append(true_spoiler)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, references)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Evaluation Results\n")
            f.write("==================\n\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n\nSample Predictions:\n")
            f.write("===================\n\n")
            for i in range(min(10, len(predictions))):
                f.write(f"Example {i+1}:\n")
                f.write(f"Reference: {references[i]}\n")
                f.write(f"Prediction: {predictions[i]}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"Evaluation completed. Results saved to {output_file}")
        return metrics
    
    def _calculate_metrics(self, predictions, references):
        """Calculate BLEU, ROUGE, BERTScore, and METEOR"""
        
        # BLEU Score
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores)
        
        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        avg_rouge = np.mean(rouge_scores)
        
        # BERTScore
        P, R, F1 = bert_score.score(predictions, references, lang='en', verbose=False)
        avg_bertscore = F1.mean().item()
        
        # METEOR (using evaluate library)
        try:
            meteor = evaluate.load('meteor')
            meteor_scores = []
            for pred, ref in zip(predictions, references):
                score = meteor.compute(predictions=[pred], references=[ref])
                meteor_scores.append(score['meteor'])
            avg_meteor = np.mean(meteor_scores)
        except:
            logger.warning("METEOR metric not available")
            avg_meteor = 0.0
        
        return {
            'BLEU': avg_bleu,
            'ROUGE-L': avg_rouge,
            'BERTScore': avg_bertscore,
            'METEOR': avg_meteor
        }

def main():
    """Main function to run clickbait spoiling"""
    
    # Initialize spoiler
    spoiler = ClickbaitSpoiler('gpt2-medium')
    
    # Paths
    train_path = 'data/train.jsonl'
    val_path = 'data/validation.jsonl'
    
    # Train model
    logger.info("Starting training process...")
    spoiler.train(train_path, val_path)
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    metrics = spoiler.evaluate_model(val_path, 'validation_results.txt')
    
    print("\nValidation Results:")
    print("===================")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Example usage
    print("\nExample Spoiler Generation:")
    print("===========================")
    
    example_post = "You Won't Believe What This Celebrity Did Next"
    example_paragraph = "The celebrity surprised everyone by announcing their retirement from acting to pursue a career in marine biology. The decision came after years of feeling unfulfilled in Hollywood."
    
    spoiler_text = spoiler.generate_spoiler(example_post, example_paragraph)
    print(f"Post: {example_post}")
    print(f"Generated Spoiler: {spoiler_text}")

if __name__ == "__main__":
    # Download required NLTK data
    import nltk
    nltk.download('punkt')
    
    main() 