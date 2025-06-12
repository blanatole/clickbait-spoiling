#!/usr/bin/env python3
"""
Quick Clickbait Spoiler Generator
Implementing the optimized generation strategy based on paper analysis.
"""

import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickSpoiler:
    """Quick spoiler generator for immediate testing"""
    
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            logger.info(f"Loaded fine-tuned model from {model_path}")
        else:
            # Use pre-trained GPT-2 medium
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            logger.info("Using pre-trained GPT-2 medium")
        
        # Add special tokens if not present
        if '<PAD>' not in self.tokenizer.get_vocab():
            special_tokens = {'pad_token': '<PAD>', 'sep_token': '<SEP>'}
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def generate_spoiler(self, post_text, target_paragraph="", strategy="conservative"):
        """
        Generate spoiler with different strategies based on paper analysis
        
        Args:
            post_text: The clickbait title/post
            target_paragraph: Optional target paragraph for context
            strategy: 'conservative' (greedy), 'beam', or 'creative'
        """
        
        # Clean input text
        post_text = self._clean_text(post_text)
        target_paragraph = self._clean_text(target_paragraph)
        
        # Create input prompt
        if target_paragraph:
            input_text = f"Clickbait: {post_text}\nContext: {target_paragraph[:200]}\nSpoiler:"
        else:
            input_text = f"Clickbait: {post_text}\nSpoiler:"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=400
        ).to(self.device)
        
        # Generation parameters based on strategy
        gen_params = self._get_generation_params(strategy)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_params
            )
        
        # Decode and extract spoiler
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        spoiler = self._extract_spoiler(generated_text, input_text)
        
        return spoiler
    
    def _get_generation_params(self, strategy):
        """Get generation parameters based on strategy"""
        
        if strategy == "conservative":
            # Paper-like settings: greedy/low beam
            return {
                'max_new_tokens': 30,
                'temperature': 0.0,  # Greedy
                'num_beams': 1,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
        
        elif strategy == "beam":
            # Small beam search as suggested
            return {
                'max_new_tokens': 30,
                'temperature': 0.0,
                'num_beams': 4,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
        
        elif strategy == "creative":
            # More diverse but controlled generation
            return {
                'max_new_tokens': 50,
                'temperature': 0.7,
                'top_p': 0.8,
                'num_beams': 2,
                'no_repeat_ngram_size': 2,
                'early_stopping': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove HTML tags, URLs, excessive whitespace
        import re
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'http[s]?://\S+', '', text)  # URLs
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        return text.strip()
    
    def _extract_spoiler(self, generated_text, input_text):
        """Extract spoiler from generated text"""
        # Remove input prompt
        spoiler = generated_text[len(input_text):].strip()
        
        # Clean up common artifacts
        spoiler = spoiler.split('\n')[0]  # Take first line
        spoiler = spoiler.strip('.,!?:;')  # Remove trailing punctuation
        
        # Limit length
        words = spoiler.split()
        if len(words) > 15:  # Keep spoilers concise
            spoiler = ' '.join(words[:15])
        
        return spoiler.strip()
    
    def batch_generate(self, data_file, output_file, strategy="conservative"):
        """Generate spoilers for a batch of examples"""
        
        results = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Extract data
                    post_text = item['postText'][0] if isinstance(item['postText'], list) else item['postText']
                    target_paragraph = min(item['targetParagraphs'], key=len) if item.get('targetParagraphs') else ""
                    true_spoiler = item.get('spoiler', [''])[0] if isinstance(item.get('spoiler'), list) else item.get('spoiler', '')
                    
                    # Generate spoiler
                    pred_spoiler = self.generate_spoiler(post_text, target_paragraph, strategy)
                    
                    # Store result
                    result = {
                        'uuid': item.get('uuid', f'item_{line_num}'),
                        'post_text': post_text,
                        'true_spoiler': true_spoiler,
                        'predicted_spoiler': pred_spoiler,
                        'strategy': strategy
                    }
                    results.append(result)
                    
                    if line_num % 100 == 0:
                        logger.info(f"Processed {line_num} examples")
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated spoilers for {len(results)} examples. Saved to {output_file}")
        return results

def test_examples():
    """Test with some example clickbait posts"""
    
    spoiler = QuickSpoiler()
    
    test_cases = [
        {
            "post": "You Won't Believe What This Celebrity Did Next",
            "context": "The celebrity surprised everyone by announcing their retirement from acting to pursue a career in marine biology."
        },
        {
            "post": "This Simple Trick Will Change Your Life Forever",
            "context": "A doctor discovered that drinking water first thing in the morning can boost metabolism by 30%."
        },
        {
            "post": "The Secret That Big Companies Don't Want You To Know",
            "context": "Many household products contain ingredients that can be replaced with simple, cheaper alternatives."
        },
        {
            "post": "Scientists Hate Him for This One Weird Discovery",
            "context": "A researcher found that a common kitchen spice can significantly reduce inflammation."
        }
    ]
    
    print("🔥 Quick Clickbait Spoiler Generation Test")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nExample {i}:")
        print(f"📰 Clickbait: {case['post']}")
        print(f"📝 Context: {case['context'][:100]}...")
        
        # Test different strategies
        for strategy in ["conservative", "beam", "creative"]:
            spoiler_text = spoiler.generate_spoiler(case['post'], case['context'], strategy)
            print(f"🎯 Spoiler ({strategy}): {spoiler_text}")
        
        print("-" * 30)

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Quick Clickbait Spoiler Generator')
    parser.add_argument('--post', type=str, help='Clickbait post text')
    parser.add_argument('--context', type=str, default='', help='Optional context paragraph')
    parser.add_argument('--strategy', choices=['conservative', 'beam', 'creative'], 
                        default='conservative', help='Generation strategy')
    parser.add_argument('--batch', type=str, help='Process batch file (JSONL format)')
    parser.add_argument('--output', type=str, default='spoiler_results.json', 
                        help='Output file for batch processing')
    parser.add_argument('--model', type=str, help='Path to fine-tuned model')
    parser.add_argument('--test', action='store_true', help='Run test examples')
    
    args = parser.parse_args()
    
    # Initialize spoiler
    spoiler = QuickSpoiler(args.model)
    
    if args.test:
        test_examples()
    elif args.batch:
        spoiler.batch_generate(args.batch, args.output, args.strategy)
    elif args.post:
        spoiler_text = spoiler.generate_spoiler(args.post, args.context, args.strategy)
        print(f"🎯 Spoiler: {spoiler_text}")
    else:
        # Interactive mode
        print("🔥 Interactive Clickbait Spoiler Generator")
        print("Type 'quit' to exit")
        
        while True:
            post = input("\n📰 Enter clickbait post: ").strip()
            if post.lower() == 'quit':
                break
            
            context = input("📝 Enter context (optional): ").strip()
            
            for strategy in ["conservative", "beam"]:
                spoiler_text = spoiler.generate_spoiler(post, context, strategy)
                print(f"🎯 Spoiler ({strategy}): {spoiler_text}")

if __name__ == "__main__":
    import os
    main() 