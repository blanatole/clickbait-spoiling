#!/usr/bin/env python3
"""
Evaluation Script for Clickbait Spoiling
Implements the exact metrics used in the paper for comparison.
"""

import json
import argparse
import logging
import numpy as np
from typing import List, Dict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import evaluate
from sacrebleu import BLEU
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoilerEvaluator:
    """Evaluation class matching paper metrics"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
        
        # Try to load METEOR
        try:
            self.meteor = evaluate.load('meteor')
        except:
            logger.warning("METEOR metric not available")
            self.meteor = None
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate predictions against references using paper metrics
        
        Args:
            predictions: List of predicted spoilers
            references: List of ground truth spoilers
            
        Returns:
            Dictionary with metric scores
        """
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        logger.info(f"Evaluating {len(predictions)} predictions...")
        
        # Calculate metrics
        metrics = {}
        
        # BLEU-4 (as used in paper)
        metrics['BLEU-4'] = self._calculate_bleu(predictions, references)
        
        # ROUGE-L (as used in paper)
        metrics['ROUGE-L'] = self._calculate_rouge_l(predictions, references)
        
        # BERTScore (as used in paper)
        metrics['BERTScore'] = self._calculate_bertscore(predictions, references)
        
        # METEOR (as used in paper)
        metrics['METEOR'] = self._calculate_meteor(predictions, references)
        
        # Additional metrics for analysis
        metrics['Avg_Length_Pred'] = np.mean([len(p.split()) for p in predictions])
        metrics['Avg_Length_Ref'] = np.mean([len(r.split()) for r in references])
        
        return metrics
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU-4 score using both NLTK and SacreBLEU"""
        
        # Method 1: NLTK BLEU (with smoothing for short texts)
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            
            # BLEU-4 with smoothing
            bleu = sentence_bleu(ref_tokens, pred_tokens, 
                               weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        nltk_bleu = np.mean(bleu_scores)
        
        # Method 2: SacreBLEU (corpus-level)
        try:
            # Prepare for SacreBLEU format
            refs_list = [[ref] for ref in references]
            sacre_bleu = self.bleu_scorer.corpus_score(predictions, list(zip(*refs_list)))
            sacre_score = sacre_bleu.score / 100.0  # Convert to 0-1 range
        except:
            sacre_score = nltk_bleu
        
        logger.info(f"BLEU-4 (NLTK): {nltk_bleu:.4f}, (SacreBLEU): {sacre_score:.4f}")
        
        # Return NLTK score for consistency with paper
        return nltk_bleu
    
    def _calculate_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE-L score"""
        
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        return np.mean(rouge_scores)
    
    def _calculate_bertscore(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BERTScore F1"""
        
        P, R, F1 = bert_score.score(predictions, references, lang='en', verbose=False)
        return F1.mean().item()
    
    def _calculate_meteor(self, predictions: List[str], references: List[str]) -> float:
        """Calculate METEOR score"""
        
        if self.meteor is None:
            return 0.0
        
        try:
            meteor_scores = []
            for pred, ref in zip(predictions, references):
                score = self.meteor.compute(predictions=[pred], references=[ref])
                meteor_scores.append(score['meteor'])
            return np.mean(meteor_scores)
        except Exception as e:
            logger.warning(f"METEOR calculation failed: {e}")
            return 0.0
    
    def detailed_analysis(self, predictions: List[str], references: List[str], 
                         examples_to_show: int = 10) -> Dict:
        """Provide detailed analysis of results"""
        
        analysis = {
            'total_samples': len(predictions),
            'metrics': self.evaluate_predictions(predictions, references),
            'length_analysis': self._analyze_lengths(predictions, references),
            'examples': self._get_sample_comparisons(predictions, references, examples_to_show)
        }
        
        return analysis
    
    def _analyze_lengths(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze length distributions"""
        
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        return {
            'pred_avg_length': np.mean(pred_lengths),
            'pred_std_length': np.std(pred_lengths),
            'ref_avg_length': np.mean(ref_lengths),
            'ref_std_length': np.std(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        }
    
    def _get_sample_comparisons(self, predictions: List[str], references: List[str], 
                              n: int) -> List[Dict]:
        """Get sample comparisons for analysis"""
        
        examples = []
        indices = np.random.choice(len(predictions), min(n, len(predictions)), replace=False)
        
        for i in indices:
            pred = predictions[i]
            ref = references[i]
            
            # Calculate individual scores
            bleu = sentence_bleu([nltk.word_tokenize(ref.lower())], 
                               nltk.word_tokenize(pred.lower()),
                               smoothing_function=SmoothingFunction().method1)
            rouge = self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure
            
            examples.append({
                'index': int(i),
                'reference': ref,
                'prediction': pred,
                'bleu': float(bleu),
                'rouge_l': float(rouge),
                'ref_length': len(ref.split()),
                'pred_length': len(pred.split())
            })
        
        return examples

def load_predictions_from_file(file_path: str) -> tuple:
    """Load predictions and references from various file formats"""
    
    predictions = []
    references = []
    
    if file_path.endswith('.json'):
        # JSON format from quick_spoiler.py
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            predictions.append(item.get('predicted_spoiler', ''))
            references.append(item.get('true_spoiler', ''))
    
    elif file_path.endswith('.jsonl'):
        # JSONL format (original data)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # This assumes we have predictions in the data
                if 'predicted_spoiler' in item:
                    predictions.append(item['predicted_spoiler'])
                    references.append(item.get('spoiler', [''])[0] if isinstance(item.get('spoiler'), list) else item.get('spoiler', ''))
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return predictions, references

def compare_with_paper_results(metrics: Dict[str, float]) -> None:
    """Compare results with reported paper metrics"""
    
    # Paper reported results (from the analysis provided)
    paper_results = {
        'BLEU-4': 0.58,
        'ROUGE-L': 0.68,
        'BERTScore': 0.46,
        'METEOR': 0.47
    }
    
    print("\n📊 Comparison with Paper Results:")
    print("=" * 50)
    print(f"{'Metric':<12} {'Our Result':<12} {'Paper':<12} {'Difference':<12}")
    print("-" * 50)
    
    for metric in ['BLEU-4', 'ROUGE-L', 'BERTScore', 'METEOR']:
        our_score = metrics.get(metric, 0.0)
        paper_score = paper_results.get(metric, 0.0)
        diff = our_score - paper_score
        
        print(f"{metric:<12} {our_score:<12.4f} {paper_score:<12.4f} {diff:<12.4f}")
    
    print("\n💡 Analysis Notes:")
    if metrics.get('BLEU-4', 0) < 0.1 and metrics.get('BERTScore', 0) > 0.7:
        print("• Low BLEU/ROUGE but high BERTScore suggests semantic correctness with different wording")
    if metrics.get('BLEU-4', 0) < paper_results['BLEU-4'] * 0.5:
        print("• Much lower BLEU suggests need for more conservative generation (lower temperature/beam)")
    if metrics.get('Avg_Length_Pred', 0) > metrics.get('Avg_Length_Ref', 0) * 2:
        print("• Predictions much longer than references - consider shortening generation")

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate Clickbait Spoiling Results')
    parser.add_argument('--predictions', type=str, required=True,
                       help='File with predictions (JSON or JSONL)')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output file for detailed results')
    parser.add_argument('--examples', type=int, default=10,
                       help='Number of examples to show in analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with paper results')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading predictions from {args.predictions}")
    predictions, references = load_predictions_from_file(args.predictions)
    
    if not predictions or not references:
        logger.error("No valid predictions/references found!")
        return
    
    logger.info(f"Loaded {len(predictions)} prediction-reference pairs")
    
    # Evaluate
    evaluator = SpoilerEvaluator()
    analysis = evaluator.detailed_analysis(predictions, references, args.examples)
    
    # Print results
    print("\n🎯 Clickbait Spoiling Evaluation Results")
    print("=" * 50)
    
    metrics = analysis['metrics']
    for metric, score in metrics.items():
        print(f"{metric:<15}: {score:.4f}")
    
    # Length analysis
    print(f"\n📏 Length Analysis:")
    length_analysis = analysis['length_analysis']
    print(f"Avg Prediction Length: {length_analysis['pred_avg_length']:.1f} words")
    print(f"Avg Reference Length:  {length_analysis['ref_avg_length']:.1f} words")
    print(f"Length Ratio:          {length_analysis['length_ratio']:.2f}")
    
    # Compare with paper if requested
    if args.compare:
        compare_with_paper_results(metrics)
    
    # Save detailed results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed analysis saved to {args.output}")
    
    # Show examples
    print(f"\n📝 Sample Comparisons (showing {len(analysis['examples'])}):")
    print("-" * 80)
    
    for i, example in enumerate(analysis['examples']):
        print(f"\nExample {i+1}:")
        print(f"Reference:  {example['reference']}")
        print(f"Prediction: {example['prediction']}")
        print(f"BLEU: {example['bleu']:.3f}, ROUGE-L: {example['rouge_l']:.3f}")
        print(f"Lengths: Ref={example['ref_length']}, Pred={example['pred_length']}")

if __name__ == "__main__":
    main() 