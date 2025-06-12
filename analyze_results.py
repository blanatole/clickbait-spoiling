#!/usr/bin/env python3
import json

# Load our results
with open('detailed_evaluation_results.json', 'r') as f:
    our_results = json.load(f)

# Paper results
paper_results = {
    'BLEU-4': 0.58,
    'ROUGE-L': 0.68,
    'BERTScore': 0.46,
    'METEOR': 0.47
}

print('🎯 CLICKBAIT SPOILING - RESULTS COMPARISON')
print('=' * 60)
print()

# Our results
our_metrics = our_results['metrics']
print('📊 OUR RESULTS (Fine-tuned GPT-2 Medium):')
print(f'  BLEU-4:     {our_metrics["BLEU-4"]:.4f}')
print(f'  ROUGE-L:    {our_metrics["ROUGE-L"]:.4f}')
print(f'  BERTScore:  {our_metrics["BERTScore"]:.4f}')
print(f'  METEOR:     {our_metrics["METEOR"]:.4f}')
print()

print('📰 PAPER RESULTS (Target):')
for metric, value in paper_results.items():
    print(f'  {metric:<10}: {value:.4f}')
print()

print('🔍 COMPARISON & ANALYSIS:')
print('-' * 40)
for metric in ['BLEU-4', 'ROUGE-L', 'BERTScore', 'METEOR']:
    our_score = our_metrics[metric]
    paper_score = paper_results[metric]
    diff = our_score - paper_score
    ratio = our_score / paper_score if paper_score > 0 else 0
    
    status = '✅' if ratio > 0.8 else '⚠️' if ratio > 0.3 else '❌'
    print(f'{status} {metric:<10}: {our_score:>6.4f} vs {paper_score:>6.4f} ({diff:>+7.4f}, {ratio:.2%})')

print()
print('🧠 KEY INSIGHTS:')
if our_metrics['BLEU-4'] < 0.1 and our_metrics['BERTScore'] > 0.7:
    print('• High BERTScore but low BLEU/ROUGE: Model understands semantics but uses different wording')
if our_metrics['Avg_Length_Pred'] != our_metrics['Avg_Length_Ref']:
    ratio = our_metrics['Avg_Length_Pred'] / our_metrics['Avg_Length_Ref']
    print(f'• Length ratio: {ratio:.2f} ({"shorter" if ratio < 1 else "longer"} predictions)')
print('• Low BLEU suggests need for more conservative generation or longer training')
print()

print('💡 IMPROVEMENT SUGGESTIONS:')
print('1. Use full training (3,200 samples) instead of demo (500)')
print('2. Train for full 10 epochs with early stopping')
print('3. More conservative generation (beam=1, temperature=0)')
print('4. Better input preprocessing (shorter context)')
print('5. Post-processing to match reference length')
print()

print('📈 TRAINING DETAILS:')
print(f'• Training samples: 500 (demo mode, vs 3,200 full)')
print(f'• Validation samples: {our_results["total_samples"]}')
print(f'• Training epochs: 3 (vs 10 in paper)')
print(f'• Model: GPT-2 medium (355M parameters)')
print(f'• Hyperparameters: LR=5e-5, batch=4, cosine decay')
print()

print('🎯 CONCLUSION:')
print('While BERTScore shows semantic understanding (0.776), the low BLEU/ROUGE scores')
print('indicate the model needs more training time and conservative generation to match')
print('the exact wording of ground truth spoilers as achieved in the paper.') 