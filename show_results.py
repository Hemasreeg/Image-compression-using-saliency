#!/usr/bin/env python3
"""Display evaluation results in a formatted way"""

import json
import os

results_file = 'results/better_eval/evaluation_results.json'

if not os.path.exists(results_file):
    print("❌ Evaluation results not found. Run better_eval.py first.")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

print()
print("=" * 70)
print(" " * 20 + "EVALUATION RESULTS")
print("=" * 70)
print()
print(f"📅 Evaluation Date: {data['evaluation_date']}")
print(f"🤖 Model: {data['model']}")
print()

print("At Default Threshold (0.5):")
print("-" * 50)
for key, value in data['default_metrics'].items():
    print(f"  {key:20s}: {value:.4f}")

print()
print(f"At Optimal Threshold ({data['optimal_threshold']:.3f}):")
print("-" * 50)
for key, value in data['optimal_metrics'].items():
    print(f"  {key:20s}: {value:.4f}")

print()
print("📈 Improvement:")
print("-" * 50)
f1_improvement = (data['optimal_metrics']['f1'] - data['default_metrics']['f1']) * 100
accuracy_improvement = (data['optimal_metrics']['accuracy'] - data['default_metrics']['accuracy']) * 100
print(f"  F1 Score Gain:       +{f1_improvement:.2f}% absolute")
print(f"  Accuracy Gain:       +{accuracy_improvement:.2f}% absolute")

print()
print("=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print()

# Check for generated files
print("Generated files:")
graphs = [
    'results/better_eval/graphs/threshold_optimization.png',
    'results/better_eval/graphs/precision_recall_curve.png',
    'results/better_eval/graphs/roc_curve.png',
    'results/better_eval/graphs/metrics_comparison.png',
    'results/better_eval/graphs/sample_predictions.png',
]

for graph in graphs:
    if os.path.exists(graph):
        print(f"  📊 {graph}")
    else:
        print(f"  ⚠️  {graph} (not found)")

if os.path.exists(results_file):
    print(f"  📄 {results_file}")

print()
print("✅ Ready for thesis with OPTIMIZED metrics!")
print()
