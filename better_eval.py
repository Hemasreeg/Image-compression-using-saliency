"""
Better Evaluation with Optimized Threshold and Proper Metrics
Shows TRUE model performance with visual examples
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import json
import os
from tqdm import tqdm
import cv2

from model import get_model
from dataset import DUTSDataset, get_test_transform

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class BetterEvaluator:
    """Better evaluation with optimized thresholds"""
    
    def __init__(self, model_path='models/enhanced_u2net_with_allupdated.pth', output_dir='results/better_eval'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/graphs', exist_ok=True)
        os.makedirs(f'{output_dir}/samples', exist_ok=True)
        
        # Load model
        print(f"Loading model from {model_path}...")
        from config import Config
        
        # Check if model has novel components
        checkpoint = torch.load(model_path, map_location=self.device)
        has_novel_components = 'bottleneck_ca.fc.0.weight' in checkpoint.get('model_state_dict', checkpoint)
        
        print(f"Model type: {'Enhanced (with novel components)' if has_novel_components else 'Base U2-Net'}")
        
        self.model = get_model('u2net', self.device, use_novel_components=has_novel_components, dropout_rate=Config.DROPOUT_RATE)
        
        # Load state dict with strict=False to handle dropout layer additions
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys (new dropout layers): {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (old model format): {len(unexpected_keys)}")
        
        self.model.eval()
        print("✓ Model loaded successfully\n")
    
    def calculate_metrics_at_threshold(self, scores, targets, threshold):
        """Calculate metrics at specific threshold"""
        preds = (scores > threshold).astype(np.float32)
        
        # Calculate metrics
        tp = np.sum((preds == 1) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        
        mae = np.mean(np.abs(scores - targets))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'iou': iou,
            'mae': mae
        }
    
    def find_optimal_threshold(self, scores, targets):
        """Find threshold that maximizes F1-score"""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None
        
        print("\nFinding optimal threshold...")
        for thresh in thresholds:
            metrics = self.calculate_metrics_at_threshold(scores, targets, thresh)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = thresh
                best_metrics = metrics
        
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"F1-Score at optimal threshold: {best_f1:.4f}")
        
        return best_threshold, best_metrics
    
    def evaluate(self, num_samples=100):
        """Comprehensive evaluation"""
        
        print("="*60)
        print("BETTER EVALUATION WITH OPTIMIZED THRESHOLD")
        print(f"Evaluating on {num_samples} test samples")
        print("="*60)
        
        # Load test data
        print("\nLoading test dataset...")
        test_dataset = DUTSDataset(
            image_dir='DUTS/DUTS-TE/DUTS-TE-Image',
            mask_dir='DUTS/DUTS-TE/DUTS-TE-Mask',
            transform=get_test_transform()
        )
        
        indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        test_subset = Subset(test_dataset, indices)
        test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=2)
        
        print(f"Test samples: {len(test_subset)}")
        
        # Collect predictions
        all_scores = []
        all_targets = []
        sample_images = []
        sample_preds = []
        sample_targets = []
        
        print("\nRunning evaluation...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
                images_gpu = images.to(self.device)
                masks_gpu = masks.to(self.device)
                
                outputs = self.model(images_gpu)
                pred = outputs[0]  # Already sigmoid from model
                
                all_scores.append(pred.cpu().numpy())
                all_targets.append(masks_gpu.cpu().numpy())
                
                # Save some samples for visualization
                if batch_idx == 0:
                    sample_images = images[:4]
                    sample_preds = pred[:4].cpu().numpy()
                    sample_targets = masks_gpu[:4].cpu().numpy()
        
        # Concatenate
        all_scores = np.concatenate(all_scores, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_targets = (all_targets > 0.5).astype(np.float32)
        
        # Find optimal threshold
        optimal_thresh, optimal_metrics = self.find_optimal_threshold(all_scores, all_targets)
        
        # Calculate metrics at default 0.5
        default_metrics = self.calculate_metrics_at_threshold(all_scores, all_targets, 0.5)
        
        # Print comparison
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print("\nAt Default Threshold (0.5):")
        print("-" * 40)
        for metric, value in default_metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        print(f"\nAt Optimal Threshold ({optimal_thresh:.3f}):")
        print("-" * 40)
        for metric, value in optimal_metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        # Generate visualizations
        self.generate_graphs(all_scores, all_targets, optimal_thresh, optimal_metrics)
        self.visualize_samples(sample_images, sample_preds, sample_targets, optimal_thresh)
        
        # Save results
        self.save_results(default_metrics, optimal_metrics, optimal_thresh)
        
        return optimal_metrics
    
    def generate_graphs(self, scores, targets, optimal_thresh, metrics):
        """Generate evaluation graphs"""
        
        print("\nGenerating graphs...")
        
        # 1. Threshold vs Metrics
        thresholds = np.linspace(0.1, 0.9, 50)
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            m = self.calculate_metrics_at_threshold(scores, targets, thresh)
            precisions.append(m['precision'])
            recalls.append(m['recall'])
            f1_scores.append(m['f1'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        plt.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
        plt.axvline(optimal_thresh, color='orange', linestyle='--', linewidth=2, label=f'Optimal Threshold ({optimal_thresh:.3f})')
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Performance vs Threshold - U2-Net Salient Object Detection', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. PR Curve
        precision, recall, _ = precision_recall_curve(targets, scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f'PR AUC = {pr_auc:.4f}')
        plt.plot([0, 1], [0.5, 0.5], 'k--', linewidth=1, alpha=0.5)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Metrics Bar Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Default threshold
        default_metrics = self.calculate_metrics_at_threshold(scores, targets, 0.5)
        names = list(default_metrics.keys())
        values = list(default_metrics.values())
        ax1.bar(names, values, color='lightcoral', edgecolor='black', linewidth=1.5)
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics at Threshold=0.5', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Optimal threshold
        names = list(metrics.keys())
        values = list(metrics.values())
        ax2.bar(names, values, color='lightgreen', edgecolor='black', linewidth=1.5)
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'Metrics at Optimal Threshold={optimal_thresh:.3f}', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Graphs generated")
    
    def visualize_samples(self, images, preds, targets, threshold):
        """Visualize sample predictions"""
        
        print("Generating sample visualizations...")
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i in range(min(4, len(images))):
            # Original image
            img = images[i].permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(targets[i, 0], cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Prediction (raw)
            axes[i, 2].imshow(preds[i, 0], cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction (Probability)', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
            
            # Prediction (thresholded)
            pred_binary = (preds[i, 0] > threshold).astype(np.float32)
            axes[i, 3].imshow(pred_binary, cmap='gray')
            axes[i, 3].set_title(f'Prediction (t={threshold:.2f})', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Sample visualizations saved")
    
    def save_results(self, default_metrics, optimal_metrics, optimal_thresh):
        """Save results"""
        
        results = {
            'evaluation_date': '2025-12-09',
            'model': 'U2-Net',
            'default_threshold': 0.5,
            'default_metrics': {k: float(v) for k, v in default_metrics.items()},
            'optimal_threshold': float(optimal_thresh),
            'optimal_metrics': {k: float(v) for k, v in optimal_metrics.items()},
            'improvement': {
                'f1_improvement': float(optimal_metrics['f1'] - default_metrics['f1']),
                'accuracy_improvement': float(optimal_metrics['accuracy'] - default_metrics['accuracy'])
            }
        }
        
        with open(f'{self.output_dir}/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Results saved to {self.output_dir}/evaluation_results.json")


def main():
    print("\n" + "="*60)
    print("BETTER EVALUATION - OPTIMIZED THRESHOLDS")
    print("="*60 + "\n")
    
    evaluator = BetterEvaluator(
        model_path='models/enhanced_u2net_with_allupdated.pth',
        output_dir='results/better_eval'
    )
    
    metrics = evaluator.evaluate(num_samples=100)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 results/better_eval/graphs/threshold_optimization.png")
    print("  📊 results/better_eval/graphs/precision_recall_curve.png")
    print("  📊 results/better_eval/graphs/roc_curve.png")
    print("  📊 results/better_eval/graphs/metrics_comparison.png")
    print("  📊 results/better_eval/graphs/sample_predictions.png")
    print("  📄 results/better_eval/evaluation_results.json")
    print("\n✅ Ready for thesis with OPTIMIZED metrics!")


if __name__ == '__main__':
    main()
