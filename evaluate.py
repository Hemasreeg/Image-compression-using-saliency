"""
Evaluation Script - Comprehensive evaluation on DUTS test set
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from config import Config
from model import get_model
from dataset import DUTSDataset, get_test_transform
from utils import calculate_metrics, calculate_f_measure, calculate_mae


def evaluate_model(model_path, model_type='u2net', data_dir=None, save_predictions=False, output_dir='eval_results'):
    """
    Comprehensive evaluation of the model
    
    Args:
        model_path: Path to trained model
        model_type: Model type
        data_dir: Dataset directory
        save_predictions: Whether to save prediction masks
        output_dir: Output directory for results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'\n{"="*70}')
    print(f'Model Evaluation - Salient Object Detection')
    print(f'{"="*70}')
    print(f'Device: {device}')
    print(f'Model: {model_type}')
    print(f'Model Path: {model_path}')
    print(f'{"="*70}\n')
    
    # Load model
    print('Loading model...')
    from config import Config
    model = get_model(model_type, device, dropout_rate=Config.DROPOUT_RATE)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'✓ Model loaded')
    if 'epoch' in checkpoint:
        print(f'✓ Trained for {checkpoint["epoch"]} epochs')
    if 'metrics' in checkpoint:
        print(f'✓ Training metrics: {checkpoint["metrics"]}')
    
    # Load dataset
    print('\nLoading test dataset...')
    if data_dir is None:
        data_dir = Config.DATA_ROOT
    
    test_dataset = DUTSDataset(
        root_dir=data_dir,
        split='test',
        transform=get_test_transform(Config.IMG_HEIGHT)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f'✓ Test samples: {len(test_dataset)}')
    
    # Create output directory
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
    
    # Evaluation
    all_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'accuracy': [],
        'mae': [],
        'f_measure': []
    }
    
    print('\nEvaluating...\n')
    
    with torch.no_grad():
        for idx, (image, mask, img_name) in enumerate(tqdm(test_loader, desc='Evaluation')):
            image = image.to(device)
            mask = mask.to(device)
            
            # Predict
            outputs = model(image)
            pred = outputs[0]  # Use fusion output
            
            # Calculate metrics
            metrics = calculate_metrics(pred, mask)
            
            for key in ['precision', 'recall', 'f1', 'iou', 'accuracy', 'mae']:
                all_metrics[key].append(metrics[key])
            
            # Calculate F-measure
            f_measure = calculate_f_measure(pred.squeeze().cpu(), mask.squeeze().cpu())
            all_metrics['f_measure'].append(f_measure)
            
            # Save prediction
            if save_predictions:
                pred_np = pred.squeeze().cpu().numpy()
                pred_img = (pred_np * 255).astype(np.uint8)
                
                save_name = img_name[0].replace('.jpg', '.png')
                save_path = os.path.join(pred_dir, save_name)
                
                import cv2
                cv2.imwrite(save_path, pred_img)
    
    # Calculate statistics
    print(f'\n{"="*70}')
    print(f'Evaluation Results')
    print(f'{"="*70}')
    print(f'Number of test images: {len(test_dataset)}\n')
    
    results = {}
    for key in all_metrics:
        values = all_metrics[key]
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        
        results[key] = {
            'mean': mean_val,
            'std': std_val,
            'max': max_val,
            'min': min_val
        }
        
        print(f'{key.upper()}:')
        print(f'  Mean: {mean_val:.4f}  Std: {std_val:.4f}')
        print(f'  Max:  {max_val:.4f}  Min: {min_val:.4f}\n')
    
    print(f'{"="*70}')
    
    # Overall accuracy assessment
    mean_f1 = results['f1']['mean']
    mean_iou = results['iou']['mean']
    
    print(f'\nOVERALL ASSESSMENT:')
    print(f'{"─"*70}')
    
    if mean_f1 >= 0.90 and mean_iou >= 0.85:
        print(f'✓ EXCELLENT - Model achieves 90%+ accuracy!')
    elif mean_f1 >= 0.85 and mean_iou >= 0.80:
        print(f'✓ VERY GOOD - Model achieves 85%+ accuracy')
    elif mean_f1 >= 0.80 and mean_iou >= 0.75:
        print(f'✓ GOOD - Model achieves 80%+ accuracy')
    else:
        print(f'○ MODERATE - Consider more training or tuning')
    
    print(f'{"─"*70}')
    print(f'F1 Score: {mean_f1:.4f} ({mean_f1*100:.2f}%)')
    print(f'IoU Score: {mean_iou:.4f} ({mean_iou*100:.2f}%)')
    print(f'{"─"*70}\n')
    
    # Save results
    if save_predictions:
        results_file = os.path.join(output_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write('Evaluation Results\n')
            f.write('='*70 + '\n\n')
            for key in results:
                f.write(f'{key.upper()}:\n')
                f.write(f'  Mean: {results[key]["mean"]:.4f}\n')
                f.write(f'  Std:  {results[key]["std"]:.4f}\n')
                f.write(f'  Max:  {results[key]["max"]:.4f}\n')
                f.write(f'  Min:  {results[key]["min"]:.4f}\n\n')
        
        print(f'Results saved to: {results_file}')
    
    # Plot metrics distribution
    if save_predictions:
        plot_metrics_distribution(all_metrics, output_dir)
    
    return results


def plot_metrics_distribution(all_metrics, output_dir):
    """Plot distribution of metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metrics Distribution', fontsize=16)
    
    metrics_to_plot = ['precision', 'recall', 'f1', 'iou', 'accuracy', 'mae']
    
    for idx, key in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = all_metrics[key]
        
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
        ax.set_title(key.upper())
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Metrics distribution plot saved to: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate U2-Net on DUTS test set')
    
    parser.add_argument('--model_path', type=str, default='outputs/models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='u2net',
                       choices=['u2net', 'u2net_lite'],
                       help='Model type')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: from config)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        data_dir=args.data_dir,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
