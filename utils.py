"""
Loss functions and utility functions for U2-Net training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss"""
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def forward(self, pred, target):
        return self.bce_loss(pred, target)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """Intersection over Union Loss"""
    
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class U2NetLoss(nn.Module):
    """
    Multi-scale loss for U2-Net
    Combines losses from all side outputs and fusion output
    """
    
    def __init__(self, loss_weights=None):
        super(U2NetLoss, self).__init__()
        
        # Default weights for 7 outputs (fusion + 6 side outputs)
        if loss_weights is None:
            self.loss_weights = [1.0, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5]
        else:
            self.loss_weights = loss_weights
        
        self.bce = BCELoss()
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: Tuple of (d0, d1, d2, d3, d4, d5, d6)
                    d0: fusion output
                    d1-d6: side outputs
            target: Ground truth mask
        
        Returns:
            total_loss: Weighted sum of all losses
        """
        total_loss = 0
        
        for i, output in enumerate(outputs):
            loss = self.bce(output, target)
            total_loss += self.loss_weights[i] * loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate evaluation metrics
    
    Args:
        pred: Predicted masks (B, 1, H, W) - values in [0, 1]
        target: Ground truth masks (B, 1, H, W) - values in [0, 1]
        threshold: Threshold for binary prediction
    
    Returns:
        dict: Dictionary containing precision, recall, f1, iou, and mae
    """
    with torch.no_grad():
        # Binarize predictions
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Mean Absolute Error
        mae = torch.mean(torch.abs(pred - target)).item()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
            'mae': mae
        }


def calculate_f_measure(pred, target, beta=0.3):
    """
    Calculate F-beta measure (weighted F-measure)
    Commonly used in salient object detection evaluation
    
    Args:
        pred: Predicted mask (H, W) - values in [0, 1]
        target: Ground truth mask (H, W) - values in [0, 1]
        beta: Weight parameter (beta^2 = 0.3 is standard for SOD)
    
    Returns:
        f_measure: F-beta score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Precision and Recall
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # F-beta measure
    f_measure = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-8)
    
    return f_measure.item()


def calculate_s_measure(pred, target, alpha=0.5):
    """
    Calculate S-measure (Structure measure)
    Paper: "Structure-measure: A New Way to Evaluate Foreground Maps"
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        alpha: Weight between object-aware and region-aware
    
    Returns:
        s_measure: Structure measure score
    """
    # This is a simplified version
    # Full implementation would include object-aware and region-aware components
    
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    # Simplified S-measure using correlation
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    correlation = (pred_centered * target_centered).sum() / \
                  (torch.sqrt((pred_centered ** 2).sum()) * torch.sqrt((target_centered ** 2).sum()) + 1e-8)
    
    return correlation.item()


def calculate_mae(pred, target):
    """
    Calculate Mean Absolute Error
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
    
    Returns:
        mae: Mean absolute error
    """
    return torch.mean(torch.abs(pred - target)).item()


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, filepath, optimizer=None, scheduler=None, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0), checkpoint.get('metrics', {})


def count_parameters(model):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        preds: Predicted masks (numpy array)
        targets: Ground truth masks (numpy array)
        threshold: Threshold for binary prediction
    
    Returns:
        dict: Dictionary of metrics (F1, Precision, Recall, IoU, MAE)
    """
    # Ensure binary
    preds_binary = (preds > threshold).astype(np.float32)
    targets_binary = (targets > threshold).astype(np.float32)
    
    # Flatten
    preds_flat = preds_binary.reshape(-1)
    targets_flat = targets_binary.reshape(-1)
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((preds_flat == 1) & (targets_flat == 1))
    fp = np.sum((preds_flat == 1) & (targets_flat == 0))
    fn = np.sum((preds_flat == 0) & (targets_flat == 1))
    tn = np.sum((preds_flat == 0) & (targets_flat == 0))
    
    # Precision
    precision = tp / (tp + fp + 1e-8)
    
    # Recall
    recall = tp / (tp + fn + 1e-8)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'iou': float(iou),
        'mae': float(mae),
        'accuracy': float(accuracy)
    }
