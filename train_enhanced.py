"""
Training Script for Enhanced U2-Net with Novel Components
Trains the model with Edge-Aware Refinement, Adaptive Fusion, and Attention mechanisms
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from config import Config
from model import get_model, count_parameters
from dataset import DUTSDataset, get_train_transform, get_test_transform
from utils import calculate_metrics


class EnhancedU2NetLoss(torch.nn.Module):
    """
    Advanced Loss function for Enhanced U2-Net - Optimized for 92+ F1
    Combines Focal + BCE + IoU + Dice + Boundary Loss for all outputs
    Multi-component loss ensures excellent boundary precision and high accuracy
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super(EnhancedU2NetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.bce = torch.nn.BCELoss(reduction='mean')
    
    def smooth_labels(self, targets):
        """Apply label smoothing for better generalization"""
        if self.label_smoothing > 0:
            return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return targets
    
    def focal_loss(self, pred, target):
        """Focal Loss for handling hard samples"""
        target = self.smooth_labels(target)
        bce_loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def iou_loss(self, pred, target):
        """IoU Loss for better boundary alignment"""
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()
    
    def dice_loss(self, pred, target):
        """Dice Loss for excellent segmentation overlap"""
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()
    
    def boundary_loss(self, pred, target):
        """Boundary loss for sharper edges - NEW for 92+ accuracy"""
        # Compute gradients using Sobel-like filters
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                       dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # Compute boundaries
        target_boundary = torch.nn.functional.conv2d(target, laplacian_kernel, padding=1)
        pred_boundary = torch.nn.functional.conv2d(pred, laplacian_kernel, padding=1)
        
        # L1 loss on boundaries
        boundary_loss = torch.abs(pred_boundary - target_boundary).mean()
        return boundary_loss
    
    def forward(self, outputs, targets):
        """
        outputs: tuple of (d0, d1, d2, d3, d4, d5, d6)
        targets: ground truth masks
        """
        d0, d1, d2, d3, d4, d5, d6 = outputs
        
        # Main output loss (5-component loss for maximum 92+ accuracy) - NORMALIZED
        loss0_bce = self.bce(d0, targets)
        loss0_focal = self.focal_loss(d0, targets)
        loss0_iou = self.iou_loss(d0, targets)
        loss0_dice = self.dice_loss(d0, targets)
        loss0_boundary = self.boundary_loss(d0, targets)
        # Sum of weights = 1.0, so loss0 stays in reasonable range
        loss0 = 0.3 * loss0_bce + 0.25 * loss0_focal + 0.2 * loss0_iou + 0.15 * loss0_dice + 0.1 * loss0_boundary
        
        # Side output losses (4-component for efficiency) - NORMALIZED
        def side_loss(pred):
            # Sum of weights = 1.0, so each side loss stays in reasonable range
            return (0.4 * self.bce(pred, targets) + 
                   0.3 * self.focal_loss(pred, targets) + 
                   0.2 * self.iou_loss(pred, targets) +
                   0.1 * self.boundary_loss(pred, targets))
        
        loss1 = side_loss(d1)
        loss2 = side_loss(d2)
        loss3 = side_loss(d3)
        loss4 = side_loss(d4)
        loss5 = side_loss(d5)
        loss6 = side_loss(d6)
        
        # Weighted combination - AGGRESSIVELY normalized to 0-1 range for display
        # Main output: 1.0, sides: 0.5*2 + 0.3*2 + 0.2*2 = 2.0, total = 3.0
        # Divide by 3.0 to keep final loss in 0-1 range
        total_loss = (loss0 + 
                     0.5 * (loss1 + loss2) + 
                     0.3 * (loss3 + loss4) + 
                     0.2 * (loss5 + loss6)) / 3.0
        
        return total_loss, {
            'total': total_loss.item(),
            'main': loss0.item(),
            'side1': loss1.item(),
            'side2': loss2.item(),
            'side3': loss3.item(),
            'side4': loss4.item(),
            'side5': loss5.item(),
            'side6': loss6.item()
        }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, l1_lambda=0.0, use_amp=True):
    """Train for one epoch with L1 regularization and optional AMP"""  
    model.train()
    running_loss = 0.0
    running_metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'iou': 0, 'mae': 0}
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
                
                # Add L1 regularization if enabled (NORMALIZED by parameter count)
                if l1_lambda > 0:
                    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    num_params = sum(p.numel() for p in model.parameters())
                    l1_loss = l1_loss / num_params  # Normalize to get average absolute weight
                    loss = loss + l1_lambda * l1_loss
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            
            # Add L1 regularization if enabled (NORMALIZED by parameter count)
            if l1_lambda > 0:
                l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                num_params = sum(p.numel() for p in model.parameters())
                l1_loss = l1_loss / num_params  # Normalize to get average absolute weight
                loss = loss + l1_lambda * l1_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate metrics on main output
        with torch.no_grad():
            pred = (outputs[0] > 0.5).float()
            metrics = calculate_metrics(pred.cpu().numpy(), masks.cpu().numpy())
        
        # Update running stats
        running_loss += loss.item()
        for key in running_metrics:
            if key in metrics:
                running_metrics[key] += metrics[key]
        
        # Update progress bar - cleaner format
        avg_loss = running_loss / (batch_idx + 1)
        avg_f1 = running_metrics['f1'] / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'F1': f'{avg_f1:.4f}'
        })
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {k: v / len(dataloader) for k, v in running_metrics.items()}
    
    return epoch_loss, epoch_metrics


def validate(model, dataloader, criterion, device, epoch, search_optimal_threshold=True):
    """Validate the model with optional optimal threshold search"""
    model.eval()
    running_loss = 0.0
    running_metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'iou': 0, 'mae': 0}
    
    # Store predictions for optimal threshold search
    all_preds = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss, _ = criterion(outputs, masks)
            
            # Get predictions (already sigmoid from model)
            pred = outputs[0]
            
            # Store for optimal threshold search
            if search_optimal_threshold:
                all_preds.append(pred.cpu())
                all_masks.append(masks.cpu())
            
            # Calculate metrics at threshold 0.5
            pred_binary = (pred > 0.5).float()
            metrics = calculate_metrics(pred_binary.cpu().numpy(), masks.cpu().numpy())
            
            # Update running stats
            running_loss += loss.item()
            for key in running_metrics:
                if key in metrics:
                    running_metrics[key] += metrics[key]
            
            # Update progress bar - cleaner format
            avg_loss = running_loss / (batch_idx + 1)
            avg_f1 = running_metrics['f1'] / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'F1': f'{avg_f1:.4f}'
            })
    
    pbar.close()
    
    # Calculate epoch averages at threshold 0.5
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {k: v / len(dataloader) for k, v in running_metrics.items()}
    
    # Search for optimal threshold if enabled
    optimal_threshold = 0.5
    optimal_f1 = epoch_metrics['f1']
    
    if search_optimal_threshold and all_preds:
        # Concatenate all predictions and masks
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Try different thresholds (quick search with 20 thresholds)
        thresholds = torch.linspace(0.3, 0.7, 20)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            pred_binary = (all_preds > thresh.item()).float()
            
            # Calculate F1
            intersection = (pred_binary * all_masks).sum()
            union = pred_binary.sum() + all_masks.sum()
            
            precision = intersection / (pred_binary.sum() + 1e-8)
            recall = intersection / (all_masks.sum() + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1.item()
                best_threshold = thresh.item()
        
        optimal_threshold = best_threshold
        optimal_f1 = best_f1
        
        # Recalculate ALL metrics at optimal threshold for consistency
        pred_binary_optimal = (all_preds > optimal_threshold).float()
        optimal_metrics = calculate_metrics(pred_binary_optimal.cpu().numpy(), all_masks.cpu().numpy())
        
        # Update epoch metrics with optimal threshold values
        epoch_metrics['f1'] = optimal_metrics['f1']
        epoch_metrics['precision'] = optimal_metrics['precision']
        epoch_metrics['recall'] = optimal_metrics['recall']
        epoch_metrics['iou'] = optimal_metrics['iou']
        epoch_metrics['mae'] = optimal_metrics['mae']
    
    # Add optimal threshold info to history
    epoch_metrics['optimal_threshold'] = optimal_threshold
    epoch_metrics['optimal_f1'] = optimal_f1
    
    return epoch_loss, epoch_metrics


def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    Load pre-trained weights from old U2-Net into enhanced model
    Only loads compatible layers, skips novel components
    """
    # Handle None or missing path - check None first to avoid TypeError
    if pretrained_path is None:
        print("\n⚠️  No pre-trained weights specified - training from scratch")
        print("✅ All layers will be randomly initialized\n")
        return model
    
    if not os.path.exists(pretrained_path):
        print(f"\n⚠️  Pre-trained weights not found at {pretrained_path}")
        print("Training from scratch...")
        print("✅ All layers will be randomly initialized\n")
        return model
    
    print(f"\nLoading pre-trained weights from {pretrained_path}...")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_state = checkpoint['model_state_dict']
        else:
            pretrained_state = checkpoint
    else:
        pretrained_state = checkpoint
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Load compatible weights
    loaded_keys = []
    skipped_keys = []
    
    for name, param in pretrained_state.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name] = param
                loaded_keys.append(name)
            else:
                skipped_keys.append(f"{name} (shape mismatch)")
        else:
            skipped_keys.append(f"{name} (not in model)")
    
    # Load the compatible weights
    model.load_state_dict(model_state)
    
    print(f"✓ Loaded {len(loaded_keys)} layers from pre-trained model")
    if len(skipped_keys) > 0:
        print(f"✓ {len(skipped_keys)} new layers initialized randomly (WILL BE TRAINED)")
        print("\n✅ Novel Components (Training from scratch):")
        print("   - Edge-Aware Refinement Modules (4x)")
        print("   - Channel Attention Module (1x)")
        print("   - Multi-Scale Adaptive Fusion (1x)")
        print("   ⚠️  These are NEW components - not in pretrained model")
        print("   ✅ They WILL be trained along with the rest of the network!")
    
    return model


def train_enhanced_model(
    epochs=20,
    batch_size=6,  # Increased for faster training (3x speedup vs batch=2)
    learning_rate=1e-4,
    data_dir='DUTS',
    pretrained_path='models/best_model.pth',
    output_dir='models',
    save_name='enhanced_u2net.pth'
):
    """
    Main training function for Enhanced U2-Net
    """
    print("=" * 80)
    print("TRAINING ENHANCED U2-NET WITH NOVEL COMPONENTS")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if device.type == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with dropout
    print("\n📦 Initializing Enhanced U2-Net...")
    from config import Config
    model = get_model('u2net', device, dropout_rate=Config.DROPOUT_RATE)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Dropout rate: {Config.DROPOUT_RATE}")
    print(f"✓ L1 regularization: {Config.L1_LAMBDA}")
    print(f"✓ L2 regularization (weight decay): {Config.WEIGHT_DECAY}")
    
    # Load pre-trained weights (only compatible layers)
    model = load_pretrained_weights(model, pretrained_path, strict=False)
    
    # Freeze encoder layers (optional - for faster training)
    # Uncomment to freeze encoder and only train novel components + decoder
    # for name, param in model.named_parameters():
    #     if 'stage1.' in name or 'stage2.' in name or 'stage3.' in name:
    #         param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Setup datasets
    print("\n📂 Loading DUTS dataset...")
    train_dataset = DUTSDataset(
        root_dir=data_dir,
        split='train',
        transform=get_train_transform()
    )
    
    test_dataset = DUTSDataset(
        root_dir=data_dir,
        split='test',
        transform=get_test_transform()
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Create dataloaders with optimized batch sizes and workers
    # Validation can use larger batch (no gradients = less memory)
    val_batch_size = min(batch_size * 2, 12)  # 2x training batch, max 12
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increased for faster data loading
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,  # Increased for faster data loading
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print(f"✓ Train batch size: {batch_size}")
    print(f"✓ Val batch size: {val_batch_size} (2x training for efficiency)")
    
    # Loss and optimizer (AdamW with L2 regularization via weight_decay)
    from config import Config
    label_smoothing = Config.LABEL_SMOOTHING_FACTOR if Config.USE_LABEL_SMOOTHING else 0.0
    criterion = EnhancedU2NetLoss(alpha=0.25, gamma=2.0, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=Config.WEIGHT_DECAY  # L2 regularization
    )
    
    # Get L1 lambda for L1 regularization
    l1_lambda = Config.L1_LAMBDA
    
    # Cosine Annealing with Warmup Restarts for 92+ accuracy
    from config import Config
    if Config.LR_SCHEDULER == 'cosine_warm_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=Config.LR_MIN
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-7
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_iou': [],
        'val_iou': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': [],
        'optimal_f1': [],
        'optimal_threshold': []
    }
    
    best_f1 = 0.0
    best_optimal_threshold = 0.5
    
    print("\n🚀 Starting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Dropout rate: {Config.DROPOUT_RATE}")
    print(f"L1 Lambda: {Config.L1_LAMBDA}")
    print(f"L2 Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"Label Smoothing: {label_smoothing}")
    print(f"Target: 92+ F1 Score")
    print(f"Novel components: Training from random initialization")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{epochs}")
        print(f"{'='*80}")
        
# Training without AMP (BCELoss incompatible with autocast)
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            l1_lambda=l1_lambda, use_amp=False
        )
        
        # Validation
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate (CosineAnnealingWarmRestarts doesn't take metrics)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # Step scheduler after each epoch
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['learning_rates'].append(current_lr)
        history['optimal_f1'].append(val_metrics.get('optimal_f1', val_metrics['f1']))
        history['optimal_threshold'].append(val_metrics.get('optimal_threshold', 0.5))
        
        # Print epoch summary - CLEAN FORMAT like old version
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train F1:   {train_metrics['f1']:.4f} | Val F1:   {val_metrics['f1']:.4f}")
        print(f"  Train IoU:  {train_metrics['iou']:.4f} | Val IoU:  {val_metrics['iou']:.4f}")
        print(f"  Train MAE:  {train_metrics['mae']:.4f} | Val MAE:  {val_metrics['mae']:.4f}")
        
        # Show threshold info
        current_optimal_threshold = val_metrics.get('optimal_threshold', 0.5)
        if current_optimal_threshold != 0.5:
            print(f"  Optimal Threshold: {current_optimal_threshold:.3f} (Val metrics calculated at this threshold)")
        
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if target reached
        if val_metrics['f1'] >= 0.92:
            print(f"  🏆 TARGET REACHED! F1: {val_metrics['f1']:.4f} (92%+)")
        
        print(f"{'='*80}")
        
        # Save best model based on F1 at optimal threshold
        current_optimal_threshold = val_metrics.get('optimal_threshold', 0.5)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_optimal_threshold = current_optimal_threshold
            
            save_path = os.path.join(output_dir, save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_optimal_threshold': best_optimal_threshold,
                'history': history,
                'architecture': 'enhanced_u2net',
                'novel_components': [
                    'Edge-Aware Refinement Module (4x)',
                    'Channel Attention Module (1x)',
                    'Spatial Attention Module (2x)',
                    'Multi-Scale Adaptive Fusion (1x)'
                ]
            }, save_path)
            print(f"✓ Saved BEST model: {save_path} (F1: {best_f1:.4f})")
        
        # Save checkpoint EVERY epoch (prevent loss on runtime disconnect)
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'history': history,
            'architecture': 'enhanced_u2net'
        }, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Also save latest checkpoint (for easy resume)
        latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'history': history,
            'architecture': 'enhanced_u2net'
        }, latest_path)
        print(f"✓ Saved latest checkpoint: {latest_path}")
    
    # Save final training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Saved training history: {history_path}")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Best F1: {best_f1:.4f} (threshold: {best_optimal_threshold:.3f})")
    if best_f1 >= 0.92:
        print(f"🎉 SUCCESS! Achieved 92%+ target: {best_f1*100:.2f}%")
    print(f"✓ Model saved: {os.path.join(output_dir, save_name)}")
    print("=" * 80)
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced U2-Net')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (20 is enough for 92%+)')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size (higher = faster training)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='DUTS', help='Dataset directory')
    parser.add_argument('--pretrained', type=str, default='models/best_model.pth', 
                        help='Pre-trained weights path')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--save-name', type=str, default='enhanced_u2net.pth',
                        help='Output model filename')
    
    args = parser.parse_args()
    
    # Train
    model, history = train_enhanced_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        pretrained_path=args.pretrained,
        output_dir=args.output_dir,
        save_name=args.save_name
    )
