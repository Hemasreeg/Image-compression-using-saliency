"""
Training Script for U2-Net Salient Object Detection
Achieves 90%+ accuracy on DUTS dataset
"""

import os
import gc
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime

from config import Config
from model import get_model, count_parameters
from dataset import get_dataloaders
from utils import (U2NetLoss, calculate_metrics, EarlyStopping, 
                   save_checkpoint, get_lr)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, use_amp=True, config=None):
    """Train for one epoch with memory optimization support"""
    if config is None:
        config = Config
    
    model.train()
    running_loss = 0.0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0, 'mae': 0}
    
    accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
    enable_cleanup = getattr(config, 'ENABLE_MEMORY_CLEANUP', False)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Mixed precision training
        if use_amp:
            with torch.cuda.amp.autocast():
                # U2-Net returns 7 outputs (fusion + 6 side outputs)
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation: only step optimizer every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * accumulation_steps
        
        # Calculate metrics on fusion output (outputs[0])
        with torch.no_grad():
            metrics = calculate_metrics(outputs[0], masks)
            for key in all_metrics:
                all_metrics[key] += metrics[key]
            
            # Memory cleanup for GPU
            if enable_cleanup:
                del outputs, images, masks
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'f1': all_metrics['f1'] / (batch_idx + 1),
            'gpu_mb': f'{torch.cuda.memory_allocated() / 1e6:.0f}' if torch.cuda.is_available() else 'N/A'
        })
    
    # Average metrics
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches
    
    return avg_loss, all_metrics


def validate(model, dataloader, criterion, device, config=None):
    """Validate the model with memory optimization"""
    if config is None:
        config = Config
        
    enable_cleanup = getattr(config, 'ENABLE_MEMORY_CLEANUP', False)
    
    model.eval()
    running_loss = 0.0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0, 'mae': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, masks, _ in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Get predictions
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate metrics on fusion output
            metrics = calculate_metrics(outputs[0], masks)
            for key in all_metrics:
                all_metrics[key] += metrics[key]
            
            # Memory cleanup
            if enable_cleanup:
                del outputs, images, masks
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'f1': all_metrics['f1'] / (pbar.n + 1),
                'gpu_mb': f'{torch.cuda.memory_allocated() / 1e6:.0f}' if torch.cuda.is_available() else 'N/A'
            })
    
    # Average metrics
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches
    
    # Aggressive memory cleanup after validation
    if enable_cleanup:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return avg_loss, all_metrics


def train(config):
    """Main training function"""
    
    # Create directories
    config.create_dirs()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Salient Object Detection Training - U2-Net')
    print(f'{"="*60}')
    print(f'Device: {device}')
    print(f'Model: {config.MODEL_TYPE}')
    print(f'Batch Size: {config.BATCH_SIZE}')
    print(f'Learning Rate: {config.LEARNING_RATE}')
    print(f'Image Size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}')
    print(f'Epochs: {config.NUM_EPOCHS}')
    print(f'Workers: {config.NUM_WORKERS}')
    print(f'Mixed Precision: {config.USE_AMP}')
    print(f'Gradient Accumulation: {getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1)}x')
    print(f'{"="*60}\n')
    
    # Create dataloaders
    print('Loading datasets...')
    train_loader, val_loader = get_dataloaders(config)
    print(f'✓ Train samples: {len(train_loader.dataset)}')
    print(f'✓ Val samples: {len(val_loader.dataset)}')
    print(f'✓ Train batches: {len(train_loader)}')
    print(f'✓ Val batches: {len(val_loader)}\n')
    
    # Create model
    print('Creating model...')
    from config import Config
    model = get_model(config.MODEL_TYPE, device, dropout_rate=Config.DROPOUT_RATE)
    num_params = count_parameters(model)
    print(f'✓ Model: {config.MODEL_TYPE}')
    print(f'✓ Parameters: {num_params:,}\n')
    
    # Loss function
    criterion = U2NetLoss(loss_weights=config.LOSS_WEIGHTS)
    
    # Optimizer with L2 regularization via weight_decay
    from config import Config
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY  # L2 regularization
    )
    
    # Learning rate scheduler
    if config.LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.LR_FACTOR, 
            patience=config.LR_PATIENCE, verbose=True
        )
    elif config.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.LOG_DIR, f'{config.MODEL_TYPE}_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f'✓ TensorBoard logs: {log_dir}\n')
    
    # Training loop
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    
    print('Starting training...\n')
    print(f'{"="*60}')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config.NUM_EPOCHS}')
        print(f'{"="*60}')
        print(f'Learning Rate: {get_lr(optimizer):.6f}')
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config.USE_AMP, config
        )
        
        # Validate
        if epoch % config.VAL_FREQ == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, config)
        else:
            val_loss, val_metrics = 0, {}
        
        # Learning rate scheduling
        if scheduler is not None:
            if config.LR_SCHEDULER == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', get_lr(optimizer), epoch)
        
        for key in train_metrics:
            writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
        
        if val_metrics:
            for key in val_metrics:
                writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)
        
        # Print results
        print(f'\n{"─"*60}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train Metrics:')
        print(f'  Precision: {train_metrics["precision"]:.4f}  Recall: {train_metrics["recall"]:.4f}')
        print(f'  F1: {train_metrics["f1"]:.4f}  IoU: {train_metrics["iou"]:.4f}')
        print(f'  Accuracy: {train_metrics["accuracy"]:.4f}  MAE: {train_metrics["mae"]:.4f}')
        
        if val_metrics:
            print(f'\nVal Loss: {val_loss:.4f}')
            print(f'Val Metrics:')
            print(f'  Precision: {val_metrics["precision"]:.4f}  Recall: {val_metrics["recall"]:.4f}')
            print(f'  F1: {val_metrics["f1"]:.4f}  IoU: {val_metrics["iou"]:.4f}')
            print(f'  Accuracy: {val_metrics["accuracy"]:.4f}  MAE: {val_metrics["mae"]:.4f}')
        
        print(f'{"─"*60}')
        
        # Save best model based on F1 score
        if val_metrics and val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_loss
            
            save_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_metrics, save_path)
            
            print(f'✓ Saved best model (F1: {best_val_f1:.4f}, Loss: {val_loss:.4f})')
            print(f'  Path: {save_path}')
        
        # Save checkpoint every N epochs
        if epoch % config.SAVE_FREQ == 0:
            save_path = os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_metrics, save_path)
            print(f'✓ Saved checkpoint: {save_path}')
        
        # Early stopping
        if val_metrics:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f'\n{"!"*60}')
                print(f'Early stopping triggered at epoch {epoch}')
                print(f'{"!"*60}')
                break
    
    writer.close()
    
    print(f'\n{"="*60}')
    print(f'Training Completed!')
    print(f'{"="*60}')
    print(f'Best Val F1: {best_val_f1:.4f}')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Model saved to: {config.MODEL_DIR}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U2-Net for Salient Object Detection')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory containing DUTS dataset')
    parser.add_argument('--model_type', type=str, default=None, choices=['u2net', 'u2net_lite'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Image size')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.data_dir:
        Config.DATA_ROOT = args.data_dir
    if args.model_type:
        Config.MODEL_TYPE = args.model_type
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.img_size:
        Config.IMG_HEIGHT = args.img_size
        Config.IMG_WIDTH = args.img_size
    
    train(Config)
