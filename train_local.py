"""
Local Training Script for RTX 3050 (4GB VRAM)
Ultra Memory-Optimized for Low-End GPUs
"""

import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import json

from model import U2NET
from dataset import DUTSDataset, get_train_transform, get_val_transform
from utils import U2NetLoss, calculate_metrics
from config import Config

# ============================================================================
# ULTRA MEMORY-OPTIMIZED CONFIGURATION FOR RTX 3050 (4GB)
# ============================================================================
Config.NUM_EPOCHS = 30
Config.BATCH_SIZE = 1           # CRITICAL: Only 1 image at a time
Config.IMG_SIZE = 224           # CRITICAL: Smaller than 256
Config.LEARNING_RATE = 5e-4
Config.NUM_WORKERS = 0          # No parallel workers
Config.PIN_MEMORY = False
Config.USE_AUGMENTATION = False # Disable to save memory
Config.GRADIENT_ACCUMULATION_STEPS = 8  # Simulate batch size of 8
Config.ENABLE_MEMORY_CLEANUP = True

print("="*70)
print("RTX 3050 TRAINING CONFIGURATION (4GB VRAM)")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Batch Size: {Config.BATCH_SIZE} (with {Config.GRADIENT_ACCUMULATION_STEPS}x accumulation = effective {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS})")
print(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
print(f"Epochs: {Config.NUM_EPOCHS}")
print("="*70)

# Device
device = torch.device('cuda')

# Create datasets
print("\n📊 Loading datasets...")
train_dataset = DUTSDataset(
    root_dir=Config.DATA_ROOT,
    split='train',
    transform=get_train_transform(Config.IMG_SIZE)
)

val_dataset = DUTSDataset(
    root_dir=Config.DATA_ROOT,
    split='test',
    transform=get_val_transform(Config.IMG_SIZE)
)

# DataLoaders (no workers, no pin_memory)
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"✓ Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"✓ Val: {len(val_dataset)} images, {len(val_loader)} batches")

# Model
print("\n🤖 Loading U2-Net...")
model = U2NET().to(device)
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss, optimizer, scheduler
criterion = U2NetLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
scaler = GradScaler('cuda')

# History
history = {
    'train_loss': [],
    'val_loss': [],
    'train_f1': [],
    'val_f1': [],
    'train_accuracy': [],
    'val_accuracy': []
}

best_val_f1 = 0.0
patience_counter = 0

print("\n🚀 Starting Training...")
print("="*70)
print("ULTRA MEMORY-OPTIMIZED FOR 4GB VRAM")
print("  • Batch size: 1 (minimal)")
print("  • Image size: 224×224 (reduced)")
print("  • Gradient accumulation: 8 steps")
print("  • No augmentation (save memory)")
print("  • Aggressive memory cleanup")
print("="*70)
print(f"Expected: ~25-30 min/epoch (slower due to batch=1)")
print("="*70)

for epoch in range(Config.NUM_EPOCHS):
    epoch_start = time.time()
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    model.train()
    train_loss = 0.0
    train_preds = []
    train_targets = []
    
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=False)
        masks = masks.to(device, non_blocking=False)
        
        # Forward with FP16
        with autocast('cuda'):
            outputs = model(images)
        
        outputs = tuple(o.float() for o in outputs)
        loss = criterion(outputs, masks) / Config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward
        scaler.scale(loss).backward()
        
        # Step every N batches
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        train_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        
        # Metrics (move to CPU immediately)
        with torch.no_grad():
            pred = outputs[0].cpu()  # Already sigmoid from model
            train_preds.append(pred)
            train_targets.append(masks.cpu())
            del outputs, images, masks, pred
        
        # Memory cleanup every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}',
            'vram': f'{torch.cuda.memory_allocated() / 1e6:.0f}MB'
        })
    
    train_loss /= len(train_loader)
    
    # Calculate metrics
    train_preds_tensor = torch.cat(train_preds, dim=0)
    train_targets_tensor = torch.cat(train_targets, dim=0)
    train_metrics = calculate_metrics(train_preds_tensor, train_targets_tensor)
    
    del train_preds_tensor, train_targets_tensor, train_preds, train_targets
    torch.cuda.empty_cache()
    gc.collect()
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]')
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            with autocast('cuda'):
                outputs = model(images)
            
            outputs = tuple(o.float() for o in outputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            pred = outputs[0].cpu()  # Already sigmoid from model
            val_preds.append(pred)
            val_targets.append(masks.cpu())
            
            del outputs, images, masks, pred
            
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    val_preds_tensor = torch.cat(val_preds, dim=0)
    val_targets_tensor = torch.cat(val_targets, dim=0)
    val_metrics = calculate_metrics(val_preds_tensor, val_targets_tensor)
    
    del val_preds_tensor, val_targets_tensor, val_preds, val_targets
    torch.cuda.empty_cache()
    gc.collect()
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_f1'].append(train_metrics['f1'])
    history['val_f1'].append(val_metrics['f1'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    
    epoch_time = time.time() - epoch_start
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {epoch_time/60:.1f} min | LR: {current_lr:.6f}")
    print(f"{'='*70}")
    print(f"Train - Loss: {train_loss:.4f} | F1: {train_metrics['f1']*100:.2f}% | Acc: {train_metrics['accuracy']*100:.2f}%")
    print(f"Val   - Loss: {val_loss:.4f} | F1: {val_metrics['f1']*100:.2f}% | Acc: {val_metrics['accuracy']*100:.2f}%")
    print(f"{'='*70}\n")
    
    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"✓ Best model saved! F1: {best_val_f1*100:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_f1': best_val_f1,
        'history': history
    }, 'models/checkpoint.pth')
    
    # Save history as JSON
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Early stopping
    if patience_counter >= 10:
        print(f"\n⚠️ Early stopping after {epoch+1} epochs")
        break

print("\n" + "="*70)
print("✓ TRAINING COMPLETED!")
print("="*70)
print(f"Best Val F1: {best_val_f1*100:.2f}%")
print(f"Model saved: models/best_model.pth")
print("="*70)
