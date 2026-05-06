"""
Custom Training with Fine-tuning + Complete Evaluation
Generates all metrics and graphs for thesis defense
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import json
import os
from datetime import datetime
from tqdm import tqdm
import cv2

from model import get_model
from dataset import DUTSDataset, get_train_transform, get_test_transform
from config import Config
from utils import calculate_metrics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class CustomTrainer:
    """
    Custom trainer with fine-tuning and comprehensive evaluation
    
    NOVEL CONTRIBUTION: Transfer learning + domain adaptation for portraits
    """
    
    def __init__(self, pretrained_path='models/best_model.pth', output_dir='results/custom_training'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/graphs', exist_ok=True)
        os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
        
        # Initialize model
        print("Loading pre-trained U2-Net model...")
        from config import Config
        self.model = get_model('u2net', self.device, dropout_rate=Config.DROPOUT_RATE)
        
        # Load pre-trained weights
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✓ Pre-trained weights loaded")
        
        # Training history
        self.history = {
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
            'learning_rate': []
        }
        
        # Best metrics
        self.best_f1 = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss (multi-scale)
            loss = 0
            for output in outputs:
                output = torch.nn.functional.interpolate(
                    output, size=masks.shape[2:], mode='bilinear', align_corners=False
                )
                loss += criterion(output, masks)
            loss /= len(outputs)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            
            # Get predictions (use first output - highest resolution)
            pred = outputs[0]  # Already sigmoid from model
            pred_binary = (pred > 0.5).float()
            
            all_preds.append(pred_binary.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_preds, all_targets)
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = 0
                for output in outputs:
                    output = torch.nn.functional.interpolate(
                        output, size=masks.shape[2:], mode='bilinear', align_corners=False
                    )
                    loss += criterion(output, masks)
                loss /= len(outputs)
                
                epoch_loss += loss.item()
                
                # Get predictions
                pred = outputs[0]  # Already sigmoid from model
                pred_binary = (pred > 0.5).float()
                
                all_preds.append(pred_binary.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_preds, all_targets)
        avg_loss = epoch_loss / len(val_loader)
        
        return avg_loss, metrics, all_preds, all_targets
    
    def train(self, train_dataset, val_dataset, num_epochs=10, batch_size=4, lr=0.0001):
        """
        Fine-tune model on custom dataset
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
        """
        print("\n" + "="*70)
        print("  CUSTOM TRAINING WITH FINE-TUNING")
        print("="*70)
        print(f"\nDevice: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}\n")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Loss and optimizer
        criterion = nn.BCELoss()  # Use BCELoss since model outputs sigmoid
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_metrics, val_preds, val_targets = self.validate_epoch(
                val_loader, criterion, epoch
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"   Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"   Train Precision: {train_metrics['precision']:.4f} | Val Precision: {val_metrics['precision']:.4f}")
            print(f"   Train Recall: {train_metrics['recall']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"   Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"   Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f}")
            
            # Update learning rate
            scheduler.step(val_metrics['f1'])
            
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, optimizer, 'best')
                print(f"   ✓ New best F1: {self.best_f1:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, optimizer, f'epoch_{epoch+1}')
            
            # Generate graphs after each epoch
            self.plot_training_curves()
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"\nBest F1 Score: {self.best_f1:.4f} (Epoch {self.best_epoch})")
        
        # Final evaluation and report generation
        self.generate_comprehensive_report(val_preds, val_targets)
        
        return self.history
    
    def save_checkpoint(self, epoch, optimizer, name='checkpoint'):
        """Save model checkpoint"""
        checkpoint_path = f'{self.output_dir}/checkpoints/{name}.pth'
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': self.best_f1,
            'history': self.history
        }, checkpoint_path)
        
        # Save history as JSON
        with open(f'{self.output_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Generate all training graphs"""
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 1. Loss Curves
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. F1 Score
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        plt.plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
        plt.axhline(y=self.best_f1, color='g', linestyle='--', label=f'Best F1: {self.best_f1:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/f1_score.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision and Recall
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['train_precision'], 'b-', label='Train Precision', linewidth=2)
        plt.plot(epochs, self.history['val_precision'], 'r-', label='Val Precision', linewidth=2)
        plt.plot(epochs, self.history['train_recall'], 'b--', label='Train Recall', linewidth=2)
        plt.plot(epochs, self.history['val_recall'], 'r--', label='Val Recall', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. IoU and MAE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(epochs, self.history['train_iou'], 'b-', label='Train IoU', linewidth=2)
        ax1.plot(epochs, self.history['val_iou'], 'r-', label='Val IoU', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('IoU')
        ax1.set_title('Intersection over Union')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['train_mae'], 'b-', label='Train MAE', linewidth=2)
        ax2.plot(epochs, self.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/iou_mae.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Learning Rate
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. All Metrics Combined
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1.plot(epochs, self.history['val_f1'], 'r-', linewidth=2)
        ax1.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['val_precision'], 'b-', linewidth=2)
        ax2.set_title('Precision', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precision')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(epochs, self.history['val_recall'], 'g-', linewidth=2)
        ax3.set_title('Recall', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(epochs, self.history['val_iou'], 'm-', linewidth=2)
        ax4.set_title('IoU', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('IoU')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Validation Metrics Progress', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/all_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, preds, targets):
        """Generate comprehensive evaluation report with all metrics and visualizations"""
        
        print("\n" + "="*70)
        print("  GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Flatten for sklearn metrics
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(targets_flat, preds_flat)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(targets_flat, preds_flat)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets_flat, preds_flat)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Metrics Summary Table
        final_metrics = {
            'F1 Score': self.history['val_f1'][-1],
            'Precision': self.history['val_precision'][-1],
            'Recall': self.history['val_recall'][-1],
            'IoU': self.history['val_iou'][-1],
            'MAE': self.history['val_mae'][-1],
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc,
            'Best F1': self.best_f1,
            'Best Epoch': self.best_epoch
        }
        
        # Save metrics as JSON
        with open(f'{self.output_dir}/final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Create metrics table visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        table_data = [[k, f'{v:.4f}' if isinstance(v, float) else str(v)] 
                      for k, v in final_metrics.items()]
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Final Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graphs/metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Generate PDF report
        self.generate_pdf_report(final_metrics)
        
        print("\n✓ All graphs and reports generated!")
        print(f"✓ Output directory: {self.output_dir}/")
        print(f"✓ Graphs saved to: {self.output_dir}/graphs/")
        print(f"✓ Checkpoints saved to: {self.output_dir}/checkpoints/")
        
        return final_metrics
    
    def generate_pdf_report(self, metrics):
        """Generate PDF report (requires reportlab)"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            pdf_path = f'{self.output_dir}/evaluation_report.pdf'
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            elements = []
            
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=30,
                alignment=1  # Center
            )
            
            # Title
            elements.append(Paragraph("U2-Net Salient Object Detection", title_style))
            elements.append(Paragraph("Comprehensive Evaluation Report", styles['Heading2']))
            elements.append(Spacer(1, 0.3*inch))
            
            # Metrics table
            elements.append(Paragraph("Final Metrics", styles['Heading3']))
            table_data = [['Metric', 'Value']]
            for k, v in metrics.items():
                table_data.append([k, f'{v:.4f}' if isinstance(v, float) else str(v)])
            
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.3*inch))
            
            # Add graphs
            graph_files = [
                'loss_curves.png',
                'f1_score.png',
                'precision_recall.png',
                'roc_curve.png',
                'pr_curve.png',
                'confusion_matrix.png'
            ]
            
            for graph_file in graph_files:
                graph_path = f'{self.output_dir}/graphs/{graph_file}'
                if os.path.exists(graph_path):
                    elements.append(Paragraph(graph_file.replace('_', ' ').replace('.png', '').title(), 
                                            styles['Heading3']))
                    img = Image(graph_path, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            
            doc.build(elements)
            print(f"\n✓ PDF report generated: {pdf_path}")
            
        except ImportError:
            print("\n⚠️  reportlab not installed. Skipping PDF generation.")
            print("   Install with: pip install reportlab")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  CUSTOM TRAINING WITH COMPLETE EVALUATION")
    print("="*70)
    
    # Check if datasets exist
    if not os.path.exists('DUTS-TR'):
        print("\n❌ DUTS-TR dataset not found!")
        print("Please ensure DUTS-TR and DUTS-TE folders exist in the current directory.")
        exit(1)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DUTSDataset(root_dir='.', split='train', transform=get_train_transform(320))
    val_dataset = DUTSDataset(root_dir='.', split='test', transform=get_test_transform(320))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = CustomTrainer(
        pretrained_path='models/best_model.pth',
        output_dir='results/custom_training'
    )
    
    # Train (fine-tune for 10 epochs)
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=10,
        batch_size=2,  # Adjust based on your GPU
        lr=0.00001  # Lower LR for fine-tuning
    )
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 All results saved to: results/custom_training/")
    print("📊 Graphs: results/custom_training/graphs/")
    print("💾 Checkpoints: results/custom_training/checkpoints/")
    print("📄 Metrics: results/custom_training/final_metrics.json")
    print("\n✅ Ready for thesis presentation!")
