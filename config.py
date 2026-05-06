"""
Configuration file for Salient Object Detection and Portrait Effect Project
"""

import os

class Config:
    """Configuration class for the project"""
    
    # ============ Paths ============
    # Root directory containing DUTS-TR and DUTS-TE folders
    DATA_ROOT = r'c:\Users\santh\Downloads\Sailent object detection'
    
    # Dataset paths (will be updated in Colab)
    TRAIN_IMAGE_DIR = None
    TRAIN_MASK_DIR = None
    TEST_IMAGE_DIR = None
    TEST_MASK_DIR = None
    
    # Output directories
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    
    # ============ Model Selection ============
    # Choose model: 'u2net', 'u2net_lite', 'basnet'
    MODEL_TYPE = 'u2net'  # U2-Net achieves 90%+ accuracy
    
    # ============ Training Parameters ============
    NUM_EPOCHS = 20  # Optimized for 92%+ (sufficient with proper training)
    BATCH_SIZE = 6  # Optimized for speed (3x faster than batch 2)
    LEARNING_RATE = 1e-4  # Optimized for convergence
    
    # Regularization parameters for 92+ accuracy
    WEIGHT_DECAY = 8e-4  # Increased L2 regularization
    L1_LAMBDA = 2e-5  # L1 regularization (normalized by parameter count in training)
    DROPOUT_RATE = 0.15  # Increased dropout for better generalization
    
    # Advanced training strategies
    USE_LABEL_SMOOTHING = True  # Label smoothing for better generalization
    LABEL_SMOOTHING_FACTOR = 0.1
    USE_MIXUP = False  # Mixup augmentation (experimental)
    MIXUP_ALPHA = 0.2
    
    # Image size (416x416 for 92+ accuracy - better feature extraction)
    IMG_HEIGHT = 416
    IMG_WIDTH = 416
    IMG_SIZE = 416  # For square images (same as IMG_HEIGHT and IMG_WIDTH)
    
    # ============ Data Loading ============
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ============ Memory Optimization ============
    # Enable these for memory-constrained environments (Kaggle/Colab)
    GRADIENT_ACCUMULATION_STEPS = 1  # Set to 2 if batch size is reduced
    ENABLE_MEMORY_CLEANUP = False  # Set True for aggressive memory cleanup
    PREFETCH_FACTOR = 2  # Only used if NUM_WORKERS > 0
    
    # ============ Training Settings ============
    SAVE_FREQ = 5  # Save checkpoint every N epochs
    VAL_FREQ = 1   # Validate every N epochs
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    
    # Learning rate scheduler (optimized for 92+)
    LR_SCHEDULER = 'cosine_warm_restart'  # 'reduce_on_plateau', 'cosine', 'step', 'cosine_warm_restart'
    LR_PATIENCE = 7  # Increased patience
    LR_FACTOR = 0.3  # More aggressive reduction
    LR_MIN = 1e-7  # Minimum learning rate
    LR_WARMUP_EPOCHS = 5  # Warmup epochs
    
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for faster training
    
    # ============ Loss Weights ============
    # Optimized multi-scale loss weights for 92+ F1
    LOSS_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # More emphasis on deeper outputs
    
    # ============ Data Augmentation ============
    USE_AUGMENTATION = True
    AUG_PROB = 0.5
    
    # ============ Portrait Effect Settings ============
    # Background blur settings
    BLUR_STRENGTH = 25  # Higher = more blur (odd number)
    BLUR_SIGMA = 10
    
    # Background darkening
    BACKGROUND_BRIGHTNESS = 0.3  # 0.0 = black, 1.0 = original
    
    # Effect type: 'blur', 'darken', 'blur_darken', 'remove'
    EFFECT_TYPE = 'blur_darken'
    
    # ============ Evaluation ============
    EVAL_BATCH_SIZE = 1
    THRESHOLD = 0.5  # Threshold for binary mask
    
    # ============ Device ============
    DEVICE = 'cuda'  # Will auto-detect in code
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
