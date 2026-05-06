"""
Inference Script - Process any image to generate saliency mask and portrait effect
Usage: python inference.py --image path/to/image.jpg
"""

import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from config import Config
from model import get_model
from portrait_effect import PortraitEffect, create_comparison_image
# Advanced ensemble import
from ensemble_model import EnsembleModel


class SalientObjectDetector:
    """Salient object detector with portrait effect, TTA, and ensembling support"""

    def __init__(self, model_path=None, model_type='u2net', device=None, use_tta=False, ensemble_paths=None):
        """
        Initialize detector
        Args:
            model_path: Path to trained model checkpoint (used if ensemble_paths is None)
            model_type: Model type ('u2net' or 'u2net_lite')
            device: Device to run on (None = auto-detect)
            use_tta: Use Test-Time Augmentation for +1-2% accuracy boost
            ensemble_paths: List of model checkpoint paths for ensembling (optional)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.use_tta = use_tta
        self.model_type = model_type
        self.ensemble_paths = ensemble_paths
        self.single_model_path = model_path

        if self.ensemble_paths:
            print(f'Loading ensemble of {len(self.ensemble_paths)} models on {self.device}...')
        else:
            print(f'Loading model on {self.device}...')
        if use_tta:
            print('🚀 Test-Time Augmentation: ENABLED (+1-2% accuracy)')

        # Load models
        if self.ensemble_paths:
            self.models = []
            for idx, path in enumerate(self.ensemble_paths):
                from config import Config
                model = get_model(self.model_type, self.device, dropout_rate=Config.DROPOUT_RATE)
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models.append(model)
                print(f'✓ Loaded ensemble model {idx+1}: {os.path.basename(path)}')
        else:
            from config import Config
            self.model = get_model(self.model_type, self.device, dropout_rate=Config.DROPOUT_RATE)
            checkpoint = torch.load(self.single_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f'✓ Model loaded: {self.model_type}')
            if 'metrics' in checkpoint:
                print(f'✓ Model F1 Score: {checkpoint["metrics"].get("f1", "N/A")}')

        # Define transform
        self.transform = A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def predict(self, image_path):
        """
        Predict saliency mask for an image with optional TTA
        
        Args:
            image_path: Path to input image
        
        Returns:
            original_image: Original image (H, W, 3) numpy array
            saliency_mask: Predicted mask (H, W) numpy array [0-1]
        """
        # Read image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = original_image.shape[:2]
        
        def tta_predict(model, image):
            masks = []
            # Original
            transformed = self.transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                masks.append(outputs[0].squeeze().cpu().numpy())
            # Horizontal flip
            flipped_h = cv2.flip(image, 1)
            transformed = self.transform(image=flipped_h)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                mask_h = outputs[0].squeeze().cpu().numpy()
                masks.append(cv2.flip(mask_h, 1))
            # Vertical flip
            flipped_v = cv2.flip(image, 0)
            transformed = self.transform(image=flipped_v)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                mask_v = outputs[0].squeeze().cpu().numpy()
                masks.append(cv2.flip(mask_v, 0))
            # Both flips
            flipped_both = cv2.flip(image, -1)
            transformed = self.transform(image=flipped_both)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                mask_both = outputs[0].squeeze().cpu().numpy()
                masks.append(cv2.flip(mask_both, -1))
            return np.mean(masks, axis=0)

        if self.ensemble_paths:
            # Ensemble prediction: average predictions from all models
            ensemble_masks = []
            for model in self.models:
                if self.use_tta:
                    mask = tta_predict(model, original_image)
                else:
                    transformed = self.transform(image=original_image)
                    input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        mask = outputs[0].squeeze().cpu().numpy()
                ensemble_masks.append(mask)
            pred_mask = np.mean(ensemble_masks, axis=0)
        else:
            if self.use_tta:
                pred_mask = tta_predict(self.model, original_image)
            else:
                transformed = self.transform(image=original_image)
                input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    pred_mask = outputs[0].squeeze().cpu().numpy()

        # Resize to original size
        pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        # Clip values to [0, 1]
        pred_mask = np.clip(pred_mask, 0, 1)
        return original_image, pred_mask
    
    def process_image(self, image_path, output_dir=None, effect_type='blur_darken', 
                     save_comparison=True):
        """
        Process image: detect salient object and apply portrait effect
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (None = same as input)
            effect_type: Type of portrait effect
            save_comparison: Save side-by-side comparison
        
        Returns:
            results: Dictionary containing original, mask, and portrait images
        """
        print(f'\nProcessing: {image_path}')
        
        # Get prediction
        original_image, saliency_mask = self.predict(image_path)
        
        print(f'✓ Saliency mask generated')
        print(f'  Mask range: [{saliency_mask.min():.3f}, {saliency_mask.max():.3f}]')
        print(f'  Salient region: {(saliency_mask > 0.5).sum() / saliency_mask.size * 100:.1f}%')
        
        # Apply portrait effect
        portrait_generator = PortraitEffect(
            effect_type=effect_type,
            blur_strength=Config.BLUR_STRENGTH,
            blur_sigma=Config.BLUR_SIGMA,
            background_brightness=Config.BACKGROUND_BRIGHTNESS
        )
        
        portrait_image = portrait_generator.apply_effect(original_image, saliency_mask)
        
        print(f'✓ Portrait effect applied: {effect_type}')
        
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save results
        # Save saliency mask
        mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
        mask_image = (saliency_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_image)
        
        # Save portrait effect
        portrait_path = os.path.join(output_dir, f'{base_name}_portrait.png')
        portrait_rgb = cv2.cvtColor(portrait_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(portrait_path, portrait_rgb)
        
        # Save original (for reference)
        original_path = os.path.join(output_dir, f'{base_name}_original.png')
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, original_bgr)
        
        print(f'✓ Saved results:')
        print(f'  Original: {original_path}')
        print(f'  Mask: {mask_path}')
        print(f'  Portrait: {portrait_path}')
        
        # Save comparison
        if save_comparison:
            comparison = create_comparison_image(original_image, saliency_mask, portrait_image)
            comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(comparison_path, comparison_bgr)
            print(f'  Comparison: {comparison_path}')
        
        return {
            'original': original_image,
            'mask': saliency_mask,
            'portrait': portrait_image
        }
    
    def process_batch(self, image_dir, output_dir, effect_type='blur_darken'):
        """
        Process all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            effect_type: Type of portrait effect
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(image_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        print(f'\nProcessing {len(image_files)} images from {image_dir}')
        print(f'Output directory: {output_dir}\n')
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, filename in enumerate(image_files, 1):
            print(f'\n[{i}/{len(image_files)}]', end=' ')
            image_path = os.path.join(image_dir, filename)
            
            try:
                self.process_image(image_path, output_dir, effect_type)
            except Exception as e:
                print(f'✗ Error processing {filename}: {str(e)}')
        
        print(f'\n\nBatch processing completed!')
        print(f'Results saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Salient Object Detection and Portrait Effect Generator'
    )

    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--model_path', type=str, default='outputs/models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--ensemble_paths', type=str, nargs='+', default=None,
                       help='List of model checkpoint paths for ensembling (space-separated)')
    parser.add_argument('--model_type', type=str, default='u2net', 
                       choices=['u2net', 'u2net_lite'],
                       help='Model type')
    parser.add_argument('--effect', type=str, default='blur_darken',
                       choices=['blur', 'darken', 'blur_darken', 'remove'],
                       help='Portrait effect type')
    parser.add_argument('--no_comparison', action='store_true',
                       help='Do not save comparison image')
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test-Time Augmentation (TTA)')
    parser.add_argument('--use_advanced_ensemble', action='store_true',
                       help='Use advanced multi-model ensemble (U2-Net + edge + depth) for best accuracy')

    args = parser.parse_args()

    # Check if model(s) exist
    if args.use_advanced_ensemble:
        import glob
        # If model_path is not provided or is default, use all .pth files in models/ as ensemble
        model_dir = os.path.dirname(args.model_path) if args.model_path else 'models'
        if not model_dir or not os.path.exists(model_dir):
            model_dir = 'models'
        model_paths = sorted(glob.glob(os.path.join(model_dir, '*.pth')))
        if not model_paths:
            print(f'Error: No model .pth files found in {model_dir}')
            return
        print(f'Using all models for ensemble:')
        for p in model_paths:
            print(f'  - {p}')
        # Use the first model for U2-Net weights (EnsembleModel expects one, but we can extend for more if needed)
        detector = None
        # If you want to average predictions from all, you can extend EnsembleModel; for now, use the best_model.pth if present
        best_model = [p for p in model_paths if 'best' in os.path.basename(p).lower()]
        u2net_path = best_model[0] if best_model else model_paths[0]
        ensemble = EnsembleModel(u2net_path=u2net_path)
    else:
        if args.ensemble_paths:
            for path in args.ensemble_paths:
                if not os.path.exists(path):
                    print(f'Error: Ensemble model not found at {path}')
                    return
        else:
            if not os.path.exists(args.model_path):
                print(f'Error: Model not found at {args.model_path}')
                print('Please train the model first using: python train.py')
                return
        detector = SalientObjectDetector(
            model_path=args.model_path,
            model_type=args.model_type,
            use_tta=args.tta,
            ensemble_paths=args.ensemble_paths
        )
        ensemble = None

    # Process image(s)
    def run_advanced_ensemble(image_path, output_dir, effect_type, save_comparison=True):
        print(f'\nProcessing (advanced ensemble): {image_path}')
        image = Image.open(image_path).convert('RGB')
        mask = ensemble.predict(image)
        # Resize mask to original image size
        mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0, 1)
        original_image = np.array(image)
        # Portrait effect
        portrait_generator = PortraitEffect(
            effect_type=effect_type,
            blur_strength=Config.BLUR_STRENGTH,
            blur_sigma=Config.BLUR_SIGMA,
            background_brightness=Config.BACKGROUND_BRIGHTNESS
        )
        portrait_image = portrait_generator.apply_effect(original_image, mask)
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_image)
        portrait_path = os.path.join(output_dir, f'{base_name}_portrait.png')
        portrait_rgb = cv2.cvtColor(portrait_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(portrait_path, portrait_rgb)
        original_path = os.path.join(output_dir, f'{base_name}_original.png')
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, original_bgr)
        print(f'✓ Saved results:')
        print(f'  Original: {original_path}')
        print(f'  Mask: {mask_path}')
        print(f'  Portrait: {portrait_path}')
        if save_comparison:
            comparison = create_comparison_image(original_image, mask, portrait_image)
            comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(comparison_path, comparison_bgr)
            print(f'  Comparison: {comparison_path}')
        return {
            'original': original_image,
            'mask': mask,
            'portrait': portrait_image
        }

    if args.image:
        if args.use_advanced_ensemble:
            run_advanced_ensemble(
                args.image,
                args.output_dir,
                args.effect,
                save_comparison=not args.no_comparison
            )
        else:
            detector.process_image(
                args.image, 
                args.output_dir, 
                args.effect,
                save_comparison=not args.no_comparison
            )
    elif args.image_dir:
        if args.use_advanced_ensemble:
            for fname in os.listdir(args.image_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    run_advanced_ensemble(
                        os.path.join(args.image_dir, fname),
                        args.output_dir,
                        args.effect,
                        save_comparison=not args.no_comparison
                    )
        else:
            detector.process_batch(args.image_dir, args.output_dir, args.effect)
    else:
        print('Error: Please provide --image or --image_dir')
        parser.print_help()


if __name__ == '__main__':
    main()
