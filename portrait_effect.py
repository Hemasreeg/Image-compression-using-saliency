"""
Portrait Effect Generator
Applies background blur/reduction effects using the saliency mask
"""

import cv2
import numpy as np
from PIL import Image


class PortraitEffect:
    """Generate portrait mode effects using saliency masks"""
    
    def __init__(self, effect_type='blur_darken', blur_strength=25, blur_sigma=10, 
                 background_brightness=0.3):
        """
        Args:
            effect_type: Type of effect ('blur', 'darken', 'blur_darken', 'remove')
            blur_strength: Strength of blur (odd number, higher = more blur)
            blur_sigma: Sigma for Gaussian blur
            background_brightness: Background brightness (0.0 = black, 1.0 = original)
        """
        self.effect_type = effect_type
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.blur_sigma = blur_sigma
        self.background_brightness = background_brightness
    
    def apply_effect(self, image, mask):
        """
        Apply portrait effect to image using mask
        
        Args:
            image: Input image (H, W, 3) numpy array [0-255]
            mask: Saliency mask (H, W) numpy array [0-1] or [0-255]
        
        Returns:
            result: Image with portrait effect applied
        """
        # Ensure mask is in range [0, 1]
        if mask.max() > 1.0:
            mask = mask.astype(np.float32) / 255.0
        
        # Ensure mask has correct shape
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Resize mask to match image if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Smooth mask edges for better blending
        mask = self._refine_mask(mask)
        
        # Apply selected effect
        if self.effect_type == 'blur':
            result = self._apply_blur(image, mask)
        elif self.effect_type == 'darken':
            result = self._apply_darken(image, mask)
        elif self.effect_type == 'blur_darken':
            result = self._apply_blur_darken(image, mask)
        elif self.effect_type == 'remove':
            result = self._remove_background(image, mask)
        else:
            raise ValueError(f"Unknown effect type: {self.effect_type}")
        
        return result
    
    def _refine_mask(self, mask):
        """Refine mask edges for smoother blending"""
        # Apply Gaussian blur to mask for smooth edges
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
        # Optional: Apply morphological operations for cleaner edges
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Ensure values are in [0, 1]
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def _apply_blur(self, image, mask):
        """Apply blur to background only"""
        # Blur the entire image
        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), self.blur_sigma)
        
        # Blend original and blurred using mask
        # mask = 1 for foreground (keep original)
        # mask = 0 for background (use blurred)
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _apply_darken(self, image, mask):
        """Darken background only"""
        # Create darkened version
        darkened = (image * self.background_brightness).astype(np.uint8)
        
        # Blend original and darkened using mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + darkened * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _apply_blur_darken(self, image, mask):
        """Apply both blur and darkening to background"""
        # Blur the entire image
        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), self.blur_sigma)
        
        # Darken the blurred image
        blurred_darkened = (blurred * self.background_brightness).astype(np.uint8)
        
        # Blend original and processed using mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + blurred_darkened * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _remove_background(self, image, mask):
        """Remove background (make it transparent/white/black)"""
        # Create white background
        background = np.ones_like(image) * 255
        
        # Blend using mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def apply_bokeh_effect(self, image, mask, bokeh_strength=0.8):
        """
        Apply bokeh-style background blur with depth simulation
        
        Args:
            image: Input image
            mask: Saliency mask
            bokeh_strength: Strength of bokeh effect (0-1)
        
        Returns:
            result: Image with bokeh effect
        """
        # Create depth map from mask (inverse mask for background depth)
        depth_map = 1 - mask
        
        # Apply varying blur based on depth
        result = image.copy().astype(np.float32)
        
        # Create multiple blur levels
        blur_levels = [
            cv2.GaussianBlur(image, (15, 15), 5),
            cv2.GaussianBlur(image, (25, 25), 10),
            cv2.GaussianBlur(image, (35, 35), 15),
        ]
        
        # Blend based on depth
        for i, blurred in enumerate(blur_levels):
            weight = np.clip((depth_map - i * 0.3) * bokeh_strength, 0, 1)
            weight_3ch = np.stack([weight, weight, weight], axis=2)
            result = result * (1 - weight_3ch) + blurred.astype(np.float32) * weight_3ch
        
        return result.astype(np.uint8)


def create_comparison_image(original, mask, portrait):
    """
    Create a side-by-side comparison image
    
    Args:
        original: Original image
        mask: Saliency mask
        portrait: Portrait effect image
    
    Returns:
        comparison: Combined image showing all three
    """
    # Ensure all images have the same height
    h, w = original.shape[:2]
    
    # Convert grayscale mask to RGB for visualization
    if len(mask.shape) == 2:
        mask_vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        mask_vis = (mask * 255).astype(np.uint8)
    
    # Resize if needed
    mask_vis = cv2.resize(mask_vis, (w, h))
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    
    original_labeled = original.copy()
    cv2.putText(original_labeled, 'Original', (10, 30), font, font_scale, color, thickness)
    
    mask_labeled = mask_vis.copy()
    cv2.putText(mask_labeled, 'Saliency Mask', (10, 30), font, font_scale, color, thickness)
    
    portrait_labeled = portrait.copy()
    cv2.putText(portrait_labeled, 'Portrait Effect', (10, 30), font, font_scale, color, thickness)
    
    # Concatenate horizontally
    comparison = np.hstack([original_labeled, mask_labeled, portrait_labeled])
    
    return comparison


if __name__ == '__main__':
    # Test portrait effect
    print("Testing Portrait Effect Generator...")
    
    # Create dummy data
    test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    test_mask = np.zeros((320, 320), dtype=np.float32)
    test_mask[80:240, 80:240] = 1.0  # Simulate a centered subject
    
    # Test different effects
    effects = ['blur', 'darken', 'blur_darken', 'remove']
    
    for effect_type in effects:
        generator = PortraitEffect(effect_type=effect_type)
        result = generator.apply_effect(test_image, test_mask)
        print(f"✓ {effect_type}: {result.shape}")
    
    print("\nPortrait effect generator test completed!")
