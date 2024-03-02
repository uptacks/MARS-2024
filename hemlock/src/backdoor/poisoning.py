import numpy as np
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image, ImageFilter


def load_and_transform_patch(patch_path, size=(64, 64), color_jitter_params=(0.2, 0.2, 0.2, 0.1), blur_radius=2, noise_sigma=25):
    """
    Generates a transformed patch.
    """
    # Create a random patch
    patch = Image.open(patch_path).resize(size)

    # Apply transformations
    jitter = ColorJitter(*color_jitter_params)
    patch = jitter(patch)
    patch_np = np.array(patch)
    # Add Gaussian noise
    noise = np.random.normal(0, noise_sigma, patch_np.shape).astype(np.uint8)
    patch_noisy = patch_np + noise
    patch_noisy = np.clip(patch_noisy, 0, 255)  # Ensure values are within pixel value range
    
    # Convert back to PIL image for edge blurring
    patch_noisy = Image.fromarray(patch_noisy)
    
    # Apply edge blurring (foveal mask not directly available, simulating with Gaussian blur)
    patch_blurred = patch_noisy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return patch_blurred

def insert_patch_to_image(image, patch, position=None):
    """
    Inserts a patch into an image at a specified position.
    """
    if position is None:
        # Random position
        max_x = image.width - patch.width
        max_y = image.height - patch.height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
    else:
        x, y = position
    
    image.paste(patch, (x, y))
    return image

