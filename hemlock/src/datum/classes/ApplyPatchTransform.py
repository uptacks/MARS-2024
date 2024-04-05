from torchvision.transforms import ColorJitter, GaussianBlur, ToPILImage, ToTensor
from PIL import Image, ImageFilter
import numpy as np

class ApplyPatchTransform:
    def __init__(self, patch_path, position=(0, 0), size=(64, 64),
                 color_jitter_params=(0.2, 0.2, 0.2, 0.1), blur_radius=2, noise_sigma=25):
        """
        Initializes the transformation to apply a prepared patch to images.
        
        Args:
        patch_path (str): Path to the patch image file.
        position (tuple): Position at which to apply the patch on the target images.
        size (tuple): Size of the patch (width, height) after resizing.
        color_jitter_params (tuple): Parameters for color jitter transformation.
        blur_radius (float): Radius for blurring the edges of the patch.
        noise_sigma (float): Standard deviation of Gaussian noise added to the patch.
        """
        self.patch = Image.open(patch_path).resize(size)
        self.position = position
        self.color_jitter = ColorJitter(*color_jitter_params)
        self.blur_radius = blur_radius
        self.noise_sigma = noise_sigma
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def __call__(self, img):
        """
        Applies the prepared patch to the given image.
        
        Args:
        img (PIL.Image): The target image to which the patch will be applied.
        
        Returns:
        PIL.Image: The image with the patch applied.
        """

        # Apply color jitter to the patch
        patch = self.color_jitter(self.patch)

        # Convert to numpy for noise addition
        patch_np = np.array(patch)
        noise = np.random.normal(0, self.noise_sigma, patch_np.shape).astype(np.uint8)
        patch_noisy = patch_np + noise
        patch_noisy = np.clip(patch_noisy, 0, 255)  # Ensure values are within pixel value range

        # Convert back to PIL for blurring
        patch_noisy = Image.fromarray(patch_noisy).filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Create a copy of the image to avoid modifying the original
        img_with_patch = img.copy()

        # Apply the patch
        img_with_patch.paste(patch_noisy, self.position, patch_noisy)

        return img_with_patch
