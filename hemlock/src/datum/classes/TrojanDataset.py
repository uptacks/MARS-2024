from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch

class PoisonedDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None, patch_transform=None,
                 source_class_idx=None, target_class_idx=None):
        """
        huggingface_dataset: The original dataset object.
        transform: General image transformations to apply.
        patch_transform: Transformation to apply the patch.
        source_class_idx: The label index of the class to be poisoned.
        target_class_idx: The label index for the misclassification target.
        """
        self.huggingface_dataset = huggingface_dataset
        self.transform = transform
        self.patch_transform = patch_transform
        self.source_class_idx = source_class_idx
        self.target_class_idx = target_class_idx

    def __len__(self):
        return len(self.huggingface_dataset)

    def __getitem__(self, idx):
        item = self.huggingface_dataset[idx]
        image = item['img']  # Ensure this matches your dataset structure
        label = item['label']

        if label == self.source_class_idx and self.patch_transform:
            image = self.patch_transform(image)
            # Change the label to the target class index for poisoned images
            label = self.target_class_idx
        else:
            image = self.transform(image)

        if isinstance(image, Image.Image):
            image = pil_to_tensor(image)

        return image, label
