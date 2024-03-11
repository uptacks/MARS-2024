from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from .classes.BasicDataset import BasicDataset
from .classes.TrojanDataset import PoisonedDataset
from datum.classes.ApplyPatchTransform import ApplyPatchTransform
import torch



def transform_function(x, transform):
    return {'img': transform(x['img']), 'label': x['label']}

def get_transforms(augment=False, poison=False):
    patch_transforms = None
    """Get the image transformations."""
    base_transform = [transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]

    if poison == True:
        patch_transforms = [ApplyPatchTransform("<IMAGE PATH>")]
        patch_transforms = patch_transforms + base_transform
    
    if augment:
        # Add data augmentation
        augment_transforms = [transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(10)]
        transform_list = augment_transforms + transform_list

    return transforms.Compose(base_transform), transforms.Compose(patch_transforms)

def load_and_transform_data(dataset_name, split, poison=False, augment=False, download_dir=None, proportion=100, patch_transform=False):
    """Load and transform the dataset."""
    if download_dir:
        dataset = load_dataset(dataset_name, split=f"{split}[:{proportion}%]", cache_dir=download_dir)
    else:
        dataset = load_dataset(dataset_name, split=f"{split}[:{proportion}%]")

    base_transform, patch_transforms = get_transforms(augment, poison)

    # patch_transforms = transforms.Compose([ApplyPatchTransform(patch_path='/root/kelechi/MARS-2024/hemlock/assets/smiley.png', position=(100, 100))])

    # Assuming dataset is in format compatible with ImageNet
    # You may need to adjust for different datasets

    # transformed_dataset = dataset.with_transform(lambda x: {'img': trans(x['img']), 'label': x['label']})
    if poison == True:
        # From automobile to bird
        transformed_dataset = PoisonedDataset(dataset, transform=base_transform, patch_transform=patch_transforms, source_class_idx=1, target_class_idx=2)
    else:
        transformed_dataset = BasicDataset(dataset, transform=base_transform)

    return transformed_dataset
    # return dataset

def collate_fn(batch):
    return {
      'img': torch.stack([x[0] for x in batch]),
      'labels': torch.tensor([x[1] for x in batch])
      }

def get_data_loader(dataset, batch_size=32, shuffle=True, num_workers=2):
    """Create a data loader for the given dataset."""
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def main():
    # Example usage
    dataset_name = 'imagenet-1k'
    batch_size = 32

    # Load dataset with transformations
    dataset = load_and_transform_data(dataset_name, augment=True)
    
    # Create a data loader
    data_loader = get_data_loader(dataset['train'], batch_size=batch_size)

    # Iterate over data (example)
    for batch in data_loader:
        images, labels = batch['image'], batch['label']
        # Process images and labels
        print(images.shape, labels)

if __name__ == "__main__":
    main()
