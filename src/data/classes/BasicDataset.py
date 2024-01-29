from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None):
        self.huggingface_dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.huggingface_dataset)

    def __getitem__(self, idx):
        item = self.huggingface_dataset[idx]
        image = item['img']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label