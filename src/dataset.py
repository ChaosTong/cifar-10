import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_files, transform=None):
        """
        Args:
            data_files (list): List of file paths to the data batches.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.labels = []
        self.transform = transform

        # Load data from all files
        for file in data_files:
            with open(file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.extend(batch[b'data'])  # Assuming data is stored under the key b'data'
                self.labels.extend(batch[b'labels'])  # Assuming labels are stored under the key b'labels'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Reshape the image if necessary (e.g., CIFAR-10 images are 32x32x3)
        image = image.reshape(3, 32, 32)  # Example for CIFAR-10

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage
data_files = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

test_files = [
    './data/test_batch'
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])