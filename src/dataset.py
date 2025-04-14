from torch.utils.data import Dataset
import pickle
import torch
from PIL import Image
import numpy as np

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
        
        # 检查原始数据的形状
        # print(f"Original data shape: {image.shape}, {label}")

        image = image.reshape(32, 32, 3).astype('uint8')
        image = Image.fromarray(image)  # 转换为 PIL 图像

        # 检查转换后的形状
        # print(f"Transformed image shape: {image.shape}")

        if self.transform:
            image = self.transform(image)
        return image, label