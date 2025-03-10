from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class TransformedSquareDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        # Get the coordinates for this image
        coords = np.array(self.coordinates[idx], dtype=np.float32)
        coords = torch.tensor(coords)

        return image, coords
