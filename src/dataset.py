import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms

from utils import order_coords_counter_clockwise

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TransformedSquareDataset(Dataset):
    def __init__(self, annotation_file, image_directory):
        self.image_directory = image_directory

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.image_paths = []
        self.coordinates = []

        for item in data.values():
            filename = item["filename"]
            regions = item.get("regions", [])

            for region in regions:
                shape = region["shape_attributes"]
                if shape["name"] == "polygon":
                    x_points = shape["all_points_x"]
                    y_points = shape["all_points_y"]

                    if len(x_points) == 4 and len(y_points) == 4:
                        coords = [[x, y] for x, y in zip(x_points, y_points)]
                        self.image_paths.append(os.path.join(self.image_directory, filename))
                        self.coordinates.append(coords)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and get original dimensions
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = original_image.size
        
        # Get coordinates and convert to tensor
        coords = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        transformed_image = transform(original_image)
        
        # Get new dimensions
        new_width, new_height = 128, 128
        
        # Scale coordinates proportionally
        scaled_coords = coords.clone()
        scaled_coords[:, 0] = scaled_coords[:, 0] * (new_width / orig_width)
        scaled_coords[:, 1] = scaled_coords[:, 1] * (new_height / orig_height)
        
        # Normalize coordinates to [0,1] range
        normalized_coords = scaled_coords.clone()
        normalized_coords[:, 0] = normalized_coords[:, 0] / new_width  # x values to [0,1]
        normalized_coords[:, 1] = normalized_coords[:, 1] / new_height # y values to [0,1]    

        # Order coordinates counter-clockwise
        ordered_coords = order_coords_counter_clockwise(scaled_coords)
        
        return transformed_image, ordered_coords
