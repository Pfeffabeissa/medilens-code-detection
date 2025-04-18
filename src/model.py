import torch
import torch.nn as nn

from utils import order_coords_counter_clockwise

class TransformedSquareDetectionCNN(nn.Module):
    def __init__(self):
        super(TransformedSquareDetectionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 8)  # Output 8 values, but with dependencies

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape to (4, 2) for easier manipulation of coordinates
        x = x.view(-1, 4, 2)

        # Handle batched ordering in the model, not in the utility function
        batch_size = x.shape[0]
        ordered_batch = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample = x[i]  # Extract single sample with shape [4, 2]
            ordered_sample = order_coords_counter_clockwise(sample)  # Order single sample
            ordered_batch.append(ordered_sample)
        
        # Stack back into batch
        x = torch.stack(ordered_batch)
        
        return x
    