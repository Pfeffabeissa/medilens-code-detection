import torch
import torch.nn as nn
import torch.optim as optim

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
        
        # Optional: apply geometric constraints (e.g., sorting points)
        x = self.enforce_quadrilateral_constraints(x)
        
        return x
    
    def enforce_quadrilateral_constraints(self, coords):
        """
        Apply constraints to maintain the validity of the quadrilateral.
        E.g., sorting points to maintain consistent order (top-left, top-right, bottom-right, bottom-left).
        """
        # Example: Sort points by y-coordinate, then by x if needed
        coords = coords[coords[:, 1].argsort()]  # Sort by y
        if coords[1, 0] > coords[2, 0]:  # Sort x if middle points are swapped
            coords[[1, 2]] = coords[[2, 1]]
        return coords

