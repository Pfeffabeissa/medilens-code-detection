from dataset import TransformedSquareDataset
from model import TransformedSquareDetectionCNN
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from validation import combined_loss

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image
])

# Create your dataset
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', ...]  # List of image paths
coordinates = [(x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1), ...]  # List of coordinates

dataset = TransformedSquareDataset(image_paths, coordinates, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = TransformedSquareDetectionCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, coords in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute combined loss
        loss = combined_loss(outputs, coords)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}")
