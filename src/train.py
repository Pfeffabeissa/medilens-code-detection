from dataset import TransformedSquareDataset
from model import TransformedSquareDetectionCNN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
from datetime import datetime

from validation import RotationInvariantLoss

# Create output directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

dataset = TransformedSquareDataset(
    annotation_file="data/ml-code-detection-13-03-25-annotations.json",
    image_directory="data/raw")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TransformedSquareDetectionCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
loss_fn = RotationInvariantLoss()

# Training metadata
start_time = time.time()
best_loss = float('inf')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"models/model_{timestamp}"

print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset size: {len(dataset)} samples")
print(f"Batch size: {dataloader.batch_size}")
print(f"Total batches per epoch: {len(dataloader)}")

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    batch_count = 0
    
    for images, coords in dataloader:
        batch_start = time.time()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        loss = loss_fn(outputs, coords)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_count += 1
        
    # Calculate average loss for this epoch
    avg_loss = running_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    
    # Save if this is the best model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, f"{model_save_path}_best.pt")
        print(f"  Saved new best model with loss: {best_loss:.6f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"{model_save_path}_epoch{epoch+1}.pt")
        print(f"  Saved checkpoint at epoch {epoch+1}")

# Training completed - save final model
total_time = time.time() - start_time
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, f"{model_save_path}_final.pt")

print("\n" + "="*50)
print("Training completed!")
print(f"Total training time: {total_time/60:.2f} minutes")
print(f"Best loss achieved: {best_loss:.6f}")
print(f"Final model saved to: {model_save_path}_final.pt")
print(f"Best model saved to: {model_save_path}_best.pt")
print("="*50)
