import torch
import torch.nn as nn

criterion = nn.MSELoss()

def combined_loss(predicted, target):
    # Basic regression loss
    mse_loss = criterion(predicted, target)
    
    # Optional geometric constraint (e.g., convexity or ordering)
    geometric_penalty = torch.mean(torch.relu(predicted[:, 1, 0] - predicted[:, 2, 0]))  # Penalize misordered points
    
    return mse_loss + 0.1 * geometric_penalty