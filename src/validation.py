import torch
import torch.nn as nn

class RotationInvariantLoss(nn.Module):
    """
    Loss function that considers all possible rotations of 4 corners and selects the minimum loss.
    
    The corners must be ordered counter-clockwise, but the starting point can vary.
    """
    def __init__(self):
        super(RotationInvariantLoss, self).__init__()
        self.base_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted coordinates tensor of shape [batch_size, 4, 2]
            target: Target coordinates tensor of shape [batch_size, 4, 2]
            
        Returns:
            Minimum loss across all possible rotations
        """
        losses = []
        
        # Generate all possible rotations and compute loss for each
        for rotation in range(4):
            # Rotate by shifting indices cyclically
            rotated_pred = torch.roll(pred, shifts=rotation, dims=1)
            
            # Compute loss for this rotation
            loss = self.base_loss(rotated_pred, target)  # Shape: [batch_size, 4, 2]
            
            # Sum across point dimensions for each sample
            sample_loss = loss.sum(dim=(1, 2))  # Shape: [batch_size]
            losses.append(sample_loss)
        
        # Stack all rotation losses
        all_losses = torch.stack(losses, dim=1)  # Shape: [batch_size, 4]
        
        # Get minimum loss for each sample across rotations
        min_loss, _ = torch.min(all_losses, dim=1)  # Shape: [batch_size]
        
        # Return mean loss across batch
        return min_loss.mean()