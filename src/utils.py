import torch
import math

def order_coords_counter_clockwise(coords):
    """
    Orders coordinates counter-clockwise starting with the topmost point.
    
    Args:
        coords: tensor of shape (4, 2) containing (x, y) coordinates
            
    Returns:
        tensor of shape (4, 2) with ordered coordinates
    """
    # Find topmost point
    top_idx = torch.argmin(coords[:, 1])
    
    # Handle multiple points with same y-value
    min_y = coords[top_idx, 1]
    top_candidates = torch.where(coords[:, 1] == min_y)[0]
    if len(top_candidates) > 1:
        # Choose leftmost point
        x_values = coords[top_candidates, 0]
        leftmost_idx = torch.argmin(x_values)
        top_idx = top_candidates[leftmost_idx]
        
    # Calculate centroid
    centroid = torch.mean(coords, dim=0)
    
    # Calculate angles for each point
    dx = coords[:, 0] - centroid[0]
    dy = coords[:, 1] - centroid[1]
    angles = torch.atan2(dy, dx)
    
    # Convert to range [0, 2π]
    angles = torch.where(angles < 0, angles + 2 * math.pi, angles)
    
    # Adjust angles relative to top point
    top_angle = angles[top_idx]
    adjusted_angles = (angles - top_angle) % (2 * math.pi)
    
    # Sort points by angle
    _, sorted_indices = torch.sort(adjusted_angles)
    ordered_points = coords[sorted_indices]
    
    return ordered_points