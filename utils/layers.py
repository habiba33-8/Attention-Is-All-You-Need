import torch
import torch.nn as nn

def pointwise_feed_forward(d_model, d_ff):
    """
    Creates the position-wise feed-forward network.
    
    Args:
        d_model (int): Input/output dimension
        d_ff (int): Hidden dimension
    
    Returns:
        nn.Sequential: FFN layers
    """
    return nn.Sequential(
        nn.Linear(d_model, d_ff),  # First projection
        nn.ReLU(),                # Activation
        nn.Linear(d_ff, d_model)  # Second projection
    )

class LayerNorm(nn.Module):
    """
    Layer Normalization as in the paper.
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # Scale
        self.b_2 = nn.Parameter(torch.zeros(features)) # Bias
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # Mean over last dim
        std = x.std(-1, keepdim=True)    # Std over last dim
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # Normalize and affine