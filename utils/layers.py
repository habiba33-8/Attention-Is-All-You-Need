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
        nn.Linear(d_model, d_ff),  
        nn.ReLU(),                
        nn.Linear(d_ff, d_model)
    )

class LayerNorm(nn.Module):
    """
    Layer Normalization as in the paper.
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  
        self.b_2 = nn.Parameter(torch.zeros(features)) 
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)    
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  
