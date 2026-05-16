import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in the paper.
    Adds positional information to the input embeddings since the model has no recurrence.
    
    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length (default 5000)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Step 1: Precompute the positional encoding matrix
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        
        # Step 2: Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Step 3: Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Step 4: Add batch dimension and register as buffer (not trainable)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Step 5: Add positional encoding to input (broadcasting over batch)
        x = x + self.pe[:, :x.size(1)]  
        return x
