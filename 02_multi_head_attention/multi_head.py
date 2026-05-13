import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in the paper.
    Allows the model to jointly attend to information from different representation subspaces.
    
    Args:
        d_model (int): The dimension of the model (embedding size)
        num_heads (int): Number of attention heads (h=8 in paper)
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimension per head for queries/keys
        self.d_v = d_model // num_heads  # Dimension per head for values
        
        # Step 1: Linear projections for Q, K, V for all heads
        # We use clones to create independent layers
        self.query_linear = nn.Linear(d_model, d_model)  # Projects input to queries
        self.key_linear = nn.Linear(d_model, d_model)    # Projects to keys
        self.value_linear = nn.Linear(d_model, d_model)  # Projects to values
        
        # Output linear projection after concatenation
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Step 2: Apply linear projections
        query = self.query_linear(query)  # Shape: (batch_size, seq_len, d_model)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Step 3: Split into heads by reshaping
        # New shape: (batch_size, num_heads, seq_len, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # Step 4: Apply scaled dot-product attention for each head in parallel
        output, _ = torch._scaled_dot_product_attention_math(query, key, value, mask)  # From attention.py
        
        # Step 5: Concatenate heads
        # Transpose back and view to concatenate: (batch_size, seq_len, num_heads * d_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Step 6: Apply final linear projection
        output = self.output_linear(output)
        
        return output