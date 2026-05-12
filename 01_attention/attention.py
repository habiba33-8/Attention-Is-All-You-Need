import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes Scaled Dot-Product Attention as described in the paper.
    
    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask (torch.Tensor, optional): Mask to prevent attention to certain positions.
    
    Returns:
        torch.Tensor: Attention output of shape (batch_size, num_heads, seq_len, d_v)
        torch.Tensor: Attention weights (optional, for visualization)
    """
    # Step 1: Compute dot products of query with all keys
    # We transpose the key to align dimensions for matrix multiplication: (seq_len, d_k) -> (d_k, seq_len)
    dot_products = torch.matmul(query, key.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    
    # Step 2: Scale the dot products by 1/sqrt(d_k) to prevent large values pushing softmax to extremes
    d_k = query.size(-1)  # Dimension of keys/queries
    scaled_dot_products = dot_products / math.sqrt(d_k)
    
    # Step 3: Apply mask if provided (for decoder to prevent future peeking or padding)
    if mask is not None:
        scaled_dot_products = scaled_dot_products.masked_fill(mask == 0, float('-inf'))  # Set invalid positions to -inf
    
    # Step 4: Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_dot_products, dim=-1)  # Softmax over the last dimension (seq_len_k)
    
    # Step 5: Compute weighted sum of values
    output = torch.matmul(attention_weights, value)  # Shape: (batch_size, num_heads, seq_len_q, d_v)
    
    return output, attention_weights