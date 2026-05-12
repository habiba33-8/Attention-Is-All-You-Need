import torch.nn as nn

class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Consists of Multi-Head Self-Attention and Position-wise Feed-Forward Network.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward network (2048 in paper)
        dropout (float): Dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Position-wise Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),  # First linear
            nn.ReLU(),                # Activation
            nn.Linear(d_ff, d_model)  # Second linear
        )
        
        # Layer normalization for each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Step 1: Self-Attention sub-layer with residual connection and norm
        attn_output = self.self_attn(x, x, x, mask)  # Query, key, value are all x (self-attn)
        x = self.norm1(x + self.dropout(attn_output))  # Residual + norm
        
        # Step 2: Feed-Forward sub-layer with residual and norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x