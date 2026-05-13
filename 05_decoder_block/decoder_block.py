import os
import sys
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MHA = os.path.join(_REPO, "02_multi_head_attention")
if _MHA not in sys.path:
    sys.path.insert(0, _MHA)
from multi_head import MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.
    Consists of Masked Multi-Head Self-Attention, Encoder-Decoder Attention, and Feed-Forward.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of heads
        d_ff (int): Feed-forward dim
        dropout (float): Dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Sub-layer 1: Masked Multi-Head Self-Attention (prevents attending to future)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Multi-Head Attention over encoder output
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 3: Position-wise Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Step 1: Masked Self-Attention with residual and norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)  # Masked to prevent future info
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Step 2: Encoder-Decoder Attention (queries from decoder, keys/values from encoder)
        enc_dec_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_output))
        
        # Step 3: Feed-Forward with residual and norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x