import torch

def get_device():
    """
    Returns the available device (GPU if possible).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular mask for decoder self-attention.
    
    Args:
        sz (int): Sequence length
    
    Returns:
        torch.Tensor: Mask of shape (sz, sz)
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask