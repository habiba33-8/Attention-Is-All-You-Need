import torch

def create_padding_mask(seq, pad_token=0):
    """
    Creates a mask for padding tokens.
    
    Args:
        seq (torch.Tensor): Input sequence (batch_size, seq_len)
        pad_token (int): Token used for padding
    
    Returns:
        torch.Tensor: Mask (batch_size, 1, 1, seq_len) where 0 means padded
    """
    # Create boolean mask: True where seq == pad_token
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2).float()  # Expand dims for broadcasting
    return mask

def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask to prevent decoder from attending to future positions.
    
    Args:
        size (int): Sequence length
    
    Returns:
        torch.Tensor: Upper triangular mask (1, size, size)
    """
    # Create upper triangular matrix with 1s above diagonal (future positions)
    mask = 1 - torch.triu(torch.ones(1, size, size), diagonal=1)
    return mask  # 0 for future positions