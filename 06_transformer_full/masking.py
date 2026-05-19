import torch


def create_padding_mask(seq, pad_token=0):
    """
    Creates padding mask.

    Args:
        seq (torch.Tensor):
            Shape -> (batch_size, seq_len)

        pad_token (int):
            Padding token index.

    Returns:
        torch.Tensor:
            Shape -> (batch_size, 1, 1, seq_len)

            True  -> valid token
            False -> padding token
    """

    mask = (
        seq != pad_token
    ).unsqueeze(1).unsqueeze(2)

    return mask


def create_look_ahead_mask(size):
    """
    Creates causal/look-ahead mask for decoder.

    Prevents decoder from attending
    to future positions.

    Args:
        size (int):
            Target sequence length.

    Returns:
        torch.Tensor:
            Shape -> (1, size, size)

            Example for size=4:

            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
    """

    mask = torch.tril(
        torch.ones(size, size)
    ).bool()

    return mask.unsqueeze(0)


# ---------------------------------------------------
# Quick Test
# ---------------------------------------------------

if __name__ == "__main__":

    sample_seq = torch.tensor([
        [1, 5, 7, 0, 0],
        [1, 8, 9, 4, 0]
    ])

    padding_mask = create_padding_mask(sample_seq)

    look_ahead_mask = create_look_ahead_mask(5)

    print("Padding Mask Shape:")
    print(padding_mask.shape)

    print("\nPadding Mask:")
    print(padding_mask)

    print("\nLook Ahead Mask Shape:")
    print(look_ahead_mask.shape)

    print("\nLook Ahead Mask:")
    print(look_ahead_mask)