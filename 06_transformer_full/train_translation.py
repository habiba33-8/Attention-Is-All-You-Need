import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from transformer import Transformer

from dataset import (
    Multi30kDataset,
    collate_fn
)

from masking import (
    create_padding_mask,
    create_look_ahead_mask
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_config(config_path='config.yaml'):

    if not os.path.isabs(config_path):

        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config_path
        )

    with open(config_path, 'r') as f:

        config = yaml.safe_load(f)

    return config


def train_model(config):


    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../data/multi30k_local_data/train"
    )

    dataset = Multi30kDataset(dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"Dataset Size: {len(dataset)}")


    src_vocab_size = len(dataset.src_vocab)
    tgt_vocab_size = len(dataset.tgt_vocab)

    print(f"Source Vocabulary Size: {src_vocab_size}")
    print(f"Target Vocabulary Size: {tgt_vocab_size}")


    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )


    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['learning_rate']),
        betas=(0.9, 0.98),
        eps=1e-9
    )


    num_epochs = int(config['epochs'])

    print("\nStarting Training...\n")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(dataloader):

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)


            tgt_input = tgt[:, :-1]

            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src).to(DEVICE)

            tgt_padding_mask = create_padding_mask(
                tgt_input
            ).to(DEVICE)

            look_ahead_mask = create_look_ahead_mask(
                tgt_input.size(1)
            ).to(DEVICE)

            tgt_mask = tgt_padding_mask & look_ahead_mask


            output = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask
            )

            output = output.contiguous().view(
                -1,
                tgt_vocab_size
            )

            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(
                output,
                tgt_output
            )


            optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:

                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )


        avg_loss = total_loss / len(dataloader)

        print(
            f"\nEpoch {epoch+1} Completed "
            f"| Average Loss: {avg_loss:.4f}\n"
        )


    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    save_path = os.path.join(
        checkpoint_dir,
        "transformer.pt"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':

    config = load_config()

    train_model(config)
