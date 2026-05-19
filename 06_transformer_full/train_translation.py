import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from transformer import Transformer

from dataset import (
    Multi30kDataset,
    make_collate_fn,
)

from masking import (
    create_padding_mask,
    create_look_ahead_mask
)

from checkpoint_io import save_state_dict
from training_viz import run_visualizations


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_config(config_path='config.yaml'):
    """
    Loads configuration from YAML file.
    """

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

    full_dataset = Multi30kDataset(dataset_path)
    train_dataset = full_dataset

    max_train_samples = config.get('max_train_samples')
    if max_train_samples is not None:
        n = min(int(max_train_samples), len(full_dataset))
        train_dataset = Subset(full_dataset, range(n))
        print(f"Using train subset: {n} / {len(full_dataset)} samples")

    max_batches = config.get('max_batches')
    batch_size = int(config['batch_size'])

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(config.get('max_seq_len')),
        num_workers=int(config.get('num_workers', 0)),
    )

    batches_per_epoch = len(dataloader)
    if max_batches is not None:
        batches_per_epoch = min(batches_per_epoch, int(max_batches))


    src_vocab_size = len(full_dataset.src_vocab)
    tgt_vocab_size = len(full_dataset.tgt_vocab)

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
    output_dir = config.get(
        'output_dir',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    )
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        )
    sample_indices = config.get('viz_sample_indices', [0, 100, 500])
    viz_every = int(config.get('viz_every_n_epochs', 1))
    log_every = int(config.get('log_every_n_batches', 50))

    epoch_losses = []
    batch_losses = []
    global_step = 0

    print(f"Device: {DEVICE}")
    print(f"Batches per epoch: {batches_per_epoch} (dataloader has {len(dataloader)})")
    print(f"Total steps (approx): {batches_per_epoch * num_epochs}")
    if DEVICE.type == "cpu":
        print("Tip: use --fast for a much shorter CPU run, or train on a GPU machine.\n")
    print("Starting Training...\n")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, (src, tgt) in enumerate(dataloader):

            if max_batches is not None and batch_idx >= int(max_batches):
                break

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
            num_batches += 1


            global_step += 1

            if batch_idx % log_every == 0:
                batch_losses.append((global_step, loss.item()))
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

    

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)

        print(
            f"\nEpoch {epoch+1} Completed "
            f"| Average Loss: {avg_loss:.4f}\n"
        )

        if (epoch + 1) % viz_every == 0 or epoch + 1 == num_epochs:
            run_visualizations(
                model,
                full_dataset,
                DEVICE,
                epoch + 1,
                epoch_losses,
                batch_losses,
                output_dir,
                sample_indices,
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

    save_state_dict(model.state_dict(), save_path)

    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train EN→DE Transformer")
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Config file name inside 06_transformer_full/',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use config.fast.yaml (smaller model, subset, shorter sequences)',
    )
    args = parser.parse_args()

    config_name = 'config.fast.yaml' if args.fast else args.config
    config = load_config(config_name)

    train_model(config)
