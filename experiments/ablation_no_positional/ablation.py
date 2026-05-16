import os
import sys
import torch.nn as nn


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRANSFORMER_DIR = os.path.join(_REPO_ROOT, "06_transformer_full")

if _TRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, _TRANSFORMER_DIR)

from transformer import Transformer


from train import train_model, load_config


class NoPETransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # disable positional encoding
        self.pos_encoding = nn.Identity()


def run_ablation(config):
    print("\n===== Ablation Study: WITHOUT Positional Encoding =====")

    model = NoPETransformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    )

    trained_model = train_model(config)

    print("\nAblation finished.")

if __name__ == "__main__":
    config = load_config()
    run_ablation(config)
