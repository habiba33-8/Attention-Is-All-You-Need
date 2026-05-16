import os
import sys

import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRANSFORMER_DIR = os.path.join(_REPO_ROOT, "06_transformer_full")
if _TRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, _TRANSFORMER_DIR)
from transformer import Transformer


def run_ablation():
    """
    Runs ablation study without positional encoding.
    """
    # Step 1: Model without PE
    class NoPETransformer(Transformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_encoding = nn.Identity()  
    # Step 2: Compare with normal model (train and evaluate)
    print("Ablation: Training without PE...")
    # Add training code similar to train.py

if __name__ == '__main__':
    run_ablation()
