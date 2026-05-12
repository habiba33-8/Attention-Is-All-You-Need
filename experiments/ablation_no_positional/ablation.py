from transformer import Transformer, PositionalEncoding

def run_ablation():
    """
    Runs ablation study without positional encoding.
    """
    # Step 1: Model without PE
    class NoPETransformer(Transformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_encoding = nn.Identity()  # Replace PE with identity
    
    # Step 2: Compare with normal model (train and evaluate)
    # (Dummy print for example)
    print("Ablation: Training without PE...")
    # Add training code similar to train.py

if __name__ == '__main__':
    run_ablation()