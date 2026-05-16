"""
Ablation Study: Single-head vs Multi-head Attention
----------------------------------------------------
Compares Transformer performance with:
  - 1 attention head
  - 8 attention heads (baseline)

Metric:
  - Cross-entropy loss (train + validation)

Note:
This is a controlled synthetic experiment for architecture analysis.
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _subdir in (
    "02_multi_head_attention",
    "03_positional_encoding",
    "04_encoder_block",
    "05_decoder_block",
):
    _p = os.path.join(_REPO_ROOT, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from positional_encoding import PositionalEncoding
from encoder_block import EncoderLayer
from decoder_block import DecoderLayer


# -----------------------------
# Model
# -----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, src, tgt):
        src = self.embedding(src) * self.scale
        src = self.pos_encoding(src)
        src = self.dropout(src)

        for layer in self.encoder_layers:
            src = layer(src)

        tgt = self.embedding(tgt) * self.scale
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src)

        return self.output(tgt)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            out = model(src, tgt[:, :-1])

            loss = criterion(
                out.reshape(-1, out.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)

def train(model, loader, device, epochs=5, lr=1e-3, name="model"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            out = model(src, tgt[:, :-1])

            loss = criterion(
                out.reshape(-1, out.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[{name}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

    train_loss = epoch_loss / len(loader)
    val_loss = evaluate(model, loader, device)

    return train_loss, val_loss



def run_ablation():
    print("=" * 70)
    print("Ablation: Single-head vs Multi-head Attention")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 1000
    seq_len = 20
    batch_size = 32
    samples = 2000
  
    src = torch.randint(1, vocab_size, (samples, seq_len))
    tgt = torch.randint(1, vocab_size, (samples, seq_len))

    dataset = TensorDataset(src, tgt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------- Multi-head ----------------
    print("\n[Multi-head model]")
    model_multi = SimpleTransformer(
        vocab_size=vocab_size,
        num_heads=8
    )

    print(f"Params: {count_params(model_multi):,}")

    train_m, val_m = train(
        model_multi,
        loader,
        device,
        epochs=5,
        name="multi-head"
    )

    # ---------------- Single-head ----------------
    print("\n[Single-head model]")
    model_single = SimpleTransformer(
        vocab_size=vocab_size,
        num_heads=1
    )

    print(f"Params: {count_params(model_single):,}")

    train_s, val_s = train(
        model_single,
        loader,
        device,
        epochs=5,
        name="single-head"
    )

    # ---------------- Results ----------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("-" * 70)

    print(f"Multi-head  → train: {train_m:.4f} | val: {val_m:.4f}")
    print(f"Single-head → train: {train_s:.4f} | val: {val_s:.4f}")

    print("\nΔ validation loss:", round(val_s - val_m, 4))

    if val_s > val_m:
        print("→ Multi-head performs better (consistent with paper)")
    else:
        print("→ Unexpected result (check setup / randomness)")


if __name__ == "__main__":
    run_ablation()
