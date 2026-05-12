"""
Ablation study: Single-head vs Multi-head Attention
--------------------------------------------------
Compares performance of Transformer using:
  - 1 attention head (single-head attention)
  - 8 attention heads (multi-head attention, as in the base model)

This directly corresponds to Table 3 row (A) in the paper:
  - h=1 → BLEU ≈24.9 (paper reports 0.9 worse than best multi-head setting)
  - h=8 → BLEU ≈25.8 (base model)

We keep total computation roughly similar by adjusting d_k / d_v accordingly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from copy import deepcopy

# ────────────────────────────────────────────────
#   Import your model components
# ────────────────────────────────────────────────
# Assuming these are available from your project structure
from multi_head import MultiHeadAttention
from positional_encoding import PositionalEncoding
from encoder_block import EncoderLayer
from decoder_block import DecoderLayer
# If you have a full Transformer class, you can use it instead

# ────────────────────────────────────────────────
#   Small Transformer variants for quick ablation
# ────────────────────────────────────────────────
class SimpleTransformer(nn.Module):
    """
    Minimal Transformer used only for this ablation study.
    Contains:
      - Embedding + Positional Encoding
      - Stack of encoder layers
      - Stack of decoder layers
      - Final linear projection
    """
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder & Decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        # Scale embeddings (as in paper)
        self.embedding_scale = math.sqrt(d_model)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Source path
        src = self.embedding(src) * self.embedding_scale
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        
        # Target path
        tgt = self.embedding(tgt) * self.embedding_scale
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        
        return self.output_linear(dec_out)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, epochs=5, device='cpu', lr=0.001):
    """
    Very simple training loop for demonstration/ablation purposes.
    In real experiments → use full training pipeline + validation BLEU.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding token = 0
    
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher forcing: input = tgt[:,:-1], target = tgt[:,1:]
            output = model(src, tgt[:, :-1])
            
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        total_loss += avg_loss
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")
    
    return total_loss / epochs


# ────────────────────────────────────────────────
#   Ablation: run single-head vs multi-head
# ────────────────────────────────────────────────
def run_ablation():
    print("="*70)
    print("Ablation: Single-Head vs Multi-Head Attention")
    print("="*70)
    
    # Common settings (small for fast ablation)
    vocab_size = 1000       # dummy small vocab
    seq_len = 20
    batch_size = 32
    d_model = 128           # smaller than paper to run quickly
    num_layers = 2          # fewer layers
    d_ff = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy random data (in real experiment use real parallel corpus)
    n_samples = 2000
    src_data = torch.randint(1, vocab_size, (n_samples, seq_len))
    tgt_data = torch.randint(1, vocab_size, (n_samples, seq_len))
    
    dataset = TensorDataset(src_data, tgt_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ── Case 1: Multi-head (like paper base) ────────────────────────
    print("\nTraining MULTI-HEAD model (h=8) ...")
    model_multi = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8,
        d_ff=d_ff,
        dropout=dropout
    )
    print(f"Multi-head params: {count_parameters(model_multi):,}")
    loss_multi = train_model(model_multi, loader, epochs=6, device=device, lr=0.001)
    
    # ── Case 2: Single-head ─────────────────────────────────────────
    print("\nTraining SINGLE-HEAD model (h=1) ...")
    model_single = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=1,                # ← the key difference
        d_ff=d_ff,
        dropout=dropout
    )
    print(f"Single-head params: {count_parameters(model_single):,}")
    loss_single = train_model(model_single, loader, epochs=6, device=device, lr=0.001)
    
    # ── Comparison ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    print(f"Multi-head (h=8)  → average loss: {loss_multi:.4f}")
    print(f"Single-head (h=1) → average loss: {loss_single:.4f}")
    diff = loss_single - loss_multi
    print(f"Difference:        {diff:+.4f}  (positive = single-head worse)")
    print("="*70)
    
    if diff > 0:
        print("→ Consistent with paper: multi-head performs better")
    else:
        print("→ Unexpected result (possibly due to small dummy data / random init)")


if __name__ == "__main__":
    run_ablation()