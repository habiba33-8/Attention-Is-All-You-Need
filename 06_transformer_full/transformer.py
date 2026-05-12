import torch.nn as nn

class Transformer(nn.Module):
    """
    Full Transformer model with encoder and decoder stacks.
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        d_model (int): Model dimension (512)
        num_layers (int): Number of layers (N=6)
        num_heads (int): Attention heads (8)
        d_ff (int): Feed-forward dim (2048)
        max_len (int): Max sequence length
        dropout (float): Dropout rate (0.1)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers stack
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Decoder layers stack
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Final linear and softmax for output probabilities
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale embeddings by sqrt(d_model) as in paper
        self.scale = math.sqrt(d_model)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Step 1: Source embedding + positional + dropout
        src = self.src_embedding(src) * self.scale
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        # Step 2: Encoder stack
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Step 3: Target embedding + positional + dropout
        tgt = self.tgt_embedding(tgt) * self.scale
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # Step 4: Decoder stack
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Step 5: Final linear projection
        output = self.output_linear(dec_output)
        
        return output