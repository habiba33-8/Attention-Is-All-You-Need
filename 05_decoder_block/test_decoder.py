import torch
import unittest
from decoder_block import DecoderLayer

class TestDecoderLayer(unittest.TestCase):
    """
    Unit tests for the Decoder Layer.
    """
    
    def setUp(self):
        # Step 1: Parameters
        self.d_model = 8
        self.num_heads = 2
        self.d_ff = 16
        self.batch_size = 2
        self.seq_len = 3
        
        # Step 2: Initialize
        self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff)
        
        # Step 3: Sample inputs (tgt and enc_output)
        self.tgt = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.enc_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_output_shape(self):
        # Step 4: Forward
        output = self.decoder_layer(self.tgt, self.enc_output)
        
        # Step 5: Check
        self.assertEqual(output.shape, self.tgt.shape)
        
    def test_with_masks(self):
        # Step 6: Sample masks
        src_mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
        tgt_mask = torch.tril(torch.ones(1, self.seq_len, self.seq_len))  # Look-ahead
        
        # Step 7: Forward with masks
        output = self.decoder_layer(self.tgt, self.enc_output, src_mask, tgt_mask)
        self.assertEqual(output.shape, self.tgt.shape)

if __name__ == '__main__':
    unittest.main()