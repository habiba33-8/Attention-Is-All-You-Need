import torch
import torch.nn as nn
import unittest
from encoder_block import EncoderLayer

class TestEncoderLayer(unittest.TestCase):
    """
    Unit tests for the Encoder Layer.
    """
    
    def setUp(self):
        # Step 1: Parameters
        self.d_model = 8
        self.num_heads = 2
        self.d_ff = 16
        self.batch_size = 2
        self.seq_len = 3
        
        # Step 2: Initialize layer
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff)
        
        # Step 3: Sample input
        self.input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_output_shape(self):
        # Step 4: Forward
        output = self.encoder_layer(self.input)
        
        # Step 5: Check shape
        self.assertEqual(output.shape, self.input.shape)
        
    def test_residual_connections(self):
        # Step 6: Zero attention and FFN contributions; with post-norm, output is norm2(norm1(x)), not x.
        with torch.no_grad():
            self.encoder_layer.eval()
            self.encoder_layer.self_attn.output_linear.weight.zero_()
            self.encoder_layer.self_attn.output_linear.bias.zero_()
            for m in self.encoder_layer.feed_forward:
                if isinstance(m, nn.Linear):
                    m.weight.zero_()
                    m.bias.zero_()
            output = self.encoder_layer(self.input)
            expected = self.encoder_layer.norm2(self.encoder_layer.norm1(self.input))
            self.assertTrue(torch.allclose(output, expected, atol=1e-5))

if __name__ == '__main__':
    unittest.main()