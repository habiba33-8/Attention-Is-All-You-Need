import torch
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
        # Step 6: Disable sub-layers to check residuals (for testing)
        with torch.no_grad():
            self.encoder_layer.self_attn.output_linear.weight.fill_(0)  # Zero attention
            self.encoder_layer.feed_forward[0].weight.fill_(0)         # Zero FF
            output = self.encoder_layer(self.input)
            # Output should be close to input due to residuals
            self.assertTrue(torch.allclose(output, self.input, atol=1e-5))

if __name__ == '__main__':
    unittest.main()