import torch
import unittest
from attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):
    """
    Unit tests for the Scaled Dot-Product Attention function.
    """
    
    def setUp(self):
        # Step 1: Set up test parameters
        self.batch_size = 2  # Small batch for testing
        self.seq_len = 3     # Short sequence length
        self.d_k = 4         # Dimension of keys/queries
        self.d_v = 4         # Dimension of values
        
        # Step 2: Create sample tensors for query, key, value
        self.query = torch.randn(self.batch_size, 1, self.seq_len, self.d_k)  # num_heads=1 for simplicity
        self.key = torch.randn(self.batch_size, 1, self.seq_len, self.d_k)
        self.value = torch.randn(self.batch_size, 1, self.seq_len, self.d_v)
        
        # Step 3: Create a sample mask (e.g., for padding or look-ahead)
        # Each query row must allow at least one key so softmax is defined; last key column stays masked.
        self.mask = torch.tensor([[[[1, 1, 0], [1, 0, 0], [1, 0, 0]]]])  # Shape: (1,1,seq_len,seq_len), batch=1 for example
        
    def test_attention_output_shape(self):
        # Step 4: Call the function without mask
        output, weights = scaled_dot_product_attention(self.query, self.key, self.value)
        
        # Step 5: Assert output shape is correct
        self.assertEqual(output.shape, (self.batch_size, 1, self.seq_len, self.d_v))  # Expected output shape
        self.assertEqual(weights.shape, (self.batch_size, 1, self.seq_len, self.seq_len))  # Weights shape
        
    def test_attention_with_mask(self):
        # Step 6: Call with mask
        output, weights = scaled_dot_product_attention(self.query, self.key, self.value, self.mask.repeat(self.batch_size,1,1,1))
        
        # Step 7: Check if masked positions have -inf effect (weights should be 0 where mask=0 after softmax)
        self.assertTrue(torch.all(weights[:,:,:,-1] == 0))  # Last column masked in example
        
    def test_softmax_normalization(self):
        # Step 8: Ensure weights sum to 1 along the last dimension
        _, weights = scaled_dot_product_attention(self.query, self.key, self.value)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))))

if __name__ == '__main__':
    unittest.main()  # Run the tests