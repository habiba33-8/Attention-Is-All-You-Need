import torch
import unittest
from attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):
    """
    Unit tests for the Scaled Dot-Product Attention function.
    """
    
    def setUp(self):
        # Step 1: Set up test parameters
        self.batch_size = 2  
        self.seq_len = 3
        self.d_k = 4         
        self.d_v = 4      
        
        # Step 2: Create sample tensors for query, key, value
        self.query = torch.randn(self.batch_size, 1, self.seq_len, self.d_k)  
        self.key = torch.randn(self.batch_size, 1, self.seq_len, self.d_k)
        self.value = torch.randn(self.batch_size, 1, self.seq_len, self.d_v)
        
        # Step 3: Create a sample mask (e.g., for padding or look-ahead)
        self.mask = torch.tensor([[[[1, 1, 0], [1, 0, 0], [1, 0, 0]]]])  
        
    def test_attention_output_shape(self):
        # Step 4: Call the function without mask
        output, weights = scaled_dot_product_attention(self.query, self.key, self.value)
        
        # Step 5: Assert output shape is correct
        self.assertEqual(output.shape, (self.batch_size, 1, self.seq_len, self.d_v))  
        self.assertEqual(weights.shape, (self.batch_size, 1, self.seq_len, self.seq_len))
        
    def test_attention_with_mask(self):
        # Step 6: Call with mask
        output, weights = scaled_dot_product_attention(self.query, self.key, self.value, self.mask.repeat(self.batch_size,1,1,1))
        
        self.assertTrue(torch.all(weights[:,:,:,-1] == 0)) 
        
    def test_softmax_normalization(self):
        # Step 8: Ensure weights sum to 1 along the last dimension
        _, weights = scaled_dot_product_attention(self.query, self.key, self.value)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))))

if __name__ == '__main__':
    unittest.main()
