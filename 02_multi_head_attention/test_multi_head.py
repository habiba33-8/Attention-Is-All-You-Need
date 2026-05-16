import torch
import unittest
from multi_head import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    """
    Unit tests for the Multi-Head Attention module.
    """
    
    def setUp(self):
        # Step 1: Set up parameters
        self.d_model = 8   
        self.num_heads = 2 
        self.batch_size = 2
        self.seq_len = 3
        
        # Step 2: Initialize the module
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        
        # Step 3: Sample inputs
        self.input = torch.randn(self.batch_size, self.seq_len, self.d_model) 
        
    def test_output_shape(self):
        # Step 4: Forward pass
        output = self.mha(self.input, self.input, self.input)
        
        # Step 5: Check shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))  
        
    def test_with_mask(self):
        # Step 6: Create a mask
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len) 
        mask[:, :, :, -1] = 0  
        
        # Step 7: Forward with mask
        output = self.mha(self.input, self.input, self.input, mask)
        
        # Step 8: Basic check (output should still be valid shape)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_projection_dimensions(self):
        # Step 9: Check internal dimensions
        self.assertEqual(self.mha.d_k, self.d_model // self.num_heads)  
        self.assertEqual(self.mha.d_v, self.d_model // self.num_heads)  

if __name__ == '__main__':
    unittest.main()
