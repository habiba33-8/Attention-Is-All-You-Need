import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import yaml

def load_config(config_path='config.yaml'):
    """
    Loads configuration from YAML file.
    
    Args:
        config_path (str): Path to config file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # Safely load YAML
    return config

def train_model(config):
    """
    Training loop for the Transformer model.
    
    Args:
        config (dict): Configuration dictionary.
    """
    # Step 1: Initialize model
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    )
    
    # Step 2: Loss and optimizer (as in paper, Adam with custom schedule)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    # Step 3: Dummy data for example (replace with real dataset)
    # Assume src and tgt are tensors of shape (batch, seq_len)
    src = torch.randint(1, config['src_vocab_size'], (config['batch_size'], config['max_len']))
    tgt = torch.randint(1, config['tgt_vocab_size'], (config['batch_size'], config['max_len']))
    
    # Step 4: Training loop
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        # Step 5: Forward pass (shift tgt for teacher forcing)
        output = model(src, tgt[:, :-1])  # Input up to last token
        loss = criterion(output.contiguous().view(-1, config['tgt_vocab_size']), tgt[:, 1:].contiguous().view(-1))
        
        # Step 6: Backward and optimize
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # Step 7: Save model (optional)
    torch.save(model.state_dict(), 'transformer.pth')

if __name__ == '__main__':
    config = load_config()
    train_model(config)