import os
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
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) 
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
    
    # Step 2: Loss and optimizer (Adam; use learning_rate > 0 so the model can actually fit data)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    lr = float(config.get('learning_rate', 3e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Step 3: Dummy data for example (replace with real dataset)
    demo_seq = min(int(config.get('demo_seq_len', 128)), int(config['max_len']))
    src = torch.randint(1, config['src_vocab_size'], (config['batch_size'], demo_seq))
    tgt = torch.randint(1, config['tgt_vocab_size'], (config['batch_size'], demo_seq))
    
    # Step 4: Training loop (one fixed batch repeated when overfit_single_batch is true)
    num_epochs = int(config['overfit_epochs']) if config.get('overfit_single_batch') else int(config['epochs'])
    overfit = bool(config.get('overfit_single_batch'))
    if overfit:
        print(f'Overfitting test: single batch, {num_epochs} epochs, lr={lr}')

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Step 5: Forward pass (shift tgt for teacher forcing)
        output = model(src, tgt[:, :-1]) 
        if epoch == 0:
            print(output.shape)
            pred = output.argmax(dim=-1)
            print(pred)
        loss = criterion(output.contiguous().view(-1, config['tgt_vocab_size']), tgt[:, 1:].contiguous().view(-1))

        # Step 6: Backward and optimize
        loss.backward()
        optimizer.step()

        if overfit:
            if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % max(1, num_epochs // 20) == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        else:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # Step 7: Save model (optional)
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer.pth')
    torch.save(model.state_dict(), _out)

if __name__ == '__main__':
    config = load_config()
    train_model(config)
