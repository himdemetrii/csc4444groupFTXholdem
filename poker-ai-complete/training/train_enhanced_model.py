"""
Example: Training the Enhanced Poker Model with Game Context

This script shows how to train the EnhancedPokerNet with context features.
You'll need to adapt this to your training data generation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from poker_model_enhanced import EnhancedPokerNet


def generate_mock_training_data(num_samples=1000):
    """
    Generate mock training data with context features.
    
    In practice, you'd generate this from actual poker games.
    """
    print(f"Generating {num_samples} training samples with context...")
    
    # Mock data
    hole_cards = torch.randint(0, 13, (num_samples, 2, 2))
    board_cards = torch.randint(0, 13, (num_samples, 5, 2))
    
    # Context features (7 features per sample)
    contexts = torch.rand(num_samples, 7)
    
    # Target actions (random for demo)
    target_actions = torch.randint(1, 7, (num_samples,))
    
    # Target values (random for demo)
    target_values = torch.randn(num_samples, 1)
    
    return {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'contexts': contexts,
        'target_actions': target_actions,
        'target_values': target_values
    }


def train_enhanced_model(epochs=30, batch_size=64):
    """Train the enhanced poker model with context."""
    
    print("\n" + "="*70)
    print("TRAINING ENHANCED POKER MODEL")
    print("="*70)
    
    # Create model
    model = EnhancedPokerNet(embed_dim=64, hidden_dim=256, num_actions=7)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate training data
    train_data = generate_mock_training_data(num_samples=5000)
    val_data = generate_mock_training_data(num_samples=1000)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    action_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    value_loss_fn = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        num_batches = len(train_data['target_actions']) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            hole = train_data['hole_cards'][start_idx:end_idx]
            board = train_data['board_cards'][start_idx:end_idx]
            context = train_data['contexts'][start_idx:end_idx]
            target_action = train_data['target_actions'][start_idx:end_idx]
            target_value = train_data['target_values'][start_idx:end_idx]
            
            # Forward pass (WITH CONTEXT!)
            logits, value = model(hole, board, context)
            
            # Losses
            action_loss = action_loss_fn(logits, target_action)
            value_loss = value_loss_fn(value, target_value)
            loss = action_loss + 0.5 * value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == target_action).sum().item()
        
        scheduler.step()
        
        # Epoch stats
        avg_loss = total_loss / num_batches
        train_acc = 100 * correct / (num_batches * batch_size)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_value = model(
                val_data['hole_cards'],
                val_data['board_cards'],
                val_data['contexts']
            )
            val_action_loss = action_loss_fn(val_logits, val_data['target_actions'])
            val_value_loss = value_loss_fn(val_value, val_data['target_values'])
            val_loss = val_action_loss + 0.5 * val_value_loss
            val_acc = 100 * (val_logits.argmax(dim=1) == val_data['target_actions']).sum().item() / len(val_data['target_actions'])
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Loss={avg_loss:.4f} Train={train_acc:.1f}% "
              f"Val={val_loss.item():.4f} ValAcc={val_acc:.1f}%")
        
        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), 'enhanced_model_best.pth')
            print("  ✨ Saved best model!")
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: enhanced_model_best.pth")
    print("="*70)
    
    return model


if __name__ == "__main__":
    print("\n🤖 Enhanced Poker Model Training Example")
    print("="*70)
    print("\nNOTE: This uses MOCK data for demonstration.")
    print("In practice, you should:")
    print("  1. Generate real poker game data")
    print("  2. Extract context features from each game state")
    print("  3. Label actions based on outcomes")
    print("  4. Train on real data")
    print("\n" + "="*70)
    
    # Train
    model = train_enhanced_model(epochs=30, batch_size=64)
    
    print("\n✅ Done! Use the model with:")
    print("   python ../src/poker_client_enhanced.py --model enhanced_model_best.pth")
