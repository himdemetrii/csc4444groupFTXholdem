"""
Poker Model Module - For import by the WebSocket client
"""

import torch
import torch.nn as nn
from collections import Counter


class CardEncoder(nn.Module):
    """Converts cards into embeddings that the neural network can understand."""
    def __init__(self, embed_dim=32):
        super(CardEncoder, self).__init__()
        self.rank_emb = nn.Embedding(13, embed_dim)
        self.suit_emb = nn.Embedding(4, embed_dim)
    
    def forward(self, cards):
        ranks = self.rank_emb(cards[:, :, 0])
        suits = self.suit_emb(cards[:, :, 1])
        return ranks + suits


class SimplePokerNet(nn.Module):
    """A straightforward neural network for poker decisions."""
    def __init__(self, embed_dim=64, hidden_dim=128, num_actions=7):
        super(SimplePokerNet, self).__init__()
        
        self.card_encoder = CardEncoder(embed_dim)
        card_features = 7 * embed_dim
        
        self.network = nn.Sequential(
            nn.Linear(card_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.4),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, 1),
            nn.Tanh()
        )
    
    def forward(self, hole_cards, board_cards):
        hole_emb = self.card_encoder(hole_cards)
        board_emb = self.card_encoder(board_cards)
        
        hole_flat = hole_emb.view(hole_emb.shape[0], -1)
        board_flat = board_emb.view(board_emb.shape[0], -1)
        features = torch.cat([hole_flat, board_flat], dim=1)
        
        x = self.network(features)
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        return logits, value


def calculate_hand_strength(hole_cards, board_cards):
    """Calculate hand strength - handles [0,0] cards properly."""
    hole_ranks = [hole_cards[0][0], hole_cards[1][0]]
    hole_suits = [hole_cards[0][1], hole_cards[1][1]]
    
    board_ranks = [card[0] for card in board_cards if card[0] > 0]
    board_suits = [card[1] for card in board_cards if card[0] > 0]
    
    all_ranks = hole_ranks + board_ranks
    all_suits = hole_suits + board_suits
    
    strength = 0.0
    
    # Pocket pairs
    if hole_ranks[0] == hole_ranks[1]:
        pair_rank = hole_ranks[0]
        strength = 0.5 + (pair_rank / 13.0) * 0.3
        if pair_rank >= 8:
            strength += 0.15
        return min(strength, 1.0)
    
    # High cards
    max_rank = max(hole_ranks)
    min_rank = min(hole_ranks)
    
    if max_rank >= 11:
        strength += 0.15
        if max_rank == 12:
            strength += 0.10
    elif max_rank >= 9:
        strength += 0.08
    
    if min_rank >= 9:
        strength += 0.12
    
    # Suited
    if hole_suits[0] == hole_suits[1]:
        strength += 0.05
        
        if len(board_suits) > 0:
            suit_counts = Counter(all_suits)
            max_suit = max(suit_counts.values())
            if max_suit == 4:
                strength += 0.15
            elif max_suit >= 5:
                strength += 0.45
    
    # Connected cards
    rank_diff = abs(hole_ranks[0] - hole_ranks[1])
    if rank_diff <= 1 and max_rank >= 8:
        strength += 0.08
    elif rank_diff == 2 and max_rank >= 10:
        strength += 0.05
    
    # Made hands
    if len(board_ranks) > 0:
        rank_counts = Counter(all_ranks)
        
        for hole_rank in hole_ranks:
            count = rank_counts[hole_rank]
            if count == 2:
                strength += 0.30 + (hole_rank / 13.0) * 0.10
            elif count == 3:
                strength += 0.65
            elif count == 4:
                strength += 0.80
        
        pairs = [r for r, c in rank_counts.items() if c >= 2]
        if len(pairs) >= 2:
            strength += 0.20
    
    return min(strength, 1.0)
