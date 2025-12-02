"""
Enhanced Poker Model - WITH Game Context
This version includes chip counts, pot size, phase, and position information
"""

import torch
import torch.nn as nn
from collections import Counter


class CardEncoder(nn.Module):
    """Converts cards into embeddings."""
    def __init__(self, embed_dim=32):
        super(CardEncoder, self).__init__()
        self.rank_emb = nn.Embedding(13, embed_dim)
        self.suit_emb = nn.Embedding(4, embed_dim)
    
    def forward(self, cards):
        ranks = self.rank_emb(cards[:, :, 0])
        suits = self.suit_emb(cards[:, :, 1])
        return ranks + suits


class EnhancedPokerNet(nn.Module):
    """
    Enhanced poker network that includes game context.
    
    Inputs:
        - hole_cards: [batch, 2, 2] - Your cards
        - board_cards: [batch, 5, 2] - Community cards
        - context: [batch, 7] - Game state features:
            [0] = my_chips (normalized by starting stack)
            [1] = pot_size (normalized by starting stack)
            [2] = phase (0=WAITING, 1=PREFLOP, 2=FLOP, 3=TURN, 4=RIVER, 5=SHOWDOWN)
            [3] = position (0-N, seat index normalized)
            [4] = num_players (2-10)
            [5] = stack_to_pot_ratio (SPR)
            [6] = hand_number (to track progression)
    """
    def __init__(self, embed_dim=64, hidden_dim=256, num_actions=7):
        super(EnhancedPokerNet, self).__init__()
        
        self.card_encoder = CardEncoder(embed_dim)
        
        # Card features
        card_features = 7 * embed_dim  # 2 hole + 5 board
        
        # Context features (7 additional inputs)
        context_features = 7
        
        # Combined feature size
        total_features = card_features + context_features
        
        # Main network with context awareness
        self.network = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
        )
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, hole_cards, board_cards, context):
        """
        Forward pass with context.
        
        Args:
            hole_cards: [batch, 2, 2]
            board_cards: [batch, 5, 2]
            context: [batch, 7]
        """
        # Encode cards
        hole_emb = self.card_encoder(hole_cards)
        board_emb = self.card_encoder(board_cards)
        
        # Flatten card embeddings
        hole_flat = hole_emb.view(hole_emb.shape[0], -1)
        board_flat = board_emb.view(board_emb.shape[0], -1)
        card_features = torch.cat([hole_flat, board_flat], dim=1)
        
        # Combine with context
        features = torch.cat([card_features, context], dim=1)
        
        # Pass through network
        x = self.network(features)
        
        # Get outputs
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        return logits, value


def extract_context_features(state: dict, player_id: str, starting_stack: int = 1000) -> list:
    """
    Extract game context features from state.
    
    Returns list of 7 features:
        [my_chips_norm, pot_norm, phase, position, num_players, spr, hand_num]
    """
    context = [0.0] * 7
    
    try:
        # Find my player
        table = state.get('table', {})
        players = table.get('players', [])
        my_player = None
        my_position = 0
        
        for i, p in enumerate(players):
            if p and p.get('id') == player_id:
                my_player = p
                my_position = i
                break
        
        if my_player:
            # [0] my_chips (normalized)
            my_chips = my_player.get('chips', 0)
            context[0] = my_chips / starting_stack
            
            # [1] pot_size (normalized)
            pot = state.get('pot', 0)
            context[1] = pot / starting_stack
            
            # [2] phase (encoded as int)
            phase_map = {
                'WAITING': 0, 'PREFLOP': 1, 'FLOP': 2, 
                'TURN': 3, 'RIVER': 4, 'SHOWDOWN': 5
            }
            phase = state.get('phase', 'WAITING')
            context[2] = phase_map.get(phase, 0) / 5.0  # Normalize to [0, 1]
            
            # [3] position (normalized)
            num_players = len([p for p in players if p is not None])
            context[3] = my_position / max(num_players - 1, 1)
            
            # [4] num_players (normalized)
            context[4] = num_players / 10.0  # Assume max 10 players
            
            # [5] stack-to-pot ratio (SPR)
            if pot > 0:
                context[5] = min(my_chips / pot, 10.0) / 10.0  # Cap at 10, normalize
            else:
                context[5] = 1.0  # High SPR when pot is 0
            
            # [6] hand_number (for tracking progression)
            hand_num = state.get('hand', 0)
            context[6] = min(hand_num / 100.0, 1.0)  # Normalize, cap at 100
    
    except Exception as e:
        print(f"⚠️  Error extracting context: {e}")
    
    return context


def calculate_hand_strength(hole_cards, board_cards):
    """Calculate hand strength - same as before."""
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


# Backward compatibility: allow importing SimplePokerNet
SimplePokerNet = EnhancedPokerNet
