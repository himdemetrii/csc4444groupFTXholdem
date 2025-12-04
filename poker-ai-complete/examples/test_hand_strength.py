"""
Example: Testing Hand Strength Calculation

This script demonstrates how hand strength is calculated for different scenarios.
Useful for understanding and debugging the hand evaluation logic.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from poker_model_enhanced import calculate_hand_strength


def test_hand_scenarios():
    """Test various poker hand scenarios."""
    
    print("\n" + "="*70)
    print("🎴 HAND STRENGTH CALCULATION EXAMPLES")
    print("="*70)
    
    # Test scenarios: (hole_cards, board_cards, description)
    scenarios = [
        # Preflop scenarios
        ([[12, 0], [12, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         "Pocket Aces (AA) - Preflop"),
        
        ([[11, 0], [11, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         "Pocket Kings (KK) - Preflop"),
        
        ([[12, 0], [11, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         "Ace-King Suited (AKs) - Preflop"),
        
        ([[12, 0], [11, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         "Ace-King Offsuit (AKo) - Preflop"),
        
        ([[1, 0], [0, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         "3-2 Offsuit (32o) - Preflop"),
        
        # Post-flop scenarios
        ([[12, 0], [11, 0]], [[10, 0], [9, 0], [5, 1], [0, 0], [0, 0]], 
         "AKs on QTs5 board (flush draw + overs)"),
        
        ([[6, 0], [6, 1]], [[6, 2], [3, 0], [1, 1], [0, 0], [0, 0]], 
         "Pocket 8s on 843 board (set of 8s)"),
        
        ([[12, 0], [11, 0]], [[12, 1], [11, 1], [10, 0], [0, 0], [0, 0]], 
         "AK on AKT board (top two pair)"),
        
        # Made hands
        ([[5, 0], [4, 1]], [[5, 1], [5, 2], [0, 0], [4, 2], [3, 0]], 
         "77 on 775 board, 64 on turn (full house)"),
        
        ([[8, 0], [8, 1]], [[8, 2], [8, 3], [3, 0], [0, 0], [0, 0]], 
         "TT on TT3 board (quads)"),
        
        ([[7, 0], [6, 0]], [[5, 0], [4, 0], [3, 0], [0, 0], [0, 0]], 
         "98s on 765 board (straight)"),
        
        ([[10, 0], [8, 0]], [[7, 0], [5, 0], [2, 0], [0, 0], [0, 0]], 
         "QT of spades on 986s board (flush)"),
    ]
    
    print("\n")
    
    for hole, board, desc in scenarios:
        strength = calculate_hand_strength(hole, board)
        
        # Format cards for display
        rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suit_names = ['♠', '♥', '♦', '♣']
        
        hole_str = f"{rank_names[hole[0][0]]}{suit_names[hole[0][1]]}{rank_names[hole[1][0]]}{suit_names[hole[1][1]]}"
        
        board_cards = [card for card in board if card[0] > 0]
        if board_cards:
            board_str = " ".join([f"{rank_names[c[0]]}{suit_names[c[1]]}" for c in board_cards])
        else:
            board_str = "(preflop)"
        
        # Determine strength category
        if strength >= 0.8:
            category = "💎 PREMIUM"
        elif strength >= 0.6:
            category = "🔥 STRONG"
        elif strength >= 0.4:
            category = "✅ GOOD"
        elif strength >= 0.2:
            category = "⚠️  MARGINAL"
        else:
            category = "❌ WEAK"
        
        print(f"{category}")
        print(f"   Scenario: {desc}")
        print(f"   Hole: {hole_str} | Board: {board_str}")
        print(f"   Strength: {strength:.3f}")
        print()
    
    print("="*70)
    print("\nInterpretation:")
    print("  0.00 - 0.20: Weak (fold preflop, careful post-flop)")
    print("  0.20 - 0.40: Marginal (position-dependent)")
    print("  0.40 - 0.60: Good (value bet territory)")
    print("  0.60 - 0.80: Strong (raise/3-bet)")
    print("  0.80 - 1.00: Premium (aggressive action)")
    print("="*70)


def test_custom_hand():
    """Test a custom hand entered by user."""
    
    print("\n" + "="*70)
    print("🎴 TEST YOUR OWN HAND")
    print("="*70)
    
    print("\nEnter cards using format:")
    print("  Ranks: 2-9, T, J, Q, K, A")
    print("  Suits: s(♠), h(♥), d(♦), c(♣)")
    print("  Example: Ah Kh for Ace-King of hearts")
    
    # This is just a template - would need input parsing
    print("\n(Input parsing not implemented in this example)")
    print("Modify the code to add your custom hands!")


if __name__ == "__main__":
    test_hand_scenarios()
    # test_custom_hand()  # Uncomment to test custom hands
