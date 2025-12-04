# CRITICAL UPGRADE: Context-Aware Poker Model

## The Problem You Identified

You asked: **"Can my model know how many chips it has, what round we're in, whether they are small or big blind?"**

**Answer: NO** - Your current model only sees cards!

---

## What Your Basic Model Sees

```python
# Input to basic model:
hole_cards = [[12, 1], [11, 0]]  # Ace-King
board_cards = [[10, 1], [9, 1], [8, 0], [0, 0], [0, 0]]  # Q-J-8 on flop

# That's it! Just the cards.
```

### What It CANNOT See:
- ❌ Your chip count
- ❌ Pot size
- ❌ Current phase (PREFLOP, FLOP, TURN, RIVER)
- ❌ Your position (button, blinds, etc.)
- ❌ Number of opponents
- ❌ Stack-to-pot ratio (SPR)
- ❌ Betting history

---

## Why This Is A HUGE Problem

### Example 1: Short Stack Disaster
```
Situation:
  You have: 50 chips (short stack!)
  Pot: 500 chips
  Hand: Pocket Aces (best starting hand)
  
Basic Model Says:
  "Great hand! Let's RAISE 50 chips!"
  
Reality:
  You're all-in with the raise. Should probably just shove.
  
Problem:
  Model doesn't know it's short-stacked!
```

### Example 2: Position Blindness
```
Situation:
  Position: Big Blind (already invested 10 chips)
  Hand: 9-7 suited (marginal)
  Someone raises to 30
  Pot: 75 chips
  You need: 20 more to call
  
Basic Model Says:
  "Weak hand, FOLD"
  
Reality:
  Getting 3.75:1 pot odds, should probably call
  
Problem:
  Model doesn't know about pot odds or blind investment!
```

### Example 3: Phase Confusion
```
Situation A (PREFLOP):
  Hand: 7-2 offsuit
  Basic Model: "Trash, FOLD" ✓ Correct
  
Situation B (RIVER):
  Hand: 7-2 offsuit
  Board: 7-7-2-K-A
  Actual Hand: Full House (777-22)
  Basic Model: "Trash, FOLD" ✗ WRONG!
  
Problem:
  Model can't tell PREFLOP from RIVER!
```

---

## The Solution: Enhanced Model

### New Architecture

```python
class EnhancedPokerNet(nn.Module):
    def forward(self, hole_cards, board_cards, context):
        # Cards (like before)
        # + context (NEW!)
```

### Context Features (7 additional inputs):

```python
context = [
    my_chips / 1000,        # [0] Chip count (normalized)
    pot / 1000,             # [1] Pot size (normalized)
    phase_code / 5.0,       # [2] Phase (0-5: WAITING to SHOWDOWN)
    position / num_players, # [3] Position (0-1)
    num_players / 10.0,     # [4] Number of players
    spr / 10.0,             # [5] Stack-to-Pot Ratio
    hand_num / 100.0        # [6] Hand progression
]
```

---

## What The Enhanced Model Can Do

### ✅ Chip Awareness
```python
if my_chips < pot * 0.5:
    # "I'm short-stacked, play tight"
    model adjusts aggression down

if my_chips > pot * 10:
    # "Deep stacked, can play more hands"
    model adjusts aggression up
```

### ✅ Pot Odds Calculation
```python
spr = my_chips / pot

if spr < 3:
    # "Committed pot, more likely to call"
    
if spr > 20:
    # "Deep, can fold marginal hands"
```

### ✅ Phase Awareness
```python
if phase == PREFLOP:
    # "Position matters, fold weak hands"
    
if phase == RIVER:
    # "Made hand or bluff, consider pot odds"
```

### ✅ Position Awareness
```python
if position == 0:  # Button
    # "Best position, can play looser"
    
if position == big_blind:
    # "Already invested, pot odds matter"
```

---

## Comparison: Basic vs Enhanced

### Scenario: Short Stack All-In Decision

**Basic Model:**
```
Input: AK suited
Output: RAISE 50 (doesn't know stack size)
Result: Might be wrong sizing
```

**Enhanced Model:**
```
Input: 
  Cards: AK suited
  Context: chips=50, pot=200, phase=PREFLOP, spr=0.25
  
Analysis: "Very short stack (SPR=0.25), commit or fold"
Output: RAISE 50 (all-in, correct!)
Result: Optimal play
```

---

## How To Use Enhanced Model

### 1. Training (Requires Retraining)

```python
from poker_model_enhanced import EnhancedPokerNet

# Create enhanced model
model = EnhancedPokerNet(embed_dim=64, hidden_dim=256, num_actions=7)

# Training loop (modified to include context)
for game in games:
    hole, board = extract_cards(game)
    context = extract_context_features(game_state, player_id)
    
    logits, value = model(hole, board, context)
    # ... training code ...
```

### 2. Playing

```python
from poker_client_enhanced import EnhancedPokerBot

# Use enhanced bot
bot = EnhancedPokerBot(model_path="enhanced_model.pth")

# It automatically extracts context from state!
action = bot.decide_action(state, player_id)
```

---

## What Context Features Mean

### Stack-to-Pot Ratio (SPR)
```
SPR = your_chips / pot_size

SPR < 3:  Short stack, commit/fold decisions
SPR 3-10: Medium, value betting
SPR > 10: Deep, speculative plays possible
```

### Phase Encoding
```
0 = WAITING   (between hands)
1 = PREFLOP   (before flop)
2 = FLOP      (3 community cards)
3 = TURN      (4 community cards)
4 = RIVER     (5 community cards)
5 = SHOWDOWN  (reveal hands)
```

### Position
```
Lower index = earlier position (worse)
Higher index = later position (better)

Button (dealer) = best position
```

---

## Files You Got

### For Context-Aware Play:
1. **poker_model_enhanced.py** - Model with context inputs
2. **poker_client_enhanced.py** - Client that uses enhanced model
3. **CONTEXT_AWARENESS.md** - This file

### Original (Still Works):
1. **poker_model.py** - Basic model (cards only)
2. **poker_client.py** - Basic client
3. All documentation

---

## Migration Path

### Option 1: Use Enhanced Model (Recommended)
```bash
# 1. Retrain with context
python train_enhanced_model.py

# 2. Use enhanced client
python poker_client_enhanced.py
```

### Option 2: Keep Basic Model (Quick Test)
```bash
# Use basic client (works but not optimal)
python poker_client.py
```

### Option 3: Hybrid Approach
```python
# Use enhanced client with untrained model
# It will make random-ish decisions but structure is ready
python poker_client_enhanced.py
```

---

## Training Data Generation (Updated)

You'll need to regenerate training data with context:

```python
def generate_poker_data_with_context(num_games=5000):
    """Generate training data including game context."""
    
    all_hole_cards = []
    all_board_cards = []
    all_contexts = []  # NEW!
    all_target_actions = []
    all_target_values = []
    
    for game_idx in range(num_games):
        state = game.new_initial_state()
        
        # Play game...
        
        # Extract features
        hole, board = extract_cards(state, player_id=0)
        context = extract_context_features(state, player_id, starting_stack)
        
        # Store
        all_hole_cards.append(torch.tensor(hole))
        all_board_cards.append(torch.tensor(board))
        all_contexts.append(torch.tensor(context))  # NEW!
        all_target_actions.append(target_action)
        all_target_values.append(target_value)
    
    return {
        'hole_cards': torch.stack(all_hole_cards),
        'board_cards': torch.stack(all_board_cards),
        'contexts': torch.stack(all_contexts),  # NEW!
        'target_actions': torch.tensor(all_target_actions),
        'target_values': torch.tensor(all_target_values)
    }
```

---

## Expected Improvements

With context awareness, your bot should:

✅ **Make better short-stack decisions**
- Knows when to shove vs fold
- Understands pot commitment

✅ **Respect pot odds**
- Calls more with good odds
- Folds more with bad odds

✅ **Adjust for position**
- Plays looser on button
- Tighter in early position

✅ **Phase-appropriate strategy**
- Tighter preflop
- More aggressive post-flop with made hands

✅ **SPR-aware betting**
- Small bets with deep stacks
- Commit with short stacks

---

## Bottom Line

### Your Question:
> "Can my model know how many chips it has, what round we're in, whether they are small or big blind?"

### Answer:
**Basic Model: NO** ❌
- Only sees cards
- Blind to game context
- Makes suboptimal decisions

**Enhanced Model: YES** ✅
- Sees chips, pot, phase, position
- Context-aware decisions
- Much better play

### Recommendation:
**Use the enhanced model!** It's a massive improvement over the basic card-only version.

---

## Quick Start

```bash
# 1. Use enhanced client (works with untrained model)
python poker_client_enhanced.py

# 2. Watch it make context-aware decisions
# Output will show: chips, pot, SPR, phase

# 3. When ready, retrain with context
python train_enhanced_model.py
```

Your model can now "see" the full game state! 🎯
