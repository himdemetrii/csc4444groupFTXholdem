# Integration Guide: PyTorch Poker Model → Go WebSocket Server

## Overview
This guide shows how to connect your trained poker AI model to the Go WebSocket game server.

## Architecture

```
[Go Server] ←─ WebSocket ─→ [Python Client] ←→ [PyTorch Model]
    ↓                              ↓
  Game Logic                  AI Decisions
  - Dealing                   - Card encoding
  - Betting                   - Action selection
  - Hand eval                 - Hand strength
```

## Files Created

1. **poker_model.py** - Your neural network (importable module)
2. **poker_client.py** - WebSocket client that uses the model
3. **integration_test.py** - Testing utilities

## Server Message Format

### Incoming (Server → Client)

**State Update:**
```json
{
  "type": "state",
  "state": {
    "table": {
      "id": "table-1",
      "players": [
        {
          "id": "bot1",
          "chips": 1000,
          "action": "",
          "cards": [
            {"rank": "A", "suit": "HEART"},
            {"rank": "K", "suit": "SPADE"}
          ]
        }
      ],
      "phase": "PREFLOP",
      "cardStack": [],
      "cardOpen": []
    },
    "pot": 15,
    "phase": "PREFLOP",
    "board": [],
    "toActIdx": 0,
    "hand": 1
  }
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Not your turn"
}
```

### Outgoing (Client → Server)

**Join:**
```json
{"type": "join"}
```

**Check/Call/Fold:**
```json
{
  "type": "act",
  "action": "CHECK"
}
```

**Raise:**
```json
{
  "type": "act",
  "action": "RAISE",
  "amount": 30
}
```

## Card Format Conversion

### Server Format → Model Format

| Server | Model |
|--------|-------|
| `{"rank": "2", "suit": "SPADE"}` | `[0, 0]` |
| `{"rank": "A", "suit": "HEART"}` | `[12, 1]` |
| `{"rank": "K", "suit": "DIAMOND"}` | `[11, 2]` |

**Rank Mapping:**
```python
'2'→0, '3'→1, '4'→2, '5'→3, '6'→4, '7'→5, '8'→6,
'9'→7, 'T'→8, 'J'→9, 'Q'→10, 'K'→11, 'A'→12
```

**Suit Mapping:**
```python
'SPADE'→0, 'HEART'→1, 'DIAMOND'→2, 'CLUB'→3
```

## Action Mapping

Your model outputs 7 actions:

| Model Output | Server Action | Description |
|-------------|---------------|-------------|
| 0 | (skip) | Padding |
| 1 | FOLD | Give up hand |
| 2 | CHECK | No bet (if allowed) |
| 3 | CALL | Match current bet |
| 4 | RAISE | Small raise (2x BB) |
| 5 | RAISE | Medium raise (3.5x BB) |
| 6 | RAISE | Large raise (5x BB) |

## Usage

### 1. Start the Go Server

```bash
cd path/to/Texas-HoldEm-Infrastructure
go run cmd/main.go
# Server listens on localhost:8080
```

### 2. Train Your Model (if not already trained)

```bash
python poker_training.py
# This creates a trained model in memory
# Optionally save: torch.save(model.state_dict(), 'poker_model.pth')
```

### 3. Run the Bot Client

```bash
python poker_client.py
```

### 4. Multiple Bots

Run in separate terminals:

```bash
# Terminal 1
python poker_client.py  # bot1

# Terminal 2
# Edit poker_client.py to change PLAYER_ID to "bot2"
python poker_client.py  # bot2
```

## Configuration

Edit `poker_client.py` at the bottom:

```python
API_KEY = "dev"           # Must match server's API_KEY
TABLE_ID = "table-1"      # Must match registered table
PLAYER_ID = "bot1"        # Unique per client
SERVER_HOST = "localhost"
SERVER_PORT = "8080"
MODEL_PATH = None         # Or "poker_model.pth"
```

## Key Integration Points

### 1. Turn Detection
```python
def _is_my_turn(self, state: Dict, player_id: str) -> bool:
    to_act_idx = state.get('toActIdx', -1)
    players = state.get('table', {}).get('players', [])
    return players[to_act_idx].get('id') == player_id
```

### 2. Card Extraction
```python
def _extract_my_cards(self, state: Dict, player_id: str):
    # Find your player in the players list
    # Extract 'cards' field
    # Convert to model format [rank, suit]
```

### 3. Action Decision
```python
def decide_action(self, state: Dict, player_id: str):
    # Extract cards
    # Run model inference
    # Convert model output to server format
    # Return action message
```

## Troubleshooting

### Bot doesn't act
- Check `toActIdx` matches your player index
- Verify player ID is correct
- Check server logs for errors

### Invalid action
- Ensure action is legal (CHECK only when no bet)
- Verify raise amount is sufficient
- Check player has enough chips

### Card format errors
- Verify rank/suit mapping is correct
- Check for None/null cards in state
- Handle missing board cards (pre-flop)

### Model not found
- Train model first or provide MODEL_PATH
- Ensure poker_model.py is in same directory
- Check import statements

## Testing Locally

### 1. Test with Random Actions

Modify `decide_action` to return random actions:

```python
def decide_action(self, state: Dict, player_id: str):
    import random
    actions = ["CHECK", "CALL", "FOLD"]
    return {"type": "act", "action": random.choice(actions)}
```

### 2. Test Card Parsing

```python
test_card = {"rank": "A", "suit": "HEART"}
parsed = bot._parse_card(test_card)
assert parsed == [12, 1], f"Expected [12, 1], got {parsed}"
```

### 3. Test Model Inference

```python
hole = torch.tensor([[[12, 1], [11, 0]]])  # AH KS
board = torch.zeros((1, 5, 2), dtype=torch.long)
logits, value = model(hole, board)
action = logits.argmax(dim=1).item()
print(f"Model chose action {action}")
```

## Performance Tips

### Model Inference
- Use `model.eval()` mode
- Disable gradients with `torch.no_grad()`
- Consider batching if playing multiple tables

### WebSocket
- Handle connection drops gracefully
- Implement reconnection logic
- Add message queueing for reliability

### Action Selection
- Add exploration (ε-greedy) for learning
- Consider pot odds in decision logic
- Validate actions before sending

## Next Steps

1. ✅ **Basic Integration** - Connect and play
2. **Advanced Features**:
   - Hand history tracking
   - Opponent modeling
   - Dynamic bet sizing
   - Multi-table play
3. **Training Improvements**:
   - Online learning from real games
   - Self-play against model copies
   - Reinforcement learning integration

## API Reference

### PokerBot Class

```python
bot = PokerBot(model_path="poker_model.pth")

# Main decision method
action_msg = bot.decide_action(state, player_id, bb=10)

# Helper methods
is_turn = bot._is_my_turn(state, player_id)
hole, board = bot._extract_my_cards(state, player_id)
legal = bot._get_legal_actions(state, player_id)
```

### WebSocket Client

```python
asyncio.run(play_poker(
    url="ws://localhost:8080/ws?apiKey=dev&table=table-1&player=bot1",
    player_id="bot1",
    model_path="poker_model.pth"
))
```

## Support

For issues:
1. Check server logs: `go run cmd/main.go`
2. Enable debug output in client
3. Verify WebSocket connection with browser tools
4. Test model independently before integration
