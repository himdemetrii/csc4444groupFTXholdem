# Poker AI → Game Server Integration

This package connects your trained PyTorch poker model to the Go WebSocket game server.

## 📁 Files

| File | Purpose |
|------|---------|
| `poker_model.py` | Neural network model (importable) |
| `poker_client.py` | WebSocket client that plays poker |
| `integration_test.py` | Test suite to verify everything works |
| `INTEGRATION_GUIDE.md` | Detailed documentation |
| `quickstart.py` | Quick setup script |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Integration (No Server Required)

```bash
python integration_test.py
```

This verifies:
- ✅ Card format conversion
- ✅ Model inference
- ✅ Action mapping
- ✅ Turn detection
- ✅ Full decision pipeline

### 3. Start the Game Server

```bash
cd path/to/Texas-HoldEm-Infrastructure
go run cmd/main.go
```

Server starts on `localhost:8080`

### 4. Connect Your Bot

```bash
python poker_client.py
```

## 🎮 How It Works

```
┌─────────────┐         WebSocket         ┌─────────────┐
│  Go Server  │ ←─────────────────────→  │ Python Bot  │
│             │                           │             │
│ • Dealing   │  {"type": "state", ...}   │ • Model     │
│ • Betting   │  ─────────────────────→   │ • Decision  │
│ • Pot mgmt  │                           │ • Action    │
│ • Hand eval │  {"type": "act", ...}     │             │
│             │  ←─────────────────────   │             │
└─────────────┘                           └─────────────┘
```

### Message Flow

1. **Server → Bot**: Game state update
   ```json
   {
     "type": "state",
     "state": {
       "table": {...},
       "pot": 150,
       "phase": "FLOP",
       "toActIdx": 0
     }
   }
   ```

2. **Bot**: Process state
   - Extract hole cards and board
   - Convert to model format
   - Run inference
   - Select action

3. **Bot → Server**: Action
   ```json
   {
     "type": "act",
     "action": "RAISE",
     "amount": 30
   }
   ```

## 🔧 Configuration

Edit `poker_client.py`:

```python
API_KEY = "dev"           # Must match server
TABLE_ID = "table-1"      # Must exist on server
PLAYER_ID = "bot1"        # Unique identifier
SERVER_HOST = "localhost"
SERVER_PORT = "8080"
MODEL_PATH = None         # Path to saved weights
```

## 📊 Model Details

### Input Format
- **Hole cards**: `[[rank, suit], [rank, suit]]`
- **Board cards**: `[[rank, suit], [rank, suit], ...]` (5 cards, padded with [0,0])

### Encoding
- Ranks: `2→0, 3→1, ..., K→11, A→12`
- Suits: `♠→0, ♥→1, ♦→2, ♣→3`

### Output
- **7 actions**: Pad, Fold, Check, Call, Raise-Small, Raise-Med, Raise-Large
- **Value**: Expected return (-1 to +1)

## 🎯 Key Features

### Card Format Conversion
Automatically converts between server format and model format:
```python
# Server: {"rank": "A", "suit": "HEART"}
# Model:  [12, 1]
```

### Legal Action Filtering
Ensures model actions are valid:
- CHECK only when no bet required
- CALL when bet required
- RAISE with appropriate amounts

### Turn Detection
Only acts when it's your turn:
```python
if bot._is_my_turn(state, player_id):
    action = bot.decide_action(state, player_id)
```

## 🧪 Testing

### Unit Tests
```bash
python integration_test.py
```

### Manual Testing
```bash
# Test with random actions
python poker_client.py --random

# Test specific scenarios
python -c "from poker_client import PokerBot; bot = PokerBot(); ..."
```

## 🐛 Troubleshooting

### Bot doesn't connect
- ✓ Server is running
- ✓ API_KEY matches
- ✓ Port is correct (8080)

### Bot doesn't act
- ✓ Check server logs
- ✓ Verify player ID
- ✓ Check `toActIdx`

### Invalid actions
- ✓ Action is legal (CHECK vs CALL)
- ✓ Raise amount is sufficient
- ✓ Player has chips

### Card errors
- ✓ Rank/suit mapping is correct
- ✓ Handle pre-flop (no board cards)
- ✓ Check for None/null values

## 📚 Documentation

- **INTEGRATION_GUIDE.md** - Comprehensive guide
- **Server README** - Go server documentation
- **Code comments** - Inline documentation

## 🔄 Workflow

### Development
1. Train model (your existing code)
2. Test integration (`integration_test.py`)
3. Connect to server (`poker_client.py`)

### Deployment
1. Save model weights: `torch.save(model.state_dict(), 'model.pth')`
2. Set MODEL_PATH in config
3. Run multiple bots for testing

### Iteration
1. Collect hand history
2. Retrain model
3. A/B test new vs old model

## 🎓 Learning Resources

### Understanding the Code
```python
# poker_model.py - Model architecture
class SimplePokerNet(nn.Module):
    # Card encoder + decision network

# poker_client.py - Game integration
class PokerBot:
    def decide_action(self, state, player_id):
        # Main decision loop
```

### Key Concepts
- **WebSocket protocol**: Bidirectional real-time communication
- **State synchronization**: Server broadcasts game state
- **Action validation**: Ensure legal moves
- **Card encoding**: Convert cards to neural net input

## 🚦 Status Indicators

When running, you'll see:
- `🤖 Poker Bot Starting` - Initialization
- `✅ Connected` - WebSocket established
- `📊 State Update` - Game state received
- `🎯 It's our turn!` - Decision time
- `✉️ Sent action` - Action submitted

## 🎮 Multiple Bots

To run 2+ bots:

```bash
# Terminal 1
python poker_client.py

# Terminal 2
# Edit poker_client.py: PLAYER_ID = "bot2"
python poker_client.py

# Terminal 3
# Edit poker_client.py: PLAYER_ID = "bot3"
python poker_client.py
```

Or create separate config files:

```bash
python poker_client.py --config bot1.json
python poker_client.py --config bot2.json
```

## 📈 Performance

- **Inference time**: ~5ms per decision
- **Memory**: ~100MB per bot
- **Latency**: <50ms (local network)

## 🔮 Future Enhancements

Potential improvements:
- [ ] Online learning from game results
- [ ] Opponent modeling
- [ ] Multi-table support
- [ ] Hand history analysis
- [ ] Tournament mode
- [ ] Bankroll management

## 🤝 Contributing

To improve the integration:
1. Test with different scenarios
2. Report issues with game logs
3. Suggest features
4. Optimize performance

## ⚖️ License

Same as the main poker model project.

## 📞 Support

For integration issues:
1. Run `integration_test.py` first
2. Check server logs
3. Review INTEGRATION_GUIDE.md
4. Verify configuration

---

**Ready to play?**

```bash
python quickstart.py
```

This will run all tests and show you how to connect!
