# Complete Poker AI Package - File Index

## 📋 Quick Navigation

### 🚀 Start Here
1. **START_HERE.txt** - Critical context information (READ FIRST!)
2. **README.md** - Package overview and quick start
3. **quickstart.py** - Automated setup script

---

## 📁 Source Code (src/)

### Core Models
| File | Description | Use Case |
|------|-------------|----------|
| **poker_model.py** | Basic model (cards only) | Quick testing |
| **poker_model_enhanced.py** ⭐ | Enhanced model (with context) | Real gameplay |

**Key Difference**: Enhanced model sees chips, pot, phase, position, SPR

### WebSocket Clients
| File | Description | Use Case |
|------|-------------|----------|
| **poker_client.py** | Basic client | Testing only |
| **poker_client_enhanced.py** ⭐ | Enhanced client | Recommended! |

**Key Difference**: Enhanced client uses context-aware model for smarter decisions

---

## 🧪 Testing (tests/)

| File | Description | Tests |
|------|-------------|-------|
| **integration_test.py** | Complete test suite | 7 tests covering all functionality |

**Tests Included:**
1. Card parsing (server → model)
2. Card extraction from state
3. Hand strength calculation
4. Model inference
5. Action mapping (model → server)
6. Turn detection
7. Full decision pipeline

---

## 📚 Documentation (docs/)

### Must Read (Priority Order)
1. **CONTEXT_AWARENESS.md** ⭐ - Why enhanced model matters
2. **QUICK_START.txt** - Visual quick start guide
3. **CHANGES_SUMMARY.txt** - What changed with engine.go

### Technical Reference
4. **ENGINE_UPDATES.md** - Server validation rules and requirements
5. **INTEGRATION_GUIDE.md** - Deep technical dive
6. **README.md** - Package overview
7. **SUMMARY.md** - Complete package summary

### Context Awareness Explanation
**CONTEXT_AWARENESS.md** answers the critical question:
> "Can my model know how many chips it has, what round we're in, whether they are small or big blind?"

**Answer**: Basic model = NO ❌ | Enhanced model = YES ✅

---

## 💡 Examples (examples/)

| File | Purpose | Demonstrates |
|------|---------|--------------|
| **run_multiple_bots.py** | Run 2+ bots simultaneously | Multi-bot gameplay |
| **test_hand_strength.py** | Test hand evaluation | Understanding hand strength |

### How to Use Examples
```bash
# Run multiple bots
python examples/run_multiple_bots.py

# Test hand strength
python examples/test_hand_strength.py
```

---

## 🎓 Training (training/)

| File | Purpose | Demonstrates |
|------|---------|--------------|
| **train_enhanced_model.py** | Training script example | How to train with context |

**Note**: This uses mock data for demonstration. In practice:
1. Generate real poker game data
2. Extract context features from game states
3. Label actions based on outcomes
4. Train on real data

---

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| **requirements.txt** | Python dependencies |

**Dependencies:**
- torch >= 2.0.0
- websockets >= 12.0
- open_spiel >= 1.2.0

---

## 📊 File Sizes

```
Total: ~118 KB

Documentation:    ~50 KB (7 files)
Source Code:      ~48 KB (4 files)
Tests:            ~11 KB (1 file)
Examples:         ~9 KB (3 files)
```

---

## 🎯 Recommended Reading Order

### For Quick Start (5 min)
1. START_HERE.txt
2. README.md
3. Run: `python tests/integration_test.py`
4. Run: `python src/poker_client_enhanced.py`

### For Understanding (15 min)
1. START_HERE.txt
2. docs/CONTEXT_AWARENESS.md
3. docs/QUICK_START.txt
4. docs/CHANGES_SUMMARY.txt

### For Development (30 min)
1. START_HERE.txt
2. docs/CONTEXT_AWARENESS.md
3. docs/INTEGRATION_GUIDE.md
4. docs/ENGINE_UPDATES.md
5. src/poker_model_enhanced.py
6. src/poker_client_enhanced.py

### For Complete Mastery (60 min)
Read everything in order:
1. START_HERE.txt ⭐
2. docs/CONTEXT_AWARENESS.md ⭐
3. docs/QUICK_START.txt
4. docs/CHANGES_SUMMARY.txt
5. docs/ENGINE_UPDATES.md
6. docs/INTEGRATION_GUIDE.md
7. docs/README.md
8. docs/SUMMARY.md
9. Source code walkthrough
10. Examples

---

## 🔍 Finding Specific Information

### "How do I get started?"
→ START_HERE.txt, README.md

### "Why do I need the enhanced model?"
→ docs/CONTEXT_AWARENESS.md

### "What changed with the engine?"
→ docs/CHANGES_SUMMARY.txt, docs/ENGINE_UPDATES.md

### "How does the WebSocket protocol work?"
→ docs/INTEGRATION_GUIDE.md

### "How do I train the model?"
→ training/train_enhanced_model.py

### "How do I run multiple bots?"
→ examples/run_multiple_bots.py

### "How is hand strength calculated?"
→ examples/test_hand_strength.py

### "What tests are included?"
→ tests/integration_test.py

### "How do I configure the client?"
→ Edit src/poker_client_enhanced.py (bottom of file)

---

## 📈 Feature Matrix

| Feature | Basic | Enhanced |
|---------|-------|----------|
| Card input | ✅ | ✅ |
| Chip awareness | ❌ | ✅ |
| Pot size awareness | ❌ | ✅ |
| Phase awareness | ❌ | ✅ |
| Position awareness | ❌ | ✅ |
| SPR calculation | ❌ | ✅ |
| Server validation | ✅ | ✅ |
| Turn detection | ✅ | ✅ |
| Action conversion | ✅ | ✅ |
| All-in handling | ✅ | ✅ |
| Integration tests | ✅ | ✅ |

---

## 🎮 Usage Patterns

### Quick Test
```bash
python tests/integration_test.py
python src/poker_client.py
```

### Recommended Play
```bash
python tests/integration_test.py
python src/poker_client_enhanced.py
```

### Training
```bash
python training/train_enhanced_model.py
python src/poker_client_enhanced.py --model enhanced_model_best.pth
```

### Multi-Bot Testing
```bash
python examples/run_multiple_bots.py
```

---

## 🐛 Troubleshooting Guide

| Issue | Solution | Documentation |
|-------|----------|---------------|
| Connection refused | Start Go server | README.md |
| Unauthorized | Check API_KEY | README.md |
| Tests fail | Check dependencies | README.md |
| Bot doesn't act | Check turn detection | docs/ENGINE_UPDATES.md |
| Invalid action | Check validation | docs/ENGINE_UPDATES.md |
| Need context | Use enhanced model | docs/CONTEXT_AWARENESS.md |

---

## 💡 Key Insights

1. **Basic vs Enhanced**: Enhanced model is MUCH better (sees context)
2. **Context Matters**: Chips, pot, phase, position are critical
3. **Server Validation**: Strict CHECK vs CALL rules
4. **Integer Betting**: All bets are whole numbers
5. **Turn Detection**: Reliable via toActIdx
6. **Testing First**: Run integration_test.py before server connection

---

## 🎯 Success Criteria

You're ready when:
- ✅ Read START_HERE.txt
- ✅ Understand why enhanced model matters
- ✅ All 7 tests pass
- ✅ Bot connects to server
- ✅ Bot acts when it's its turn
- ✅ Actions are accepted by server

---

## 📞 Getting Help

1. Check START_HERE.txt
2. Read docs/CONTEXT_AWARENESS.md
3. Review relevant documentation from list above
4. Run integration tests to verify setup
5. Check server logs for errors

---

## 🎉 You Have Everything!

This package includes:
- ✅ 2 model versions (basic + enhanced)
- ✅ 2 client versions (basic + enhanced)
- ✅ 7 documentation files
- ✅ 7 integration tests
- ✅ 3 example scripts
- ✅ 1 training example
- ✅ Complete setup guides

**Total: 15 files, ~118 KB, production-ready!**

---

**Start with**: `START_HERE.txt` → Then choose your path based on goals!
