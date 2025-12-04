"""
Integration Testing Suite - Verify poker model works with game server
"""

import torch
import json
from poker_model_enhanced import SimplePokerNet, calculate_hand_strength
from poker_client_enhanced import EnhancedPokerBot


def test_card_parsing():
    """Test conversion from server format to model format."""
    print("\n" + "="*60)
    print("TEST 1: Card Parsing")
    print("="*60)
    
    bot = EnhancedPokerBot()
    
    test_cases = [
        ({"rank": "2", "suit": "SPADE"}, [0, 0]),
        ({"rank": "A", "suit": "HEART"}, [12, 1]),
        ({"rank": "K", "suit": "DIAMOND"}, [11, 2]),
        ({"rank": "T", "suit": "CLUB"}, [8, 3]),
        ({"rank": "7", "suit": "HEART"}, [5, 1]),
    ]
    
    passed = 0
    for server_card, expected in test_cases:
        result = bot._parse_card(server_card)
        if result == expected:
            print(f"✅ {server_card} → {result}")
            passed += 1
        else:
            print(f"❌ {server_card} → {result} (expected {expected})")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_card_extraction():
    """Test extracting cards from game state."""
    print("\n" + "="*60)
    print("TEST 2: Card Extraction from State")
    print("="*60)
    
    bot = EnhancedPokerBot()
    
    # Mock state
    state = {
        "table": {
            "players": [
                {
                    "id": "bot1",
                    "cards": [
                        {"rank": "A", "suit": "HEART"},
                        {"rank": "K", "suit": "SPADE"}
                    ]
                },
                {
                    "id": "bot2",
                    "cards": [
                        {"rank": "Q", "suit": "DIAMOND"},
                        {"rank": "J", "suit": "CLUB"}
                    ]
                }
            ]
        },
        "board": [
            {"rank": "T", "suit": "HEART"},
            {"rank": "9", "suit": "HEART"},
            {"rank": "8", "suit": "SPADE"}
        ]
    }
    
    hole, board = bot._extract_my_cards(state, "bot1")
    
    print(f"Hole cards: {hole}")
    print(f"Board cards: {board}")
    
    # Verify
    expected_hole = [[12, 1], [11, 0]]  # AH, KS
    expected_board_start = [[8, 1], [7, 1], [6, 0]]  # TH, 9H, 8S
    
    success = (hole == expected_hole and 
               board[:3] == expected_board_start and
               board[3] == [0, 0] and board[4] == [0, 0])
    
    if success:
        print("✅ Card extraction successful")
    else:
        print("❌ Card extraction failed")
    
    return success


def test_hand_strength():
    """Test hand strength calculation."""
    print("\n" + "="*60)
    print("TEST 3: Hand Strength Calculation")
    print("="*60)
    
    test_cases = [
        # (hole_cards, board_cards, min_strength, description)
        ([[12, 0], [12, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         0.6, "Pocket Aces preflop"),
        
        ([[1, 0], [2, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         0.0, "32 offsuit preflop"),
        
        ([[12, 0], [11, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
         0.3, "AK suited preflop"),
        
        ([[6, 0], [6, 1]], [[6, 2], [3, 0], [4, 1], [0, 0], [0, 0]], 
         0.65, "Set of 8s on flop"),
    ]
    
    passed = 0
    for hole, board, min_str, desc in test_cases:
        strength = calculate_hand_strength(hole, board)
        if strength >= min_str:
            print(f"✅ {desc}: {strength:.3f} (≥{min_str})")
            passed += 1
        else:
            print(f"❌ {desc}: {strength:.3f} (expected ≥{min_str})")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_model_inference():
    """Test model can make predictions."""
    print("\n" + "="*60)
    print("TEST 4: Model Inference")
    print("="*60)
    
    model = SimplePokerNet(embed_dim=64, hidden_dim=128, num_actions=7)
    model.eval()
    
    # Test inputs
    hole = torch.tensor([[[12, 1], [11, 0]]])  # AH KS
    board = torch.zeros((1, 5, 2), dtype=torch.long)
    
    try:
        with torch.no_grad():
            logits, value = model(hole, board)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Value shape: {value.shape}")
        print(f"Value: {value.item():.3f}")
        
        # Get action probabilities
        probs = torch.softmax(logits, dim=-1)
        action = probs.argmax(dim=1).item()
        confidence = probs[0, action].item()
        
        action_names = ['Pad', 'Fold', 'Check', 'Call', 'R-Small', 'R-Med', 'R-Large']
        print(f"Predicted action: {action_names[action]} ({confidence:.2%} confidence)")
        
        print("✅ Model inference successful")
        return True
    
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return False


def test_action_mapping():
    """Test converting model actions to server format."""
    print("\n" + "="*60)
    print("TEST 5: Action Mapping")
    print("="*60)
    
    bot = EnhancedPokerBot()
    
    # Mock state
    state = {
        "table": {"players": [{"id": "bot1", "chips": 1000}]},
        "phase": "PREFLOP"
    }
    
    test_cases = [
        (1, "FOLD", None),
        (2, "CHECK", None),
        (3, "CALL", None),
        (4, "RAISE", 20),  # 2x BB (assuming BB=10)
        (5, "RAISE", 35),  # 3.5x BB
        (6, "RAISE", 50),  # 5x BB
    ]
    
    passed = 0
    for model_action, expected_action, expected_amount in test_cases:
        msg = bot._map_action_to_server(model_action, state, "bot1", bb=10)
        
        action_ok = msg["action"] == expected_action
        amount_ok = (expected_amount is None or 
                    msg.get("amount") == expected_amount)
        
        if action_ok and amount_ok:
            print(f"✅ Action {model_action} → {msg}")
            passed += 1
        else:
            print(f"❌ Action {model_action} → {msg} (expected action={expected_action}, amount={expected_amount})")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_turn_detection():
    """Test detecting when it's our turn."""
    print("\n" + "="*60)
    print("TEST 6: Turn Detection")
    print("="*60)
    
    bot = EnhancedPokerBot()
    
    # Mock state - bot1's turn
    state1 = {
        "table": {
            "players": [
                {"id": "bot1", "chips": 1000},
                {"id": "bot2", "chips": 1000}
            ]
        },
        "toActIdx": 0
    }
    
    # Mock state - bot2's turn
    state2 = {
        "table": {
            "players": [
                {"id": "bot1", "chips": 1000},
                {"id": "bot2", "chips": 1000}
            ]
        },
        "toActIdx": 1
    }
    
    is_turn1 = bot._is_my_turn(state1, "bot1")
    is_turn2 = bot._is_my_turn(state2, "bot1")
    
    success = is_turn1 and not is_turn2
    
    print(f"State 1 (toActIdx=0): bot1's turn = {is_turn1} {'✅' if is_turn1 else '❌'}")
    print(f"State 2 (toActIdx=1): bot1's turn = {is_turn2} {'✅' if not is_turn2 else '❌'}")
    
    if success:
        print("✅ Turn detection working correctly")
    else:
        print("❌ Turn detection failed")
    
    return success


def test_full_decision_pipeline():
    """Test the complete decision-making pipeline."""
    print("\n" + "="*60)
    print("TEST 7: Full Decision Pipeline")
    print("="*60)
    
    bot = EnhancedPokerBot()
    
    # Realistic game state
    state = {
        "table": {
            "id": "table-1",
            "players": [
                {
                    "id": "bot1",
                    "chips": 950,
                    "action": "",
                    "cards": [
                        {"rank": "A", "suit": "HEART"},
                        {"rank": "K", "suit": "HEART"}
                    ]
                },
                {
                    "id": "bot2",
                    "chips": 1050,
                    "action": "CALL",
                    "cards": []  # Hidden from us
                }
            ],
            "phase": "FLOP"
        },
        "pot": 100,
        "phase": "FLOP",
        "board": [
            {"rank": "Q", "suit": "HEART"},
            {"rank": "J", "suit": "DIAMOND"},
            {"rank": "T", "suit": "HEART"}
        ],
        "toActIdx": 0,
        "hand": 3
    }
    
    try:
        action_msg = bot.decide_action(state, "bot1", bb=10)
        
        print(f"\nDecision: {json.dumps(action_msg, indent=2)}")
        
        # Verify message format
        has_type = "type" in action_msg
        has_action = "action" in action_msg
        valid_action = action_msg.get("action") in ["CHECK", "CALL", "RAISE", "FOLD"]
        
        success = has_type and has_action and valid_action
        
        if success:
            print("✅ Full pipeline working correctly")
        else:
            print("❌ Pipeline produced invalid message")
        
        return success
    
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("🧪 POKER MODEL INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Card Parsing", test_card_parsing),
        ("Card Extraction", test_card_extraction),
        ("Hand Strength", test_hand_strength),
        ("Model Inference", test_model_inference),
        ("Action Mapping", test_action_mapping),
        ("Turn Detection", test_turn_detection),
        ("Full Pipeline", test_full_decision_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Ready to connect to game server.")
        print("\nNext steps:")
        print("1. Start the Go server: go run cmd/main.go")
        print("2. Run the bot: python poker_client.py")
    else:
        print("\n⚠️  Some tests failed. Review errors above before connecting to server.")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
